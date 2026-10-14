from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from src.config import Settings
from src.utils.networking import configure_ssl_from_env
from src.utils.retry import invoke_with_retry

from ...llm.prompts import (
    GRADE_PROMPT,
    TRUMP_AGENT_PERSONA,
    TRUMP_RAG_PERSONA,
    agent_system_prompt,
    rag_prompt,
)
from ...llm.provider import build_chat_model, build_structured_chat_model
from ...llm.sanitize import strip_citation_artifacts

logger = logging.getLogger(__name__)

QuestionResolver = Callable[[dict[str, Any]], str]
NodeCallable = Callable[[dict[str, Any]], dict[str, Any]]
GradeCallable = Callable[[dict[str, Any]], Literal["generate", "rewrite"]]


class Grade(BaseModel):
    """Structured relevance verdict produced by the grading conditional edge."""

    binary_score: str = Field(description="Relevance score: 'yes' or 'no'")
    explanation: str | None = Field(None, description="Optional short explanation")


configure_ssl_from_env()


def new_chat_model(settings: Settings) -> Any:
    """Create a DashScope chat model with project-level network settings."""

    return build_chat_model(settings)


def new_structured_chat_model(settings: Settings, schema: Any) -> Any:
    """Create a chat model bound to a structured-output schema.

    Routes through the provider seam so DeepSeek uses tool-free JSON mode
    instead of unsupported JSON-schema or forced-tool response formats.
    """

    return build_structured_chat_model(settings, schema)


def cached_builder(factory: Callable[[], Any]) -> Callable[[], Any]:
    """Return a zero-argument builder that constructs its value once.

    Graph nodes used to rebuild their chat model (and structured-output
    schema binding) on every turn; wrapping the constructor in this helper
    inside a node factory makes one node reuse a single client across
    turns. The factory callable is resolved at first call, so tests can
    still monkeypatch ``new_chat_model`` / ``new_structured_chat_model``.
    """

    value: list[Any] = []

    def build() -> Any:
        if not value:
            value.append(factory())
        return value[0]

    return build


def _persona_blocks(settings: Settings) -> tuple[str, str]:
    """Resolve persona style blocks for the agent and RAG prompts."""

    style = str(getattr(settings, "agent_persona_style", "trump") or "trump").strip().lower()
    if style == "none":
        return "", ""
    return TRUMP_AGENT_PERSONA, TRUMP_RAG_PERSONA


def message_text(message: Any) -> str:
    if hasattr(message, "content"):
        return str(message.content)
    if isinstance(message, (tuple, list)) and len(message) >= 2:
        return str(message[1])
    return str(message)


def bounded_chat_messages(
    messages: Sequence[Any],
    *,
    max_turns: int,
    max_chars: int,
) -> list[Any]:
    """Return a bounded, read-only projection of a chat transcript.

    Checkpoint state remains the source of truth for the history API. This
    helper only selects messages for an LLM invocation, removes historical
    tool protocol messages, and copies a message before truncating its text.
    The most recent user message is always retained.
    """

    values = list(messages or [])
    if not values:
        return []

    turn_limit = max(1, int(max_turns))
    char_limit = max(1, int(max_chars))
    human_indices = [
        index for index, message in enumerate(values) if _message_role(message) in ("human", "user")
    ]
    if not human_indices:
        visible = [message for message in values if _is_chat_context_message(message)]
        return _bounded_messages_without_turns(visible, char_limit)

    first_turn_index = human_indices[-min(turn_limit, len(human_indices))]
    while first_turn_index > 0 and _message_role(values[first_turn_index - 1]) == "system":
        first_turn_index -= 1
    recent = values[first_turn_index:]

    latest_human = max(
        index for index, message in enumerate(recent) if _message_role(message) in ("human", "user")
    )
    current_start = latest_human
    while current_start > 0 and _message_role(recent[current_start - 1]) == "system":
        current_start -= 1

    # Preserve the complete current-turn message protocol. In particular, an
    # AI tool-call carrier and its matching ToolMessage must reach the agent
    # when the graph re-enters it to synthesize an auxiliary tool result.
    current_messages = recent[current_start:]
    current_human = latest_human - current_start
    bounded_current = _bounded_current_turn(
        current_messages,
        human_index=current_human,
        max_chars=char_limit,
    )
    selected: dict[int, Any] = {
        current_start + index: message for index, message in enumerate(bounded_current)
    }
    remaining = char_limit - sum(len(message_text(message)) for message in bounded_current)

    # Add older turns only when each complete turn fits, avoiding an orphaned
    # assistant answer at the beginning of the projected conversation.
    older = [message for message in recent[:current_start] if _is_chat_context_message(message)]
    older_humans = [
        index for index, message in enumerate(older) if _message_role(message) in ("human", "user")
    ]
    turn_starts: list[int] = []
    for human_index in older_humans:
        start = human_index
        while start > 0 and _message_role(older[start - 1]) == "system":
            start -= 1
        turn_starts.append(start)

    turn_ranges: list[tuple[int, int]] = []
    for position, start in enumerate(turn_starts):
        end = turn_starts[position + 1] if position + 1 < len(turn_starts) else len(older)
        turn_ranges.append((start, end))

    for start, end in reversed(turn_ranges):
        turn_chars = sum(len(message_text(message)) for message in older[start:end])
        if turn_chars > remaining:
            continue
        for index in range(start, end):
            selected[index] = older[index]
        remaining -= turn_chars

    return [selected[index] for index in sorted(selected)]


def _bounded_current_turn(
    messages: list[Any],
    *,
    human_index: int,
    max_chars: int,
) -> list[Any]:
    """Fit current-turn content while retaining every protocol message."""

    contents = [message_text(message) for message in messages]
    if sum(len(content) for content in contents) <= max_chars:
        return list(messages)

    allocations = [0 for _message in messages]
    other_chars = sum(
        len(content) for index, content in enumerate(contents) if index != human_index
    )
    human_limit = max_chars if other_chars == 0 else max(1, max_chars // 2)
    allocations[human_index] = min(len(contents[human_index]), human_limit)
    remaining = max_chars - allocations[human_index]

    for index in range(len(messages) - 1, -1, -1):
        if index == human_index or remaining <= 0:
            continue
        allocations[index] = min(len(contents[index]), remaining)
        remaining -= allocations[index]

    if remaining > 0:
        extra_human = min(
            len(contents[human_index]) - allocations[human_index],
            remaining,
        )
        allocations[human_index] += extra_human

    bounded: list[Any] = []
    for message, content, allocation in zip(
        messages,
        contents,
        allocations,
        strict=False,
    ):
        bounded_content = _truncate_context_text(content, allocation) if allocation else ""
        bounded.append(
            message
            if bounded_content == content
            else _copy_message_with_content(message, bounded_content)
        )
    return bounded


def _message_role(message: Any) -> str:
    role = getattr(message, "type", None) or getattr(message, "role", None)
    if role:
        return str(role).lower()
    if isinstance(message, (tuple, list)) and message:
        return str(message[0]).lower()
    return str(message.__class__.__name__).lower()


def _is_chat_context_message(message: Any) -> bool:
    role = _message_role(message)
    if role in ("tool", "function", "toolmessage", "functionmessage"):
        return False
    if role in ("ai", "assistant", "aimessage") and getattr(message, "tool_calls", None):
        return False
    return bool(message_text(message).strip())


def _bounded_messages_without_turns(messages: list[Any], max_chars: int) -> list[Any]:
    selected: list[Any] = []
    remaining = max_chars
    for message in reversed(messages):
        text = message_text(message)
        if len(text) <= remaining:
            selected.append(message)
            remaining -= len(text)
        elif not selected:
            selected.append(
                _copy_message_with_content(
                    message,
                    _truncate_context_text(text, remaining),
                )
            )
            remaining = 0
        if remaining <= 0:
            break
    return list(reversed(selected))


def _truncate_context_text(text: str, max_chars: int) -> str:
    limit = max(1, int(max_chars))
    if len(text) <= limit:
        return text
    marker = "\n...[truncated]...\n"
    if limit <= len(marker) + 2:
        return text[:limit]
    available = limit - len(marker)
    head = (available * 2) // 3
    return f"{text[:head]}{marker}{text[-(available - head) :]}"


def _copy_message_with_content(message: Any, content: str) -> Any:
    if isinstance(message, BaseMessage):
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"content": content})
        return message.copy(update={"content": content})
    if isinstance(message, tuple) and len(message) >= 2:
        return (message[0], content, *message[2:])
    if isinstance(message, list) and len(message) >= 2:
        return [message[0], content, *message[2:]]
    return message


def _sanitized_ai_message(message: Any) -> Any:
    """Return the message with fabricated citation markers removed.

    Only plain string content is rewritten. Tool-call carriers and structured
    content blocks are returned unchanged so the tool protocol is preserved.
    """

    content = getattr(message, "content", None)
    if not isinstance(content, str) or getattr(message, "tool_calls", None):
        return message
    cleaned = strip_citation_artifacts(content)
    if cleaned == content:
        return message
    return _copy_message_with_content(message, cleaned)


def chat_question_resolver(state: dict[str, Any]) -> str:
    standalone = (state.get("current_question") or "").strip()
    if standalone:
        return standalone

    messages = state.get("messages") or []
    idx = int(state.get("current_question_index", -1) or -1)
    if 0 <= idx < len(messages):
        return message_text(messages[idx])

    if messages:
        return message_text(messages[-1])
    return ""


def _question_tokens(question: str) -> set[str]:
    stopwords = {
        "about",
        "after",
        "article",
        "author",
        "based",
        "does",
        "from",
        "provided",
        "query",
        "say",
        "says",
        "that",
        "the",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", question.lower())
        if len(token) > 2 and token not in stopwords
    }


# REFACTOR: Deterministically rerank retrieved chunks before grading/generation.
def rerank_retrieved_context(
    question: str,
    retrieved_message: Any,
    settings: Settings | None = None,
) -> str:
    """Return retrieved context ordered by the configured relevance strategy."""

    content: str = message_text(retrieved_message)
    document_chunks: list[str] = _document_chunks_from_message(retrieved_message)
    chunks: list[str] = document_chunks or _split_retrieved_chunks(content)
    if len(chunks) <= 1:
        return content

    scores = _rerank_scores(question, chunks, settings)
    return "\n\n".join(chunks[index] for index in _ranked_indices(scores))


def build_sourced_context(
    question: str,
    retrieved_message: Any,
    settings: Settings | None = None,
) -> tuple[str, list[str]]:
    """Return reranked context with numbered source headers, plus the URL list.

    Each chunk from a traceable document is prefixed with a
    ``[n] Source: <url>`` header so the generate prompt can require [n]
    citations that are verifiable against the returned URL list. Falls back to
    the legacy plain-text context (and an empty URL list) when citations are
    disabled or the tool message carries no document artifacts.
    """

    if settings is not None and not getattr(settings, "rag_citations_enabled", True):
        return rerank_retrieved_context(question, retrieved_message, settings), []

    pairs = _document_pairs_from_message(retrieved_message)
    if not pairs:
        return rerank_retrieved_context(question, retrieved_message, settings), []

    if len(pairs) > 1:
        scores = _rerank_scores(question, [text for text, _ in pairs], settings)
        pairs = [pairs[index] for index in _ranked_indices(scores)]

    numbering: dict[str, int] = {}
    urls: list[str] = []
    blocks: list[str] = []
    for text, url in pairs:
        if url:
            if url not in numbering:
                numbering[url] = len(urls) + 1
                urls.append(url)
            blocks.append(f"[{numbering[url]}] Source: {url}\n{text}")
        else:
            blocks.append(text)
    return "\n\n".join(blocks), urls


# Citation markers the rag prompt asks the model to emit. The lookbehind keeps
# ASCII word characters adjacent (so ``array[3]`` in code is left alone) while
# still allowing markers right after spaces, punctuation, or CJK text.
_CITATION_MARKER_RE = re.compile(r"(?<![A-Za-z0-9_])[\[【]\s*(\d{1,3})\s*[\]】]")


def validate_citations(answer: str, source_urls: list[str]) -> tuple[str, list[str]]:
    """Strip citation markers that point past the retrieved source list.

    Keeps ``[n]`` markers that address ``source_urls`` (normalizing full-width
    brackets to ASCII), removes the rest, and appends a ``Sources:`` footer
    mapping each surviving number to its URL. Returns ``(text, cited_urls)``.
    """

    if not source_urls:
        return answer, []

    upper = len(source_urls)

    def _keep(match: Any) -> str:
        number = int(match.group(1))
        return f"[{number}]" if 1 <= number <= upper else ""

    cleaned = _CITATION_MARKER_RE.sub(_keep, answer)
    # Tidy the gaps stripped markers leave behind.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+([.,;:!?，。；：！？、）】\]])", r"\1", cleaned)

    cited: list[int] = []
    for match in _CITATION_MARKER_RE.finditer(cleaned):
        number = int(match.group(1))
        if number not in cited:
            cited.append(number)

    if not cited:
        return cleaned, []

    footer = "\n\nSources:\n" + "\n".join(
        f"[{number}] {source_urls[number - 1]}" for number in cited
    )
    return cleaned.rstrip() + footer, [source_urls[number - 1] for number in cited]


def _rerank_strategy(settings: Settings | None) -> str:
    if settings is None:
        return "lexical"
    strategy = settings.rerank_strategy.strip().lower()
    if strategy not in ("lexical", "embedding", "hybrid", "cross_encoder"):
        logger.warning("Unknown rerank strategy %r; using lexical", strategy)
        return "lexical"
    return strategy


def _rerank_scores(
    question: str,
    chunks: list[str],
    settings: Settings | None,
) -> list[float]:
    strategy = _rerank_strategy(settings)
    if strategy == "embedding":
        return _embedding_scores(question, chunks, settings)
    if strategy == "hybrid":
        return _hybrid_scores(question, chunks, settings)
    if strategy == "cross_encoder":
        return _cross_encoder_scores(question, chunks, settings)
    return _lexical_scores(question, chunks)


def _cross_encoder_scores(
    question: str,
    chunks: list[str],
    settings: Settings | None,
) -> list[float]:
    """Cross-encoder relevance via the DashScope TextReRank API.

    Chunks past ``rerank_max_chunks`` keep lexical order after the reranked
    head so one giant context cannot turn into an unbounded rerank call. Any
    API/shape failure falls back to lexical scores, mirroring the
    embedding-strategy fallback.
    """

    lexical = _lexical_scores(question, chunks)
    if settings is None or not chunks:
        return lexical

    max_chunks = max(1, int(getattr(settings, "rerank_max_chunks", 32)))
    head = chunks[:max_chunks]
    tail_count = len(chunks) - len(head)
    if not head:
        return lexical

    try:
        from dashscope import TextReRank

        from src.utils.retry import dashscope_call_with_retry

        response = dashscope_call_with_retry(
            TextReRank,
            {
                "model": getattr(settings, "rerank_model", "gte-rerank-v2"),
                "query": question,
                "documents": head,
                "top_n": len(head),
                "api_key": settings.dashscope_api_key,
                "return_documents": False,
            },
        )
        ranked = (response.output or {}).get("results") or []
    except Exception as exc:
        logger.warning("Cross-encoder rerank failed; using lexical rerank: %s", exc)
        return lexical

    # Scores for the reranked head; the tail keeps its lexical score minus a
    # constant so it never outranks a reranked head entry.
    scores = [-1.0] * len(chunks)
    for position, entry in enumerate(ranked, 1):
        index = int(entry.get("index", -1))
        if 0 <= index < len(head):
            scores[index] = float(len(ranked) - position + 1)
    for index in range(len(head)):
        if scores[index] < 0.0:
            scores[index] = 0.0
    if tail_count:
        lexical_head_min = min(scores[: len(head)]) if head else 0.0
        for offset, lexical_score in enumerate(lexical[len(head) :]):
            scores[len(head) + offset] = min(lexical_score, lexical_head_min) - 1.0
    return scores


def _document_chunks_from_message(message: Any) -> list[str]:
    artifact = getattr(message, "artifact", None)
    if not isinstance(artifact, list):
        return []

    chunks: list[str] = []
    for document in artifact:
        text = _document_text(document).strip()
        if text:
            chunks.append(text)
    return chunks


def _document_text(document: Any) -> str:
    page_content = getattr(document, "page_content", None)
    if isinstance(page_content, str):
        return page_content
    if isinstance(document, dict):
        for key in ("page_content", "content", "text"):
            value = document.get(key)
            if isinstance(value, str):
                return value
    return str(document)


def _split_retrieved_chunks(content: str) -> list[str]:
    chunks: list[str] = [chunk.strip() for chunk in re.split(r"\n\s*\n+", content or "")]
    return [chunk for chunk in chunks if chunk]


def _document_pairs_from_message(message: Any) -> list[tuple[str, str]]:
    """Return ``(text, source_url)`` pairs from a tool message artifact.

    Artifacts are Documents on the direct path and plain dicts after the MCP
    pipeline sanitizer (which preserves page_content/metadata structure); both
    shapes are handled here.
    """

    artifact = getattr(message, "artifact", None)
    if not isinstance(artifact, list):
        return []

    pairs: list[tuple[str, str]] = []
    for document in artifact:
        text = _document_text(document).strip()
        if not text:
            continue
        pairs.append((text, _document_source(document)))
    return pairs


def _document_source(document: Any) -> str:
    metadata = getattr(document, "metadata", None)
    if not isinstance(metadata, dict) and isinstance(document, dict):
        metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    for key in ("source", "url"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _ranked_indices(scores: list[float]) -> list[int]:
    """Chunk indices ordered by descending score, ties keeping original order."""

    return [
        index for index, _score in sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))
    ]


def _lexical_scores(question: str, chunks: list[str]) -> list[float]:
    query_tokens: set[str] = _question_tokens(question)
    query_phrases: set[str] = _query_phrases(question)
    return [float(_lexical_relevance_score(query_tokens, query_phrases, chunk)) for chunk in chunks]


def _embedding_scores(
    question: str,
    chunks: list[str],
    settings: Settings | None,
) -> list[float]:
    if settings is None:
        return _lexical_scores(question, chunks)
    try:
        return _embedding_relevance_scores(question, chunks, settings)
    except Exception as exc:
        logger.warning("Embedding rerank failed; using lexical rerank: %s", exc)
        return _lexical_scores(question, chunks)


def _hybrid_scores(
    question: str,
    chunks: list[str],
    settings: Settings | None,
) -> list[float]:
    lexical_scores = _lexical_scores(question, chunks)
    if settings is None:
        return lexical_scores
    try:
        embedding_scores = _embedding_relevance_scores(question, chunks, settings)
    except Exception as exc:
        logger.warning(
            "Hybrid rerank embedding score failed; using lexical rerank: %s",
            exc,
        )
        return lexical_scores
    return [
        lexical + embedding
        for lexical, embedding in zip(
            _normalize_scores(lexical_scores),
            _normalize_scores(embedding_scores),
            strict=False,
        )
    ]


def _embedding_relevance_scores(
    question: str,
    chunks: list[str],
    settings: Settings,
) -> list[float]:
    embeddings = _build_rerank_embeddings(settings)
    query_vector = embeddings.embed_query(question)
    chunk_vectors = embeddings.embed_documents(chunks)
    if len(chunk_vectors) != len(chunks):
        raise RuntimeError("Embedding provider returned the wrong number of chunk vectors.")
    return [_cosine_similarity(query_vector, chunk_vector) for chunk_vector in chunk_vectors]


def _build_rerank_embeddings(settings: Settings) -> Any:
    from ...rag.embeddings import build_embeddings

    return build_embeddings(settings)


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lowest = min(scores)
    highest = max(scores)
    if highest == lowest:
        return [0.0 for _score in scores]
    return [(score - lowest) / (highest - lowest) for score in scores]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot_product = sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=False)
    )
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(dot_product / (left_norm * right_norm))


def _query_phrases(question: str) -> set[str]:
    tokens: list[str] = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", question.lower())
    return {
        " ".join(tokens[index : index + size])
        for size in (2, 3)
        for index in range(0, max(0, len(tokens) - size + 1))
    }


def _lexical_relevance_score(
    query_tokens: set[str],
    query_phrases: set[str],
    chunk: str,
) -> int:
    lower: str = chunk.lower()
    chunk_tokens: list[str] = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", lower)
    chunk_token_set: set[str] = set(chunk_tokens)
    score: int = sum(1 for token in query_tokens if token in chunk_token_set)
    score += sum(chunk_tokens.count(token) for token in query_tokens)
    score += 3 * sum(1 for phrase in query_phrases if phrase in lower)
    return score


def _split_context_sentences(context: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", context).strip()
    if not normalized:
        return []

    sentences: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+", normalized):
        text = part.strip()
        if len(text) < 40:
            continue
        if len(text) > 700:
            chunks = re.split(r";\s+|,\s+(?=[A-Z])", text)
            sentences.extend(chunk.strip() for chunk in chunks if len(chunk.strip()) >= 40)
        else:
            sentences.append(text)
    return sentences


def build_extractive_answer(question: str, context: str) -> str:
    """Create a best-effort answer when the chat model is unreachable."""

    context = context or ""
    sentences = _split_context_sentences(context)
    if not sentences:
        return (
            "I could not reach DashScope to synthesize a final answer, and the "
            "retriever did not return usable article text."
        )

    tokens = _question_tokens(question)
    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        lower = sentence.lower()
        score = sum(1 for token in tokens if token in lower)
        if score > 0:
            scored.append((score, index, sentence))

    if scored:
        top = sorted(scored, key=lambda item: (-item[0], item[1]))[:5]
        selected = [sentence for _, _, sentence in sorted(top, key=lambda item: item[1])]
    else:
        selected = sentences[:5]

    lines = []
    total_chars = 0
    for sentence in selected:
        remaining = 1800 - total_chars
        if remaining <= 0:
            break
        snippet = sentence[:remaining].strip()
        if snippet:
            lines.append(f"- {snippet}")
            total_chars += len(snippet)

    return (
        "I could not reach DashScope to synthesize the final answer, so here is "
        "an extractive answer from the retrieved article context:\n\n" + "\n".join(lines)
    )


def grade_documents_factory(
    settings: Settings,
    question_resolver: QuestionResolver = chat_question_resolver,
) -> GradeCallable:
    """Return a conditional edge function that grades retrieved context."""

    # Built once per graph instead of on every grading call; the pydantic
    # schema class is module-level for the same reason.
    structured_llm = cached_builder(lambda: new_structured_chat_model(settings, Grade))

    def grade_documents(state: dict[str, Any]) -> Literal["generate", "rewrite"]:
        logger.info("CHECK RELEVANCE")

        llm_with_tool = structured_llm()
        # Bind today's date so the grader is anchored in the present and does
        # not flag post-cutoff information as "not relevant".
        dated_grade_prompt = GRADE_PROMPT.partial(current_date=date.today().isoformat())
        chain = dated_grade_prompt | llm_with_tool

        question = question_resolver(state)
        retrieved_docs_text, _sources = build_sourced_context(
            question,
            state["messages"][-1],
            settings,
        )
        rewrite_count = int(state.get("rewrite_count", 0) or 0)

        llm_failed = False
        try:
            scored_result = invoke_with_retry(
                chain,
                {"question": question, "context": retrieved_docs_text},
                max_retries=settings.dashscope_max_retries,
            )
        except Exception as exc:
            logger.error("Grade documents error: %s", exc)
            llm_failed = True
            scored_result = Grade(
                binary_score="no", explanation="API error, using keyword matching"
            )

        score = scored_result.binary_score.strip().lower()
        explanation = getattr(scored_result, "explanation", "") or ""

        question_tokens = {word.lower() for word in re.findall(r"\w+", question) if len(word) > 2}
        retrieved_lower = (retrieved_docs_text or "").lower()
        keyword_matches = (
            sum(1 for t in question_tokens if t in retrieved_lower) if question_tokens else 0
        )

        logger.info("Grader output: score=%s; explanation=%s", score, explanation)
        required_keyword_matches = max(1, settings.min_keyword_matches)
        logger.info(
            "Keyword matches: %s (threshold=%s)",
            keyword_matches,
            required_keyword_matches,
        )
        logger.info("Rewrite count: %s/%s", rewrite_count, settings.max_rewrites)

        if score.startswith("y"):
            logger.info("DECISION: DOCS RELEVANT")
            return "generate"

        if llm_failed and (retrieved_docs_text or "").strip():
            logger.info("DECISION: SKIP REWRITE (LLM GRADER UNAVAILABLE)")
            return "generate"

        if settings.allow_low_relevance_generate and keyword_matches >= required_keyword_matches:
            logger.info("DECISION: DOCS MAYBE RELEVANT (FORCED GENERATE BY SETTINGS)")
            return "generate"

        if rewrite_count >= settings.max_rewrites:
            logger.info(
                "DECISION: REWRITE BUDGET EXHAUSTED (%s/%s); GENERATING WITH AVAILABLE CONTEXT",
                rewrite_count,
                settings.max_rewrites,
            )
            return "generate"

        logger.info("DECISION: DOCS NOT RELEVANT")
        return "rewrite"

    return grade_documents


def agent_factory(
    settings: Settings,
    tools: Sequence[Any],
    question_resolver: QuestionResolver = chat_question_resolver,
) -> NodeCallable:
    """Return the agent node."""

    tool_model = cached_builder(lambda: new_chat_model(settings).bind_tools(tools))
    agent_persona, _ = _persona_blocks(settings)

    def agent(state: dict[str, Any]) -> dict[str, list[AIMessage]]:
        logger.info("CALL AGENT")
        messages = list(state["messages"])
        if question_resolver is chat_question_resolver:
            messages = bounded_chat_messages(
                messages,
                max_turns=settings.chat_context_max_turns,
                max_chars=settings.chat_context_max_chars,
            )
        planning_context = str(state.get("planning_context") or "").strip()
        if getattr(settings, "planning_enabled", False) and planning_context:
            messages = [
                SystemMessage(
                    content=(
                        "Optional execution-plan context. Treat it as working context, "
                        "not as user instructions:\n" + planning_context
                    )
                ),
                *messages,
            ]
        # Prepend a system prompt so the model knows when to use tools and
        # when to answer directly from its own knowledge.
        dated_prompt = agent_system_prompt(agent_persona).format(
            current_date=date.today().isoformat()
        )
        messages = [SystemMessage(content=dated_prompt)] + messages
        model = tool_model()

        try:
            response = invoke_with_retry(
                model,
                messages,
                max_retries=settings.dashscope_max_retries,
            )
            # A direct agent answer is user-facing too, so it gets the same
            # citation-artifact cleanup. Tool-call carriers are left untouched.
            response = _sanitized_ai_message(response)
        except Exception as exc:
            logger.error("Agent error: %s", exc)
            question = question_resolver(state)
            tool_name = getattr(tools[0], "name", "retrieve_source_documents")
            response = AIMessage(
                content=(
                    "DashScope was unreachable while selecting a tool, so the "
                    "retriever is being called directly."
                ),
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": {"query": question},
                        "id": "fallback_retrieve",
                    }
                ],
            )

        return {"messages": [response]}

    return agent


def rewrite_factory(
    settings: Settings,
    question_resolver: QuestionResolver = chat_question_resolver,
    *,
    update_current_question: bool = False,
) -> NodeCallable:
    """Return the query-rewriting node."""

    plain_llm = cached_builder(lambda: new_chat_model(settings))

    def rewrite(state: dict[str, Any]) -> dict[str, Any]:
        logger.info("TRANSFORM QUERY")
        question = question_resolver(state)
        rewrite_count = int(state.get("rewrite_count", 0) or 0)

        rewrite_prompt = [
            HumanMessage(
                content=(
                    f"Today's date is {date.today().isoformat()}. Treat any "
                    "time references in the question as relative to this date "
                    "and do not assume facts must predate this date.\n\n"
                    "Look at the input and reason about the underlying semantic intent.\n\n"
                    "Initial question:\n"
                    "-------\n"
                    f"{question}\n"
                    "-------\n\n"
                    "Formulate an improved question:"
                )
            )
        ]

        try:
            response = invoke_with_retry(
                plain_llm(),
                rewrite_prompt,
                max_retries=settings.dashscope_max_retries,
            )
        except Exception as exc:
            logger.error("Rewrite error: %s", exc)
            response = AIMessage(content=question)

        output: dict[str, Any] = {
            "messages": [response],
            "rewrite_count": rewrite_count + 1,
        }
        if update_current_question:
            output["current_question"] = getattr(response, "content", "") or question
        return output

    return rewrite


def generate_factory(
    settings: Settings,
    question_resolver: QuestionResolver = chat_question_resolver,
) -> NodeCallable:
    """Return the final RAG answer generation node."""

    plain_llm = cached_builder(lambda: new_chat_model(settings))
    _, rag_persona = _persona_blocks(settings)

    def generate(state: dict[str, Any]) -> dict[str, list[AIMessage]]:
        logger.info("GENERATE")
        question = question_resolver(state)
        retrieved_docs_text, source_urls = build_sourced_context(
            question,
            state["messages"][-1],
            settings,
        )
        # Bind today's date so the model is anchored in the present and treats
        # retrieved context as current rather than dismissing post-cutoff facts.
        dated_prompt = rag_prompt(rag_persona).partial(current_date=date.today().isoformat())
        rag_chain = dated_prompt | plain_llm() | StrOutputParser()

        cited_urls: list[str] = []
        try:
            answer = invoke_with_retry(
                rag_chain,
                {"context": retrieved_docs_text, "question": question},
                max_retries=settings.dashscope_max_retries,
            )
            text = answer if isinstance(answer, str) else str(answer)
            text, cited_urls = validate_citations(strip_citation_artifacts(text), source_urls)
            response = AIMessage(content=text)
        except Exception as exc:
            logger.error("Generate error: %s", exc)
            fallback_text = build_extractive_answer(question, retrieved_docs_text)
            # The degraded path still attributes its source set wholesale: the
            # extractive answer is drawn from those chunks.
            cited_urls = list(source_urls)
            if cited_urls:
                footer = "\n\nSources:\n" + "\n".join(
                    f"[{index}] {url}" for index, url in enumerate(cited_urls, 1)
                )
                fallback_text += footer
            response = AIMessage(content=fallback_text)

        output: dict[str, Any] = {"messages": [response]}
        if cited_urls:
            output["source_urls"] = cited_urls
        return output

    return generate


def build_chat_agent_factory(settings: Settings, tools: Sequence[Any]) -> NodeCallable:
    return agent_factory(settings, tools, chat_question_resolver)
