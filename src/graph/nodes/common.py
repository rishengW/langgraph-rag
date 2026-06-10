from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import date
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from ...config import Settings
from ...llm.provider import build_chat_model
from ...llm.prompts import AGENT_SYSTEM_PROMPT, GRADE_PROMPT, RAG_PROMPT
from ...utils.networking import configure_ssl_from_env
from ...utils.retry import invoke_with_retry

logger = logging.getLogger(__name__)

QuestionResolver = Callable[[dict[str, Any]], str]

configure_ssl_from_env()


def new_chat_model(settings: Settings):
    """Create a DashScope chat model with project-level network settings."""

    return build_chat_model(settings)


def message_text(message: Any) -> str:
    if hasattr(message, "content"):
        return str(message.content)
    if isinstance(message, (tuple, list)) and len(message) >= 2:
        return str(message[1])
    return str(message)


def qa_question_resolver(state: dict[str, Any]) -> str:
    messages = state["messages"]
    return message_text(messages[0])


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

    strategy = _rerank_strategy(settings)
    if strategy == "embedding":
        ranked_chunks = _rank_chunks_by_embedding(question, chunks, settings)
    elif strategy == "hybrid":
        ranked_chunks = _rank_chunks_by_hybrid_score(question, chunks, settings)
    else:
        ranked_chunks = _rank_chunks(question, chunks)
    return "\n\n".join(ranked_chunks)


def _rerank_strategy(settings: Settings | None) -> str:
    if settings is None:
        return "lexical"
    strategy = settings.rerank_strategy.strip().lower()
    if strategy not in ("lexical", "embedding", "hybrid"):
        logger.warning("Unknown rerank strategy %r; using lexical", strategy)
        return "lexical"
    return strategy


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


def _rank_chunks(question: str, chunks: list[str]) -> list[str]:
    query_tokens: set[str] = _question_tokens(question)
    query_phrases: set[str] = _query_phrases(question)
    scored: list[tuple[int, int, str]] = []
    for index, chunk in enumerate(chunks):
        score: int = _lexical_relevance_score(query_tokens, query_phrases, chunk)
        scored.append((score, index, chunk))
    return [chunk for _, _, chunk in sorted(scored, key=lambda item: (-item[0], item[1]))]


def _rank_chunks_by_embedding(
    question: str,
    chunks: list[str],
    settings: Settings | None,
) -> list[str]:
    if settings is None:
        return _rank_chunks(question, chunks)
    try:
        scores = _embedding_relevance_scores(question, chunks, settings)
    except Exception as exc:
        logger.warning("Embedding rerank failed; using lexical rerank: %s", exc)
        return _rank_chunks(question, chunks)
    return _rank_chunks_by_scores(scores, chunks)


def _rank_chunks_by_hybrid_score(
    question: str,
    chunks: list[str],
    settings: Settings | None,
) -> list[str]:
    query_tokens: set[str] = _question_tokens(question)
    query_phrases: set[str] = _query_phrases(question)
    lexical_scores = [
        float(_lexical_relevance_score(query_tokens, query_phrases, chunk))
        for chunk in chunks
    ]
    if settings is None:
        return _rank_chunks_by_scores(lexical_scores, chunks)
    try:
        embedding_scores = _embedding_relevance_scores(question, chunks, settings)
    except Exception as exc:
        logger.warning(
            "Hybrid rerank embedding score failed; using lexical rerank: %s",
            exc,
        )
        return _rank_chunks_by_scores(lexical_scores, chunks)
    scores = [
        lexical + embedding
        for lexical, embedding in zip(
            _normalize_scores(lexical_scores),
            _normalize_scores(embedding_scores),
        )
    ]
    return _rank_chunks_by_scores(scores, chunks)


def _embedding_relevance_scores(
    question: str,
    chunks: list[str],
    settings: Settings,
) -> list[float]:
    embeddings = _build_rerank_embeddings(settings)
    query_vector = embeddings.embed_query(question)
    chunk_vectors = embeddings.embed_documents(chunks)
    if len(chunk_vectors) != len(chunks):
        raise RuntimeError(
            "Embedding provider returned the wrong number of chunk vectors."
        )
    return [
        _cosine_similarity(query_vector, chunk_vector)
        for chunk_vector in chunk_vectors
    ]


def _build_rerank_embeddings(settings: Settings) -> Any:
    from ...rag.embeddings import build_embeddings

    return build_embeddings(settings)


def _rank_chunks_by_scores(scores: list[float], chunks: list[str]) -> list[str]:
    scored = [
        (score, index, chunk)
        for index, (score, chunk) in enumerate(zip(scores, chunks))
    ]
    return [
        chunk
        for _, _, chunk in sorted(scored, key=lambda item: (-item[0], item[1]))
    ]


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
        left_value * right_value
        for left_value, right_value in zip(left, right)
    )
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


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

    sentences = []
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
        if "reinforcement learning" in lower and {"reinforcement", "learning"} <= tokens:
            score += 2
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
        "an extractive answer from the retrieved article context:\n\n"
        + "\n".join(lines)
    )


def grade_documents_factory(
    settings: Settings,
    question_resolver: QuestionResolver = qa_question_resolver,
):
    """Return a conditional edge function that grades retrieved context."""

    def grade_documents(state) -> Literal["generate", "rewrite"]:
        logger.info("CHECK RELEVANCE")

        class Grade(BaseModel):
            binary_score: str = Field(description="Relevance score: 'yes' or 'no'")
            explanation: Optional[str] = Field(None, description="Optional short explanation")

        llm_with_tool = new_chat_model(settings).with_structured_output(Grade)
        # Bind today's date so the grader is anchored in the present and does
        # not flag post-cutoff information as "not relevant".
        dated_grade_prompt = GRADE_PROMPT.partial(current_date=date.today().isoformat())
        chain = dated_grade_prompt | llm_with_tool

        question = question_resolver(state)
        retrieved_docs_text = rerank_retrieved_context(
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
            scored_result = Grade(binary_score="no", explanation="API error, using keyword matching")

        score = scored_result.binary_score.strip().lower()
        explanation = getattr(scored_result, "explanation", "") or ""

        question_tokens = set(w.lower() for w in re.findall(r"\w+", question) if len(w) > 2)
        retrieved_lower = (retrieved_docs_text or "").lower()
        keyword_matches = sum(1 for t in question_tokens if t in retrieved_lower) if question_tokens else 0

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
    tools,
    question_resolver: QuestionResolver = qa_question_resolver,
):
    """Return the agent node."""

    def agent(state):
        logger.info("CALL AGENT")
        messages = state["messages"]
        # Prepend a system prompt so the model knows when to use tools and
        # when to answer directly from its own knowledge.
        dated_prompt = AGENT_SYSTEM_PROMPT.format(current_date=date.today().isoformat())
        messages = [SystemMessage(content=dated_prompt)] + list(messages)
        model = new_chat_model(settings).bind_tools(tools)

        try:
            response = invoke_with_retry(
                model,
                messages,
                max_retries=settings.dashscope_max_retries,
            )
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
    question_resolver: QuestionResolver = qa_question_resolver,
    *,
    update_current_question: bool = False,
):
    """Return the query-rewriting node."""

    def rewrite(state):
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
                new_chat_model(settings),
                rewrite_prompt,
                max_retries=settings.dashscope_max_retries,
            )
        except Exception as exc:
            logger.error("Rewrite error: %s", exc)
            response = AIMessage(content=question)

        output = {
            "messages": [response],
            "rewrite_count": rewrite_count + 1,
        }
        if update_current_question:
            output["current_question"] = getattr(response, "content", "") or question
        return output

    return rewrite


def generate_factory(
    settings: Settings,
    question_resolver: QuestionResolver = qa_question_resolver,
):
    """Return the final RAG answer generation node."""

    def generate(state):
        logger.info("GENERATE")
        question = question_resolver(state)
        retrieved_docs_text = rerank_retrieved_context(
            question,
            state["messages"][-1],
            settings,
        )
        # Bind today's date so the model is anchored in the present and treats
        # retrieved context as current rather than dismissing post-cutoff facts.
        dated_prompt = RAG_PROMPT.partial(current_date=date.today().isoformat())
        rag_chain = dated_prompt | new_chat_model(settings) | StrOutputParser()

        try:
            answer = invoke_with_retry(
                rag_chain,
                {"context": retrieved_docs_text, "question": question},
                max_retries=settings.dashscope_max_retries,
            )
            response = AIMessage(content=answer if isinstance(answer, str) else str(answer))
        except Exception as exc:
            logger.error("Generate error: %s", exc)
            response = AIMessage(content=build_extractive_answer(question, retrieved_docs_text))

        return {"messages": [response]}

    return generate


def build_core_agent_factory(settings: Settings, tools):
    return agent_factory(settings, tools, qa_question_resolver)


def build_chat_agent_factory(settings: Settings, tools):
    return agent_factory(settings, tools, chat_question_resolver)

