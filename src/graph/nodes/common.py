from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from ...config import Settings
from ...llm.provider import build_chat_model
from ...llm.prompts import GRADE_PROMPT, RAG_PROMPT
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
        chain = GRADE_PROMPT | llm_with_tool

        question = question_resolver(state)
        retrieved_docs_text = state["messages"][-1].content
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
        logger.info(
            "Keyword matches: %s (threshold=%s)",
            keyword_matches,
            settings.min_keyword_matches,
        )
        logger.info("Rewrite count: %s/%s", rewrite_count, settings.max_rewrites)

        if score.startswith("y"):
            logger.info("DECISION: DOCS RELEVANT")
            return "generate"

        if llm_failed and (retrieved_docs_text or "").strip():
            logger.info("DECISION: SKIP REWRITE (LLM GRADER UNAVAILABLE)")
            return "generate"

        if settings.allow_low_relevance_generate and keyword_matches >= settings.min_keyword_matches:
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
        retrieved_docs_text = state["messages"][-1].content
        rag_chain = RAG_PROMPT | new_chat_model(settings) | StrOutputParser()

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

