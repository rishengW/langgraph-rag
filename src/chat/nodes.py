"""Chat-flavored graph nodes.

These mirror the single-shot nodes in ``src.core.nodes`` but resolve the
"current user question" via the chat state's ``current_question`` field
instead of assuming it is ``messages[0]``.

A new ``condense_question`` node runs at the start of each turn to rewrite
multi-turn follow-ups ("what about that one?") into self-contained
questions before retrieval. This is the standard fix for RAG-chat: the
retriever embedding has no conversational context, so a vague follow-up
turns into a noisy retrieval unless we condense it first.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field

from ..core.config import Settings
from ..core.nodes import (
    RAG_PROMPT,
    _build_extractive_answer,
    _invoke_with_retry,
    _new_chat_model,
)


logger = logging.getLogger(__name__)


CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Given the following conversation and a follow-up question, rewrite the \
follow-up so it is a standalone question that can be understood without the \
prior context. Preserve the user's intent and language. If the follow-up is \
already self-contained, return it unchanged.

Conversation history:
{history}

Follow-up question:
{question}

Standalone question:"""
)


def _format_history(messages) -> str:
    """Render the prior conversation (everything before the current user turn)
    as a compact transcript for the condense prompt."""

    lines: list[str] = []
    for msg in messages:
        role = getattr(msg, "type", None) or msg.__class__.__name__.lower()
        content = getattr(msg, "content", None)
        if not content:
            continue
        # Skip tool-call payloads; the condenser only needs human/assistant turns
        if role in ("tool", "function"):
            continue
        if role.startswith("human") or role == "user":
            lines.append(f"User: {content}")
        elif role.startswith("ai") or role == "assistant":
            # An AI turn that triggered a tool call usually has empty content;
            # we already filtered those above.
            lines.append(f"Assistant: {content}")
    return "\n".join(lines) if lines else "(no prior turns)"


def _latest_user_index(messages) -> int:
    """Return the index of the most recent HumanMessage."""

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        role = getattr(msg, "type", None) or msg.__class__.__name__.lower()
        if role.startswith("human") or role == "user":
            return i
    return 0


def condense_question_factory(settings: Settings):
    """First node of every chat turn.

    Reads the latest user message, condenses it into a standalone question
    using the prior history (skipped on turn 1), and stashes the result on
    state so downstream nodes can use it without re-deriving."""

    def condense_question(state):
        print("---CONDENSE QUESTION---")
        messages = list(state["messages"])
        if not messages:
            return {"current_question": "", "current_question_index": 0}

        latest_idx = _latest_user_index(messages)
        latest_msg = messages[latest_idx]
        latest_text = getattr(latest_msg, "content", str(latest_msg))

        history = messages[:latest_idx]
        if not history:
            # Turn 1: no prior context, the question is already standalone.
            return {
                "current_question": latest_text,
                "current_question_index": latest_idx,
                # Reset rewrite budget per turn so budgets don't carry over.
                "rewrite_count": 0,
            }

        history_text = _format_history(history)
        chain = CONDENSE_PROMPT | _new_chat_model(settings) | StrOutputParser()

        try:
            standalone = _invoke_with_retry(
                chain,
                {"history": history_text, "question": latest_text},
                max_retries=settings.dashscope_max_retries,
            )
            standalone = (standalone or "").strip() or latest_text
        except Exception as exc:
            logger.error(f"Condense error; using raw question: {exc}")
            standalone = latest_text

        print(f"Condensed: {standalone!r}")
        return {
            "current_question": standalone,
            "current_question_index": latest_idx,
            "rewrite_count": 0,
        }

    return condense_question


def _question_from_state(state) -> str:
    """Resolve the current user question from chat state.

    Prefers the condensed standalone question. Falls back to the raw text at
    ``current_question_index`` and finally to ``messages[-1]``."""

    standalone = (state.get("current_question") or "").strip()
    if standalone:
        return standalone

    messages = state.get("messages") or []
    idx = int(state.get("current_question_index", -1) or -1)
    if 0 <= idx < len(messages):
        return getattr(messages[idx], "content", str(messages[idx]))

    if messages:
        return getattr(messages[-1], "content", str(messages[-1]))
    return ""


def agent_factory(settings: Settings, tools):
    """Chat agent: like the single-shot agent, but uses the condensed
    question for the fallback retriever call instead of ``messages[0]``."""

    def agent(state):
        print("---CALL AGENT---")
        messages = state["messages"]
        model = _new_chat_model(settings).bind_tools(tools)

        try:
            response = _invoke_with_retry(
                model,
                messages,
                max_retries=settings.dashscope_max_retries,
            )
        except Exception as exc:
            logger.error(f"Agent error: {exc}")
            question = _question_from_state(state)
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


def grade_documents_factory(settings: Settings):
    """Grader keyed on the condensed question rather than ``messages[0]``."""

    def grade_documents(state) -> Literal["generate", "rewrite"]:
        print("---CHECK RELEVANCE---")

        class Grade(BaseModel):
            binary_score: str = Field(description="Relevance score: 'yes' or 'no'")
            explanation: Optional[str] = Field(None, description="Optional short explanation")

        model = _new_chat_model(settings)
        llm_with_tool = model.with_structured_output(Grade)

        prompt = PromptTemplate(
            template=(
                "You are a grader assessing relevance of a retrieved document to a user question.\n\n"
                "Retrieved document:\n{context}\n\n"
                "User question: {question}\n\n"
                "If the document contains keyword(s) or semantic meaning related to the user "
                "question, grade it as relevant. Give a binary score 'yes' or 'no'.\n"
                "Also provide a short explanation of your judgement."
            ),
            input_variables=["context", "question"],
        )

        chain = prompt | llm_with_tool

        question = _question_from_state(state)
        retrieved_docs_text = state["messages"][-1].content
        rewrite_count = int(state.get("rewrite_count", 0) or 0)

        llm_failed = False
        try:
            scored_result = _invoke_with_retry(
                chain,
                {"question": question, "context": retrieved_docs_text},
                max_retries=settings.dashscope_max_retries,
            )
        except Exception as exc:
            logger.error(f"Grade documents error: {exc}")
            llm_failed = True
            scored_result = Grade(binary_score="no", explanation="API error, using keyword matching")

        score = scored_result.binary_score.strip().lower()
        explanation = getattr(scored_result, "explanation", "") or ""

        question_tokens = set(w.lower() for w in re.findall(r"\w+", question) if len(w) > 2)
        retrieved_lower = (retrieved_docs_text or "").lower()
        keyword_matches = sum(1 for t in question_tokens if t in retrieved_lower) if question_tokens else 0

        print(f"Grader output: score={score}; explanation={explanation}")
        print(f"Keyword matches: {keyword_matches} (threshold={settings.min_keyword_matches})")
        print(f"Rewrite count: {rewrite_count}/{settings.max_rewrites}")

        if score.startswith("y"):
            print("---DECISION: DOCS RELEVANT---")
            return "generate"

        if llm_failed and (retrieved_docs_text or "").strip():
            print("---DECISION: SKIP REWRITE (LLM GRADER UNAVAILABLE)---")
            return "generate"

        if settings.allow_low_relevance_generate and keyword_matches >= settings.min_keyword_matches:
            print("---DECISION: DOCS MAYBE RELEVANT (FORCED GENERATE BY SETTINGS)---")
            return "generate"

        if rewrite_count >= settings.max_rewrites:
            print(
                f"---DECISION: REWRITE BUDGET EXHAUSTED ({rewrite_count}/{settings.max_rewrites}); "
                "GENERATING WITH AVAILABLE CONTEXT---"
            )
            return "generate"

        print("---DECISION: DOCS NOT RELEVANT---")
        return "rewrite"

    return grade_documents


def rewrite_factory(settings: Settings):
    """Rewriter: refines the *condensed* question on retrieval failure.

    On API failure we emit an AIMessage carrying the original condensed
    question (so the agent loop sees a non-user turn and re-tries), and
    increment ``rewrite_count`` so the grader can cap the cycle."""

    def rewrite(state):
        print("---TRANSFORM QUERY---")
        question = _question_from_state(state)
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
            response = _invoke_with_retry(
                _new_chat_model(settings),
                rewrite_prompt,
                max_retries=settings.dashscope_max_retries,
            )
        except Exception as exc:
            logger.error(f"Rewrite error: {exc}")
            response = AIMessage(content=question)

        # Update the standalone question to the refined version so the next
        # retrieval pass uses it. Falls back to the original on errors.
        new_question = getattr(response, "content", "") or question

        return {
            "messages": [response],
            "rewrite_count": rewrite_count + 1,
            "current_question": new_question,
        }

    return rewrite


def generate_factory(settings: Settings):
    """Final answer generation, keyed on the condensed question."""

    def generate(state):
        print("---GENERATE---")
        question = _question_from_state(state)
        retrieved_docs_text = state["messages"][-1].content

        rag_chain = RAG_PROMPT | _new_chat_model(settings) | StrOutputParser()

        try:
            answer = _invoke_with_retry(
                rag_chain,
                {"context": retrieved_docs_text, "question": question},
                max_retries=settings.dashscope_max_retries,
            )
            response = AIMessage(content=answer)
        except Exception as exc:
            logger.error(f"Generate error: {exc}")
            response = AIMessage(
                content=_build_extractive_answer(question, retrieved_docs_text)
            )

        return {"messages": [response]}

    return generate
