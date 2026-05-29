from __future__ import annotations

from typing import Any, Literal, Optional
import re
import os
import time
import logging
import ssl
import urllib3
from requests.exceptions import RequestException

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from pydantic import BaseModel, Field

from .config import Settings

logger = logging.getLogger(__name__)

# Suppress SSL warnings if verification is disabled
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _configure_ssl():
    """Configure SSL settings for safer connections to Dashscope."""
    # Allow disabling SSL verification via environment variable (for debugging)
    if os.getenv("DISABLE_SSL_VERIFY", "").lower() == "true":
        os.environ["REQUESTS_CA_BUNDLE"] = ""
        os.environ["CURL_CA_BUNDLE"] = ""
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except Exception as e:
            logger.warning(f"Could not disable SSL verification: {e}")


def _new_chat_model(settings: Settings) -> ChatTongyi:
    """Create a DashScope chat model with project-level network settings."""

    model_kwargs: dict[str, Any] = {
        "request_timeout": settings.dashscope_request_timeout,
    }
    if settings.dashscope_http_base_url:
        model_kwargs["base_address"] = settings.dashscope_http_base_url

    return ChatTongyi(
        model=settings.qwen_model,
        max_retries=settings.dashscope_max_retries,
        model_kwargs=model_kwargs,
    )


def _is_retryable_connection_error(error: Exception) -> bool:
    error_msg = str(error).upper()
    retry_markers = (
        "SSL",
        "CERTIFICATE",
        "EOF",
        "CONNECTION",
        "MAX RETRIES",
        "TIMEOUT",
        "REMOTE END",
        "TEMPORARILY UNAVAILABLE",
    )
    return any(marker in error_msg for marker in retry_markers)


def _invoke_with_retry(chain, input_data, max_retries=3, base_delay=1.0):
    """Invoke a chain with retry logic for transient SSL/connection errors."""
    last_error = None
    max_retries = max(1, max_retries)
    
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        except (OSError, ConnectionError, TimeoutError, RequestException, ssl.SSLError) as e:
            last_error = e
            if _is_retryable_connection_error(e):
                logger.warning(
                    f"SSL/Connection error on attempt {attempt + 1}/{max_retries}: {e}"
                )
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + (os.urandom(1)[0] / 256)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
            else:
                # Not a retryable error, raise immediately
                raise
        except Exception as e:
            # Non-retryable error
            logger.error(f"Non-retryable error: {e}")
            raise
    
    # If we got here, all retries failed
    raise last_error


def _message_text(message: Any) -> str:
    if hasattr(message, "content"):
        return str(message.content)
    if isinstance(message, (tuple, list)) and len(message) >= 2:
        return str(message[1])
    return str(message)


def _question_from_state(state) -> str:
    messages = state["messages"]
    return _message_text(messages[0])


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


def _build_extractive_answer(question: str, context: str) -> str:
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


# Configure SSL on module load
_configure_ssl()


RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks.

Use the following retrieved context to answer the question.
If you do not know the answer from the context, say that you do not know.
Keep the answer concise.

Question:
{question}

Context:
{context}

Answer:"""
)


def grade_documents_factory(settings: Settings):
    """Return a conditional edge function that grades retrieved context."""

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

        messages = state["messages"]
        question = messages[0].content
        retrieved_docs_text = messages[-1].content
        rewrite_count = int(state.get("rewrite_count", 0) or 0)

        llm_failed = False
        try:
            scored_result = _invoke_with_retry(
                chain, 
                {"question": question, "context": retrieved_docs_text},
                max_retries=settings.dashscope_max_retries,
            )
        except Exception as e:
            logger.error(f"Grade documents error: {e}")
            llm_failed = True
            scored_result = Grade(binary_score="no", explanation="API error, using keyword matching")

        score = scored_result.binary_score.strip().lower()
        explanation = getattr(scored_result, "explanation", "") or ""

        # Simple keyword overlap heuristic as a fallback
        question_tokens = set(w.lower() for w in re.findall(r"\w+", question) if len(w) > 2)
        retrieved_lower = (retrieved_docs_text or "").lower()
        keyword_matches = sum(1 for t in question_tokens if t in retrieved_lower) if question_tokens else 0

        print(f"Grader output: score={score}; explanation={explanation}")
        print(f"Keyword matches: {keyword_matches} (threshold={settings.min_keyword_matches})")
        print(f"Rewrite count: {rewrite_count}/{settings.max_rewrites}")

        # Decision rules: accept yes; otherwise allow generation if setting enabled and keywords match
        if score.startswith("y"):
            print("---DECISION: DOCS RELEVANT---")
            return "generate"

        if llm_failed and retrieved_docs_text.strip():
            print("---DECISION: SKIP REWRITE (LLM GRADER UNAVAILABLE)---")
            return "generate"

        if settings.allow_low_relevance_generate and keyword_matches >= settings.min_keyword_matches:
            print("---DECISION: DOCS MAYBE RELEVANT (FORCED GENERATE BY SETTINGS)---")
            return "generate"

        # Hard cap on rewrites: once we've burned our rewrite budget, stop looping
        # and let `generate` produce an answer (or its extractive fallback) from
        # whatever context we have rather than spinning forever.
        if rewrite_count >= settings.max_rewrites:
            print(
                f"---DECISION: REWRITE BUDGET EXHAUSTED ({rewrite_count}/{settings.max_rewrites}); "
                "GENERATING WITH AVAILABLE CONTEXT---"
            )
            return "generate"

        print("---DECISION: DOCS NOT RELEVANT---")
        return "rewrite"

    return grade_documents


def agent_factory(settings: Settings, tools):
    """Return the agent node."""

    def agent(state):
        print("---CALL AGENT---")
        messages = state["messages"]

        model = _new_chat_model(settings)
        model = model.bind_tools(tools)
        
        try:
            response = _invoke_with_retry(
                model,
                messages,
                max_retries=settings.dashscope_max_retries,
            )
        except Exception as e:
            logger.error(f"Agent error: {e}")
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


def rewrite_factory(settings: Settings):
    """Return the query-rewriting node."""

    def rewrite(state):
        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content
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

        model = _new_chat_model(settings)

        try:
            response = _invoke_with_retry(
                model,
                rewrite_prompt,
                max_retries=settings.dashscope_max_retries,
            )
        except Exception as e:
            logger.error(f"Rewrite error: {e}")
            # Fallback: emit an AIMessage so the graph treats this as a model
            # turn (not a fresh user turn) and the agent re-invokes the
            # retriever with the original question. Returning a HumanMessage
            # here would disguise a fallback as user input and confuse tracing.
            response = AIMessage(content=question)

        # The original notebook returned {"message": ...}, which does not update
        # the graph state. This corrected key keeps the loop working.
        return {
            "messages": [response],
            "rewrite_count": rewrite_count + 1,
        }

    return rewrite


def generate_factory(settings: Settings):
    """Return the final RAG answer generation node."""

    def generate(state):
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        retrieved_docs_text = messages[-1].content

        llm = _new_chat_model(settings)
        rag_chain = RAG_PROMPT | llm | StrOutputParser()

        try:
            response = _invoke_with_retry(
                rag_chain,
                {"context": retrieved_docs_text, "question": question},
                max_retries=settings.dashscope_max_retries,
            )
        except Exception as e:
            logger.error(f"Generate error: {e}")
            response = AIMessage(content=_build_extractive_answer(question, retrieved_docs_text))

        return {"messages": [response]}

    return generate
