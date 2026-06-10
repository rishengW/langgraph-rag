from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

AGENT_SYSTEM_PROMPT = (
    "Today's date is {current_date}.\n\n"
    "You are a helpful, conversational AI assistant. You have access to tools "
    "that can search external document sources and the live web.\n\n"
    "IMPORTANT — When to use tools:\n"
    "- Use tools ONLY when the question requires information from the configured "
    "document sources, or up-to-date / real-time information beyond your "
    "knowledge cutoff.\n"
    "- Answer DIRECTLY from your own knowledge for everything else, including:\n"
    "  * Math, calculations, and logic puzzles\n"
    "  * Programming concepts, syntax explanations, and debugging advice\n"
    "  * General knowledge, history, science, and well-known facts\n"
    "  * Definitions and explanations of established concepts\n"
    "  * Greetings, chitchat, and conversational turns\n"
    "- Do NOT call a tool when you can confidently answer from your own training.\n\n"
    "When you DO call a tool, formulate the query as search-engine keywords "
    "(not a natural-language question): extract core concepts and named entities, "
    "drop filler words, and include the current year for time-sensitive queries."
)

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks.

Today's date is {current_date}. The context below was retrieved from sources
that reflect the current state of the world and may be MORE UP TO DATE than
your own training data. When the context conflicts with your prior knowledge,
trust the context. Do not dismiss information as future, unreleased, or
non-existent merely because it postdates your training cutoff.

IMPORTANT — Semantic matching: The question may use different words than the
retrieved context. Match on MEANING, not exact phrasing. For example: if the
question asks about the 'latest model' and the context describes a 'new release'
or 'V4 Pro launched in 2026', those ARE the answer — do NOT reject them because
the context doesn't use the exact words 'latest model'. Bridge vocabulary gaps
between the question and the sources.

Use the following retrieved context to answer the question.
If you do not know the answer from the context after semantic matching, say that
you do not know. Keep the answer concise.

Question:
{question}

Context:
{context}

Answer:"""
)

CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Today's date is {current_date}. Treat any time references in the
follow-up as relative to this date.

Given the following conversation and a follow-up question, rewrite the \
follow-up so it is a standalone question that can be understood without the \
prior context. Preserve the user's intent and language. If the follow-up is \
already self-contained, return it unchanged.

Conversation history:
{history}

Follow-up question:
{question}

Standalone question:"""
)

GRADE_PROMPT = PromptTemplate(
    template=(
        "Today's date is {current_date}. Information that postdates your "
        "training cutoff is not 'fictional' or 'unreleased' — if the document "
        "describes it, treat it as real and current.\n\n"
        "You are a grader assessing relevance of a retrieved document to a user question.\n\n"
        "Retrieved document:\n{context}\n\n"
        "User question: {question}\n\n"
        "Match on MEANING, not exact phrasing. If the document discusses the "
        "same subject the question asks about (even with different vocabulary, "
        "like 'new release' vs 'latest model', or contains a date/version that "
        "answers a 'when'/'which' question), grade it as relevant. Give a "
        "binary score 'yes' or 'no'.\n"
        "Also provide a short explanation of your judgement."
    ),
    input_variables=["context", "question", "current_date"],
)

