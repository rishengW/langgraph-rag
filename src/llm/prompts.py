from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks.

Today's date is {current_date}. The context below was retrieved from sources
that reflect the current state of the world and may be MORE UP TO DATE than
your own training data. When the context conflicts with your prior knowledge,
trust the context. Do not dismiss information as future, unreleased, or
non-existent merely because it postdates your training cutoff.

Use the following retrieved context to answer the question.
If you do not know the answer from the context, say that you do not know.
Keep the answer concise.

Question:
{question}

Context:
{context}

Answer:"""
)

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

GRADE_PROMPT = PromptTemplate(
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

