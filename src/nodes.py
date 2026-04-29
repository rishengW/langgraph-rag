\
from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from pydantic import BaseModel, Field

from .config import Settings


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

        model = ChatTongyi(model=settings.qwen_model)
        llm_with_tool = model.with_structured_output(Grade)

        prompt = PromptTemplate(
            template=(
                "You are a grader assessing relevance of a retrieved document to a user question.\n\n"
                "Retrieved document:\n{context}\n\n"
                "User question: {question}\n\n"
                "If the document contains keyword(s) or semantic meaning related to the user "
                "question, grade it as relevant. Give a binary score 'yes' or 'no'."
            ),
            input_variables=["context", "question"],
        )

        chain = prompt | llm_with_tool

        messages = state["messages"]
        question = messages[0].content
        retrieved_docs_text = messages[-1].content

        scored_result = chain.invoke(
            {"question": question, "context": retrieved_docs_text}
        )

        score = scored_result.binary_score.strip().lower()

        if score.startswith("y"):
            print("---DECISION: DOCS RELEVANT---")
            return "generate"

        print("---DECISION: DOCS NOT RELEVANT---")
        return "rewrite"

    return grade_documents


def agent_factory(settings: Settings, tools):
    """Return the agent node."""

    def agent(state):
        print("---CALL AGENT---")
        messages = state["messages"]

        model = ChatTongyi(model=settings.qwen_model)
        model = model.bind_tools(tools)
        response = model.invoke(messages)

        return {"messages": [response]}

    return agent


def rewrite_factory(settings: Settings):
    """Return the query-rewriting node."""

    def rewrite(state):
        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

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

        model = ChatTongyi(model=settings.qwen_model)
        response = model.invoke(rewrite_prompt)

        # The original notebook returned {"message": ...}, which does not update
        # the graph state. This corrected key keeps the loop working.
        return {"messages": [response]}

    return rewrite


def generate_factory(settings: Settings):
    """Return the final RAG answer generation node."""

    def generate(state):
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        retrieved_docs_text = messages[-1].content

        llm = ChatTongyi(model=settings.qwen_model)
        rag_chain = RAG_PROMPT | llm | StrOutputParser()

        response = rag_chain.invoke(
            {"context": retrieved_docs_text, "question": question}
        )

        return {"messages": [response]}

    return generate
