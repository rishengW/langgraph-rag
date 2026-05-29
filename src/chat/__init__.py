"""Multi-turn chatbot application.

Wraps the shared ``src.core`` engine with conversation memory (LangGraph's
``MemorySaver``), per-thread session management, and a standalone-question
rewrite step so follow-up turns ("what about that?") still produce useful
retriever queries.

Exposes both a CLI (``python -m src.chat.main``) and a FastAPI web app.
"""
