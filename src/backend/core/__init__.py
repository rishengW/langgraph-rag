"""Shared engine for the only Subscribers project.

Houses configuration, embeddings, the LangGraph workflow, retriever, web
search helpers, and graph state. The chatbot app (``src.frontend.chat``) and the
inbound MCP server import from here; most modules are deprecated re-export
shims that point at ``src.backend.graph``, ``src.backend.rag``, ``src.config``, and
``src.backend.web_search``.
"""
