"""Shared engine for the only Subcribers project.

Houses configuration, embeddings, the LangGraph workflow, retriever, web
search helpers, and graph state. The chatbot app (``src.chat``) and the
inbound MCP server import from here; most modules are deprecated re-export
shims that point at ``src.graph``, ``src.rag``, ``src.config``, and
``src.web_search``.
"""
