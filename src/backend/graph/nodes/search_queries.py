from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.messages import ToolMessage

from ...web_search.query_constraints import validate_query_candidate
from ...web_search.query_prep import plan_search_queries
from ..edges import WEB_SEARCH_TOOL_NAME

logger = logging.getLogger(__name__)

WEB_SEARCH_MAX_QUERIES = 6
WEB_SEARCH_MAX_CONCURRENCY = 3

_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}]+")


def search_queries_factory(
    tools: Sequence[Any],
    *,
    max_queries: int = WEB_SEARCH_MAX_QUERIES,
    max_concurrency: int = WEB_SEARCH_MAX_CONCURRENCY,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a node that executes the current search-query batch in parallel.

    A single aggregate ``ToolMessage`` satisfies the agent's original tool-call
    protocol. Structured per-query URL sets are also written to state so merge
    can retain provider rank and cross-query overlap.
    """

    search_tool = _find_search_tool(tools)
    query_limit = max(1, int(max_queries))
    concurrency_limit = max(1, int(max_concurrency))

    def search_queries(state: dict[str, Any]) -> dict[str, Any]:
        update: dict[str, Any]
        queries = _resolve_queries(state, query_limit)
        tool_call = _current_search_tool_call(state)
        base_args = _tool_args(tool_call)

        if not queries:
            update = {
                "search_queries": [],
                "web_search_results": [],
                "web_search_result_metadata": [],
            }
            if tool_call is not None:
                update["messages"] = [
                    ToolMessage(
                        content="No live web search query was available.",
                        name=WEB_SEARCH_TOOL_NAME,
                        tool_call_id=str(tool_call.get("id") or "bounded_web_search"),
                    )
                ]
            return update

        worker_count = min(len(queries), concurrency_limit)

        def invoke(query: str) -> tuple[str, list[dict[str, Any]]]:
            args = dict(base_args)
            args["query"] = query
            try:
                result = search_tool.invoke(
                    {
                        "name": WEB_SEARCH_TOOL_NAME,
                        "args": args,
                        "id": "bounded_web_search_result",
                        "type": "tool_call",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - one failed query must not cancel the batch
                logger.warning("Live web search failed for %r: %s", query, exc)
                return f"No live web search results found for: {query}", []
            content = getattr(result, "content", result)
            output = str(content or "")
            return output, _metadata_from_result(result, output)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            search_outputs = list(executor.map(invoke, queries))

        outputs = [output for output, _metadata in search_outputs]
        metadata_sets = [metadata for _output, metadata in search_outputs]
        result_sets = [_urls_from_output(output) for output in outputs]
        content = _aggregate_outputs(queries, outputs)
        update = {
            "search_queries": queries,
            "web_search_results": result_sets,
            "web_search_result_metadata": metadata_sets,
        }
        if tool_call is not None:
            update["messages"] = [
                ToolMessage(
                    content=content,
                    name=WEB_SEARCH_TOOL_NAME,
                    tool_call_id=str(tool_call.get("id") or "bounded_web_search"),
                    artifact={
                        "queries": queries,
                        "result_sets": metadata_sets,
                    },
                )
            ]
        return update

    return search_queries


def _find_search_tool(tools: Sequence[Any]) -> Any:
    for candidate in tools:
        if getattr(candidate, "name", None) == WEB_SEARCH_TOOL_NAME:
            return candidate
    raise ValueError(f"{WEB_SEARCH_TOOL_NAME!r} is required for bounded web search")


def _resolve_queries(state: dict[str, Any], limit: int) -> list[str]:
    original_question = _state_original_question(state)
    keys = (
        ("expanded_queries", "search_queries", "sub_questions")
        if state.get("expansion_attempted")
        else ("search_queries", "sub_questions", "expanded_queries")
    )
    for key in keys:
        queries = _clean_queries(
            state.get(key),
            limit,
            original_question=original_question,
        )
        if queries:
            return queries

    tool_call = _current_search_tool_call(state)
    args = (tool_call or {}).get("args")
    query = args.get("query") if isinstance(args, dict) else None
    return _clean_queries([query], limit, original_question=original_question)


def _clean_queries(
    values: Any,
    limit: int,
    *,
    original_question: str = "",
) -> list[str]:
    if not isinstance(values, list):
        return []
    queries: list[str] = []
    if original_question:
        _append_planned_queries(queries, original_question, limit)
    for value in values:
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        if not candidate:
            continue
        if (
            original_question
            and candidate != original_question
            and not validate_query_candidate(
                original_question,
                candidate,
                allow_partial=True,
            )
        ):
            logger.info(
                "Discarded search query that changed language or constraints: %r", candidate
            )
            continue
        _append_planned_queries(queries, candidate, limit)
        if len(queries) >= limit:
            return queries
    return queries


def _append_planned_queries(queries: list[str], raw_query: str, limit: int) -> None:
    for planned_query in plan_search_queries(raw_query):
        query = planned_query.strip()
        if query and query not in queries:
            queries.append(query)
        if len(queries) >= limit:
            return


def _state_original_question(state: dict[str, Any]) -> str:
    current = str(state.get("current_question") or "").strip()
    if current:
        return current
    messages = state.get("messages") or []
    for message in reversed(list(messages)):
        role = str(getattr(message, "type", "") or "").lower()
        if role not in {"human", "user"}:
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _current_search_tool_call(state: dict[str, Any]) -> dict[str, Any] | None:
    messages = state.get("messages") or []
    if not messages:
        return None
    calls = getattr(list(messages)[-1], "tool_calls", None) or []
    for call in reversed(list(calls)):
        if isinstance(call, dict) and call.get("name") == WEB_SEARCH_TOOL_NAME:
            return call
    return None


def _tool_args(tool_call: dict[str, Any] | None) -> dict[str, Any]:
    if not tool_call:
        return {}
    args = tool_call.get("args")
    if not isinstance(args, dict):
        return {}
    # The query is supplied by the bounded batch. Preserve only the search
    # result limit from the agent's original call.
    max_results = args.get("max_results")
    return {"max_results": max_results} if max_results is not None else {}


def _urls_from_output(output: str) -> list[str]:
    urls: list[str] = []
    for match in _URL_RE.findall(output):
        url = match.rstrip(".,;)]}")
        if url and url not in urls:
            urls.append(url)
    return urls


def _metadata_from_result(result: Any, output: str) -> list[dict[str, Any]]:
    artifact = getattr(result, "artifact", None)
    values = artifact.get("results") if isinstance(artifact, dict) else None
    metadata: list[dict[str, Any]] = []
    if isinstance(values, list):
        for value in values:
            if not isinstance(value, dict):
                continue
            url = str(value.get("url") or "").strip()
            if not url:
                continue
            metadata.append(
                {
                    "url": url,
                    "title": str(value.get("title") or ""),
                    "snippet": str(value.get("snippet") or ""),
                    "provider": str(value.get("provider") or ""),
                    "provider_rank": _int_value(value.get("provider_rank"), len(metadata)),
                    "relevance_score": _int_value(value.get("relevance_score"), 0),
                    "quality_score": _int_value(value.get("quality_score"), 0),
                }
            )
    if metadata:
        return metadata
    return [
        {
            "url": url,
            "provider_rank": provider_rank,
            "relevance_score": 0,
            "quality_score": 0,
        }
        for provider_rank, url in enumerate(_urls_from_output(output))
    ]


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _aggregate_outputs(queries: list[str], outputs: list[str]) -> str:
    if len(outputs) == 1:
        return outputs[0]
    sections = [
        f"Search query {index}: {query}\n{output}"
        for index, (query, output) in enumerate(zip(queries, outputs, strict=True), start=1)
    ]
    return "\n\n".join(sections)


__all__ = [
    "WEB_SEARCH_MAX_CONCURRENCY",
    "WEB_SEARCH_MAX_QUERIES",
    "search_queries_factory",
]
