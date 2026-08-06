"""Agent-callable long-term memory tools.

Three tools share one module because they share one store handle and one feature
flag; splitting them would triplicate the wiring for no gain. Precedent:
``file_read_enabled`` gates four readers that share ``_files.py``.

Two properties are load-bearing here:

* **No filesystem or thread parameter is exposed to the model.** The store path
  comes from ``Settings`` alone, and ``thread_id`` arrives through the injected
  ``RunnableConfig``, which LangChain keeps out of ``args_schema``.
* **Nothing raises.** ``_guarded`` is the single error boundary. Every failure,
  including an unexpected exception type, comes back as a string prefixed with
  ``MEMORY_ERROR:`` so the agent loop continues.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ..memory.models import CATEGORIES, MAX_QUERY_CHARS, MAX_TAG_CHARS, MAX_TAGS, SCOPES
from ..memory.recall import MEMORY_BUDGET, TRUNCATION_MARKER, MemoryCallBudget
from ..memory.store import MemoryStore, get_memory_store

if TYPE_CHECKING:
    from ..config import Settings

logger = logging.getLogger(__name__)

#: Prefix that marks a tool result as a failure. Success payloads never start
#: with it, so the model can tell a failure from a legitimate "nothing found".
FAILURE_MARKER = "MEMORY_ERROR:"

#: Upper bound on a failure string handed back to the model.
MAX_ERROR_CHARS = 500

#: Static sanity bound on content length. The authoritative limit is
#: ``memory_max_record_chars``, enforced in the store, because a Pydantic field
#: constraint cannot read runtime Settings.
MAX_CONTENT_CHARS = 10_000


class SaveMemoryInput(BaseModel):
    """Input schema for save_memory."""

    content: str = Field(
        ...,
        min_length=1,
        max_length=MAX_CONTENT_CHARS,
        description=(
            "The single fact, preference, or task to remember, written as a "
            "short self-contained statement about the user."
        ),
    )
    category: str | None = Field(
        default=None,
        description=f"One of: {', '.join(CATEGORIES)}. Defaults to fact.",
    )
    tags: list[str] = Field(
        default_factory=list,
        max_length=MAX_TAGS,
        description=(
            f"Up to {MAX_TAGS} short keywords, each at most {MAX_TAG_CHARS} "
            "characters, to make later recall easier."
        ),
    )
    scope: str | None = Field(
        default=None,
        description=(
            f"One of: {', '.join(SCOPES)}. Use global for anything that should "
            "outlive this conversation. Defaults to the configured scope."
        ),
    )


class RecallMemoryInput(BaseModel):
    """Input schema for recall_memory."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description=(
            "Keywords describing what to look up, for example 'name', "
            "'units', or 'current project'."
        ),
    )


class ForgetMemoryInput(BaseModel):
    """Input schema for forget_memory."""

    memory_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=32,
        description=(
            "The 32-character id of one memory to delete, as shown in recall "
            "results. Takes precedence over query."
        ),
    )
    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description=(
            "Keywords selecting the memories to delete when no memory_id is "
            "known. Deletes at most 10 per call."
        ),
    )


def build_save_memory_tool(
    settings: Settings,
    *,
    store: MemoryStore | None = None,
    budget: MemoryCallBudget | None = None,
) -> BaseTool:
    """Create the save_memory tool."""

    def _run_save(
        content: str,
        category: str | None = None,
        tags: list[str] | None = None,
        scope: str | None = None,
        config: RunnableConfig = None,  # type: ignore[assignment]
    ) -> str:
        thread_id = thread_id_from_config(config)
        return _guarded(
            "save_memory",
            settings,
            store=store,
            budget=budget,
            thread_id=thread_id,
            call=lambda target: target.save(
                content=content,
                category=category,
                tags=tags,
                scope=scope,
                thread_id=thread_id,
            ),
        )

    return StructuredTool.from_function(
        func=_run_save,
        name="save_memory",
        description=(
            "Remember one durable fact, preference, or task the user states "
            "about themselves, so later conversations can use it. Call this "
            "when the user gives their name, states a preference, describes an "
            "attribute of themselves, or mentions an ongoing task."
        ),
        args_schema=SaveMemoryInput,
    )


def build_recall_memory_tool(
    settings: Settings,
    *,
    store: MemoryStore | None = None,
    budget: MemoryCallBudget | None = None,
) -> BaseTool:
    """Create the recall_memory tool."""

    def _run_recall(query: str, config: RunnableConfig = None) -> str:  # type: ignore[assignment]
        thread_id = thread_id_from_config(config)
        return _guarded(
            "recall_memory",
            settings,
            store=store,
            budget=budget,
            thread_id=thread_id,
            call=lambda target: target.recall(
                query=query,
                thread_id=thread_id,
                top_k=settings.memory_recall_top_k,
                max_chars=settings.memory_context_max_chars,
            ),
        )

    return StructuredTool.from_function(
        func=_run_recall,
        name="recall_memory",
        description=(
            "Look up what you already remember about the user, by keyword. "
            "Call this before answering a question about the user that the "
            "current conversation does not already answer, such as their name "
            "or a preference stated in an earlier session."
        ),
        args_schema=RecallMemoryInput,
    )


def build_forget_memory_tool(
    settings: Settings,
    *,
    store: MemoryStore | None = None,
    budget: MemoryCallBudget | None = None,
) -> BaseTool:
    """Create the forget_memory tool."""

    def _run_forget(
        memory_id: str | None = None,
        query: str | None = None,
        config: RunnableConfig = None,  # type: ignore[assignment]
    ) -> str:
        thread_id = thread_id_from_config(config)
        return _guarded(
            "forget_memory",
            settings,
            store=store,
            budget=budget,
            thread_id=thread_id,
            call=lambda target: target.forget(
                memory_id=memory_id,
                query=query,
                thread_id=thread_id,
            ),
        )

    return StructuredTool.from_function(
        func=_run_forget,
        name="forget_memory",
        description=(
            "Delete stored memories, by id or by keyword. Call this when the "
            "user asks you to forget something about them."
        ),
        args_schema=ForgetMemoryInput,
    )


def build_memory_tools(
    settings: Settings,
    *,
    store: MemoryStore | None = None,
    budget: MemoryCallBudget | None = None,
) -> list[BaseTool]:
    """Return the three memory tools in a fixed order.

    Both graph tool-resolution sites call this, so the two graphs cannot end up
    exposing different names or schemas.
    """

    return [
        build_save_memory_tool(settings, store=store, budget=budget),
        build_recall_memory_tool(settings, store=store, budget=budget),
        build_forget_memory_tool(settings, store=store, budget=budget),
    ]


def thread_id_from_config(config: RunnableConfig | None) -> str | None:
    """Extract ``thread_id`` from the injected run config.

    The model cannot influence this: LangChain injects the config and keeps it
    out of the tool's argument schema.
    """

    if not config:
        return None
    configurable: Any
    if isinstance(config, dict):
        configurable = config.get("configurable")
    else:  # pragma: no cover - defensive, config is a TypedDict in practice
        configurable = getattr(config, "configurable", None)
    if not isinstance(configurable, dict):
        return None
    value = configurable.get("thread_id")
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _guarded(
    operation: str,
    settings: Settings,
    *,
    store: MemoryStore | None,
    budget: MemoryCallBudget | None,
    thread_id: str | None,
    call: Callable[[MemoryStore], Any],
) -> str:
    """Run one memory operation and always return a string.

    Also enforces the per-turn call budget and emits the one success/failure log
    record per completed tool call.
    """

    active_budget = budget if budget is not None else MEMORY_BUDGET
    if not active_budget.consume(thread_id):
        return _failure(
            operation,
            f"the limit of {active_budget.limit} memory tool calls for this "
            "turn is already reached",
            record_count=None,
        )

    try:
        target = store if store is not None else get_memory_store(settings)
        outcome = call(target)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 - the agent loop must continue
        return _failure(operation, str(exc) or exc.__class__.__name__, record_count=None)

    if not getattr(outcome, "ok", False):
        return _failure(
            operation,
            getattr(outcome, "message", "unknown error"),
            record_count=getattr(outcome, "record_count", None),
        )

    _log(operation, "success", getattr(outcome, "record_count", None))
    return _truncate_success(
        _strip_marker(str(getattr(outcome, "message", ""))),
        settings.memory_context_max_chars,
    )


def _failure(operation: str, reason: str, *, record_count: int | None) -> str:
    _log(operation, "failure", record_count)
    message = f"{FAILURE_MARKER} {operation} failed: {reason}"
    if len(message) > MAX_ERROR_CHARS:
        message = message[: MAX_ERROR_CHARS - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER
    return message


def _log(operation: str, outcome: str, record_count: int | None) -> None:
    """Record one line per tool call. A broken logger must not change the result."""

    try:
        level = logging.INFO if outcome == "success" else logging.WARNING
        logger.log(
            level,
            "memory tool %s outcome=%s records=%s",
            operation,
            outcome,
            "unknown" if record_count is None else record_count,
        )
    except Exception:  # noqa: BLE001 - logging must never break a tool call
        pass


def _strip_marker(text: str) -> str:
    """Guarantee a success payload never looks like a failure."""

    if text.startswith(FAILURE_MARKER):
        return text[len(FAILURE_MARKER) :].strip()
    return text


def _truncate_success(text: str, max_chars: int) -> str:
    limit = max(1, int(max_chars))
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(TRUNCATION_MARKER))
    return text[:keep] + TRUNCATION_MARKER


__all__ = [
    "FAILURE_MARKER",
    "ForgetMemoryInput",
    "RecallMemoryInput",
    "SaveMemoryInput",
    "build_forget_memory_tool",
    "build_memory_tools",
    "build_recall_memory_tool",
    "build_save_memory_tool",
    "thread_id_from_config",
]
