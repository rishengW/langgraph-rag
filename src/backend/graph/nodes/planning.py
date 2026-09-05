"""Optional bounded planning, sub-goal execution, and answer reflection nodes.

The nodes persist only concise plans, evidence summaries, statuses, and errors.
They never request or retain hidden chain-of-thought. Every LLM-backed node has
a deterministic fallback and every execution or reflection loop is bounded.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.config import Settings
from src.utils.retry import invoke_with_retry

from .common import (
    QuestionResolver,
    chat_question_resolver,
    message_text,
    new_chat_model,
    new_structured_chat_model,
)

logger = logging.getLogger(__name__)

MAX_PLAN_SUBGOALS = 8
MAX_REFLECTION_RETRIES = 2
MAX_SCRATCHPAD_CHARS = 8_000
MAX_SUBGOAL_RESULT_CHARS = 6_000

SubgoalStatus = Literal["pending", "in_progress", "completed", "failed"]


class PlanSubgoal(BaseModel):
    """Structured representation returned by the planner."""

    id: str = Field(description="Stable short identifier, for example sg-1")
    description: str = Field(description="A self-contained task to execute")
    dependencies: list[str] = Field(default_factory=list)
    status: SubgoalStatus = "pending"
    result: str = ""
    reasoning_scratchpad: str = ""


class PlanResult(BaseModel):
    """LLM output for a bounded dependency-aware plan."""

    subgoals: list[PlanSubgoal] = Field(default_factory=list)
    reasoning_scratchpad: str = ""


class AnswerCritique(BaseModel):
    """Three-dimensional final-answer quality assessment."""

    correctness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    groundedness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    critique_notes: str = ""
    revision_suggestions: str = ""


class _CritiqueResult(AnswerCritique):
    """Private alias retained for structured-output provider compatibility."""


def planner_node(
    settings: Settings,
    question_resolver: QuestionResolver = chat_question_resolver,
    *,
    max_subgoals: int = MAX_PLAN_SUBGOALS,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the optional planner node.

    The model is asked for independently executable sub-goals, dependency
    identifiers, and a short persisted rationale.  The returned plan is
    normalized before it enters graph state: identifiers are unique, unknown
    dependencies are removed, cycles are broken, and the list is clamped.
    """

    limit = max(1, min(int(max_subgoals), MAX_PLAN_SUBGOALS))

    def plan(state: dict[str, Any]) -> dict[str, Any]:
        question = _resolve_question(state, question_resolver)
        if not question:
            return {"plan": [], "global_scratchpad": "No user question was available."}

        context = _planning_context(state)
        prompt = (
            "Create a concise dependency-aware plan for the user request below.\n"
            f"Return at most {limit} sub-goals. Each sub-goal must be self-contained, "
            "use ids sg-1, sg-2, ... and list only ids it truly depends on. "
            "Mark every new sub-goal pending. Include a short rationale in "
            "reasoning_scratchpad; do not include hidden chain-of-thought or private "
            "deliberation. Return one JSON object with keys subgoals and "
            "reasoning_scratchpad.\n\n"
            f"User request:\n{question}\n\n"
            f"Available context:\n{context or '(none)'}"
        )

        try:
            result = invoke_with_retry(
                new_structured_chat_model(settings, PlanResult),
                [HumanMessage(content=prompt)],
                max_retries=settings.dashscope_max_retries,
            )
            raw_items = getattr(result, "subgoals", None)
            rationale = str(getattr(result, "reasoning_scratchpad", "") or "")
        except Exception as exc:  # noqa: BLE001 - planner must not stop a graph
            logger.warning("Planner failed; using one atomic sub-goal: %s", exc)
            raw_items, rationale = [], "Planner unavailable; using the original request."

        normalized = normalize_plan(raw_items, question, max_subgoals=limit)
        if not normalized:
            normalized = [
                {
                    "id": "sg-1",
                    "description": question,
                    "dependencies": [],
                    "status": "pending",
                    "result": "",
                    "reasoning_scratchpad": "",
                }
            ]
        scratchpad = _append_scratchpad(
            str(state.get("global_scratchpad") or ""),
            rationale or f"Planned {len(normalized)} bounded sub-goal(s).",
        )
        return {
            "plan": normalized,
            "global_scratchpad": scratchpad,
            "planning_question": question,
            "planning_input_context": context,
            "planning_context": "",
            "planning_run_id": max(0, int(state.get("planning_run_id", 0) or 0)) + 1,
            "subgoal_results": [],
            "reflection_retry_count": 0,
            "answer_critique": None,
        }

    return plan


def normalize_plan(
    raw_items: Any,
    question: str = "",
    *,
    max_subgoals: int = MAX_PLAN_SUBGOALS,
    preserve_status: bool = False,
) -> list[dict[str, Any]]:
    """Normalize model/user plan data into the state-compatible dictionaries."""

    limit = max(1, min(int(max_subgoals), MAX_PLAN_SUBGOALS))
    values = raw_items if isinstance(raw_items, list) else []
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(values[:limit], start=1):
        if isinstance(item, BaseModel):
            item = item.model_dump()
        if not isinstance(item, Mapping):
            continue
        description = str(item.get("description") or "").strip()
        if not description:
            continue
        raw_id = str(item.get("id") or f"sg-{index}").strip()
        identifier = raw_id or f"sg-{index}"
        if identifier in seen:
            identifier = f"sg-{index}"
        while identifier in seen:
            index += 1
            identifier = f"sg-{index}"
        seen.add(identifier)
        dependencies = item.get("dependencies")
        deps = [str(value).strip() for value in dependencies] if isinstance(dependencies, list) else []
        status = str(item.get("status") or "pending")
        if status not in {"pending", "in_progress", "completed", "failed"}:
            status = "pending"
        output.append(
            {
                "id": identifier,
                "description": description,
                "dependencies": deps,
                "status": status if preserve_status else "pending",
                "result": str(item.get("result") or "")[:MAX_SUBGOAL_RESULT_CHARS],
                "reasoning_scratchpad": str(item.get("reasoning_scratchpad") or "")[
                    :MAX_SCRATCHPAD_CHARS
                ],
            }
        )

    if not output and question.strip():
        return [
            {
                "id": "sg-1",
                "description": question.strip(),
                "dependencies": [],
                "status": "pending",
                "result": "",
                "reasoning_scratchpad": "",
            }
        ]

    known = {item["id"] for item in output}
    for item in output:
        item["dependencies"] = [
            dep for dep in item["dependencies"] if dep in known and dep != item["id"]
        ]
    _break_dependency_cycles(output)
    return output


def subgoal_dispatcher_node(
    state: dict[str, Any],
    *,
    max_dispatch: int = MAX_PLAN_SUBGOALS,
) -> dict[str, Any]:
    """Mark dependency-ready sub-goals in progress for LangGraph fan-out.

    The node does not execute work itself.  Use :func:`route_subgoals` as the
    conditional edge to emit one ``Send`` per ready sub-goal, allowing
    LangGraph to run independent branches concurrently.
    """

    plan = normalize_plan(
        state.get("plan"),
        max_subgoals=max_dispatch,
        preserve_status=True,
    )
    by_id = {str(item["id"]): item for item in plan}

    # Propagate failed prerequisites before selecting the next wave. This turns
    # malformed or partially failed plans into a terminal state instead of a
    # dispatcher/aggregator loop with permanently pending work.
    changed = True
    while changed:
        changed = False
        for item in plan:
            if item["status"] != "pending":
                continue
            failed_dependencies = [
                dep
                for dep in item["dependencies"]
                if by_id.get(dep, {}).get("status") == "failed"
            ]
            if failed_dependencies:
                item["status"] = "failed"
                item["reasoning_scratchpad"] = (
                    "Blocked by failed dependencies: " + ", ".join(failed_dependencies)
                )[:MAX_SCRATCHPAD_CHARS]
                changed = True

    ready: list[dict[str, Any]] = []
    for item in plan:
        if len(ready) >= max(1, int(max_dispatch)):
            break
        if item["status"] != "pending":
            continue
        if all(by_id.get(dep, {}).get("status") == "completed" for dep in item["dependencies"]):
            item["status"] = "in_progress"
            ready.append(dict(item))

    # normalize_plan removes unknown dependencies and breaks cycles. Keep one
    # final defensive escape hatch for externally supplied stale state.
    if not ready and any(item["status"] == "pending" for item in plan):
        blocked = next(item for item in plan if item["status"] == "pending")
        blocked["status"] = "failed"
        blocked["reasoning_scratchpad"] = "No dependency-ready execution path remained."
    note = f"Dispatched {len(ready)} dependency-ready sub-goal(s)."
    return {
        "plan": plan,
        "dispatched_subgoals": ready,
        "global_scratchpad": _append_scratchpad(str(state.get("global_scratchpad") or ""), note),
    }


def subgoal_worker_node(
    settings: Settings,
    question_resolver: QuestionResolver = chat_question_resolver,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build one independent sub-agent worker for a ``Send`` branch.

    Workers receive a serialized sub-goal plus the preserved original request
    and concise planning context. This keeps branches isolated while retaining
    the information needed to execute each task. The ``subgoal_results``
    reducer joins outputs from the current wave.
    """

    def execute(state: dict[str, Any]) -> dict[str, Any]:
        raw_goal = state.get("subgoal") or state.get("active_subgoal") or {}
        if not isinstance(raw_goal, Mapping):
            return {"subgoal_results": []}
        goal_id = str(raw_goal.get("id") or "sg-unknown")
        run_id = max(0, int(state.get("planning_run_id", 0) or 0))
        description = str(raw_goal.get("description") or "").strip()
        if not description:
            return {
                "subgoal_results": [
                    {
                        "id": goal_id,
                        "result": "",
                        "status": "failed",
                        "error": "empty sub-goal",
                        "planning_run_id": run_id,
                    }
                ]
            }
        question = _resolve_question(state, question_resolver)
        prompt = (
            "Execute this independent sub-goal for a larger user request. Return a "
            "concise evidence-focused result, not a plan and not hidden chain-of-thought. "
            "If the available context is insufficient, state that clearly.\n\n"
            f"Original request:\n{question}\n\nSub-goal:\n{description}\n\n"
            f"Context:\n{_planning_context(state) or '(none)'}"
        )
        try:
            response = invoke_with_retry(
                new_chat_model(settings),
                [HumanMessage(content=prompt)],
                max_retries=settings.dashscope_max_retries,
            )
            result = message_text(response).strip()[:MAX_SUBGOAL_RESULT_CHARS]
            status: SubgoalStatus = "completed" if result else "failed"
            error = "" if result else "worker returned an empty result"
        except Exception as exc:  # noqa: BLE001 - one branch must not abort siblings
            logger.warning("Sub-goal worker %s failed: %s", goal_id, exc)
            result, status, error = "", "failed", type(exc).__name__
        return {
            "subgoal_results": [
                {
                    "id": goal_id,
                    "result": result,
                    "status": status,
                    "error": error,
                    "reasoning_scratchpad": f"Worker completed {goal_id}.",
                    "planning_run_id": run_id,
                }
            ]
        }

    return execute


def route_subgoals(
    state: dict[str, Any],
    *,
    target: str = "execute_subgoal",
    empty_target: str = "subgoal_aggregator",
) -> list[Any] | str:
    """Return worker ``Send`` objects or continue when no work is ready."""

    from langgraph.constants import Send

    dispatched = state.get("dispatched_subgoals") or []
    planning_question = str(
        state.get("planning_question") or state.get("current_question") or ""
    ).strip()
    planning_input_context = str(
        state.get("planning_input_context") or state.get("planning_context") or ""
    )[-MAX_SCRATCHPAD_CHARS:]
    sends: list[Any] = []
    for item in dispatched:
        if isinstance(item, Mapping):
            sends.append(
                Send(
                    target,
                    {
                        "subgoal": dict(item),
                        "current_question": planning_question,
                        "planning_question": planning_question,
                        "planning_input_context": planning_input_context,
                        "planning_run_id": max(
                            0, int(state.get("planning_run_id", 0) or 0)
                        ),
                    },
                )
            )
    return sends if sends else empty_target


def subgoal_aggregator_node(state: dict[str, Any]) -> dict[str, Any]:
    """Merge parallel worker outputs and advance each sub-goal status.

    Workers may publish ``subgoal_results`` as ``{id: result}``, a list of
    ``{"id", "result", "error"}`` records, or a single ``subgoal_result``
    record.  This permissive contract makes the node usable with both Send
    branches and custom executors.
    """

    plan = normalize_plan(
        state.get("plan"),
        max_subgoals=MAX_PLAN_SUBGOALS,
        preserve_status=True,
    )
    active_run_id = max(0, int(state.get("planning_run_id", 0) or 0)) or None
    results = _result_records(state, planning_run_id=active_run_id)
    dispatched_ids = {
        str(item.get("id"))
        for item in state.get("dispatched_subgoals") or []
        if isinstance(item, Mapping) and item.get("id")
    }
    for item in plan:
        record = results.get(item["id"])
        if record is None:
            if item["status"] == "in_progress" and item["id"] in dispatched_ids:
                item["status"] = "failed"
                item["reasoning_scratchpad"] = "Worker returned no result."
            continue
        if isinstance(record, Mapping):
            value = record.get("result", "")
            error = record.get("error")
            status = str(record.get("status") or "")
            scratchpad = str(record.get("reasoning_scratchpad") or "")
        else:
            value, error, status, scratchpad = record, None, "", ""
        item["result"] = str(value or "")[:MAX_SUBGOAL_RESULT_CHARS]
        item["reasoning_scratchpad"] = scratchpad[:MAX_SCRATCHPAD_CHARS]
        item["status"] = "failed" if error or status == "failed" else "completed"

    completed = sum(item["status"] == "completed" for item in plan)
    failed = sum(item["status"] == "failed" for item in plan)
    note = f"Aggregated sub-goal results: {completed} completed, {failed} failed."
    return {
        "plan": plan,
        "planning_context": _plan_context(plan),
        "dispatched_subgoals": [],
        "global_scratchpad": _append_scratchpad(str(state.get("global_scratchpad") or ""), note),
    }


def route_after_subgoal_aggregation(state: dict[str, Any]) -> str:
    """Run another dependency-ready wave while bounded plan work remains."""

    plan = normalize_plan(
        state.get("plan"),
        max_subgoals=MAX_PLAN_SUBGOALS,
        preserve_status=True,
    )
    if any(item["status"] in {"pending", "in_progress"} for item in plan):
        return "subgoal_dispatcher"
    return "agent"


def answer_self_critique_node(
    settings: Settings,
    question_resolver: QuestionResolver = chat_question_resolver,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a post-answer critic for correctness, grounding, and completeness."""

    def critique(state: dict[str, Any]) -> dict[str, Any]:
        answer = _latest_answer(state)
        question = _resolve_question(state, question_resolver)
        context = _answer_context(state)
        if not answer:
            return {"answer_critique": _critique_dict(0.0, 0.0, 0.0, "No answer was generated.", "Generate an answer.")}
        prompt = (
            "Critique the final answer against the question and evidence. Score each "
            "dimension from 0 to 1: correctness, groundedness, completeness. "
            "Groundedness means claims are supported by supplied context; do not "
            "reward plausible unsupported facts. Return one JSON object with keys "
            "correctness_score, groundedness_score, completeness_score, "
            "critique_notes, revision_suggestions.\n\n"
            f"Question:\n{question}\n\nAnswer:\n{answer}\n\nEvidence/context:\n{context or '(none)'}"
        )
        try:
            result = invoke_with_retry(
                new_structured_chat_model(settings, _CritiqueResult),
                [HumanMessage(content=prompt)],
                max_retries=settings.dashscope_max_retries,
            )
            critique_value = _critique_dict(
                _score(getattr(result, "correctness_score", 0.0)),
                _score(getattr(result, "groundedness_score", 0.0)),
                _score(getattr(result, "completeness_score", 0.0)),
                str(getattr(result, "critique_notes", "") or ""),
                str(getattr(result, "revision_suggestions", "") or ""),
            )
        except Exception as exc:  # noqa: BLE001 - critique cannot break answer delivery
            logger.warning("Answer critique failed; allowing answer through: %s", exc)
            critique_value = _critique_dict(1.0, 1.0, 1.0, "Critique unavailable.", "")
        return {"answer_critique": critique_value}

    return critique


def reflection_revise_node(
    settings: Settings,
    question_resolver: QuestionResolver = chat_question_resolver,
    *,
    max_retries: int = MAX_REFLECTION_RETRIES,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a bounded revision node that uses the latest critic feedback."""

    budget = max(0, int(max_retries))

    def revise(state: dict[str, Any]) -> dict[str, Any]:
        retry_count = max(0, int(state.get("reflection_retry_count", 0) or 0))
        if retry_count >= budget:
            return {"reflection_retry_count": retry_count}
        answer = _latest_answer(state)
        question = _resolve_question(state, question_resolver)
        critique = state.get("answer_critique") or {}
        context = _answer_context(state)
        prompt = (
            "Revise the answer using the critic feedback. Preserve supported facts, "
            "remove unsupported claims, and address completeness gaps. Return only "
            "the revised user-facing answer; never mention this critique or private "
            "reasoning.\n\n"
            f"Question:\n{question}\n\nCurrent answer:\n{answer}\n\n"
            f"Critique:\n{_compact_value(critique)}\n\nEvidence:\n{context or '(none)'}"
        )
        try:
            response = invoke_with_retry(
                new_chat_model(settings),
                [HumanMessage(content=prompt)],
                max_retries=settings.dashscope_max_retries,
            )
            revised = message_text(response).strip()
        except Exception as exc:  # noqa: BLE001 - preserve previous answer on failure
            logger.warning("Reflection revision failed; retaining current answer: %s", exc)
            revised = answer
        if not revised:
            revised = answer
        return {
            "messages": [AIMessage(content=revised)],
            "reflection_retry_count": retry_count + 1,
            "global_scratchpad": _append_scratchpad(
                str(state.get("global_scratchpad") or ""),
                f"Reflection revision attempt {retry_count + 1} completed.",
            ),
        }

    return revise


def route_after_self_critique(
    state: dict[str, Any],
    *,
    max_retries: int = MAX_REFLECTION_RETRIES,
    threshold: float = 0.7,
) -> str:
    """Route poor answers to revision while enforcing the retry budget."""

    critique = state.get("answer_critique") or {}
    scores = [
        _score(critique.get("correctness_score")),
        _score(critique.get("groundedness_score")),
        _score(critique.get("completeness_score")),
    ]
    retries = max(0, int(state.get("reflection_retry_count", 0) or 0))
    if retries < max(0, int(max_retries)) and min(scores) < float(threshold):
        return "reflection_revise"
    return "__end__"


def _resolve_question(state: dict[str, Any], resolver: QuestionResolver) -> str:
    preserved = str(
        state.get("planning_question") or state.get("current_question") or ""
    ).strip()
    try:
        return str(resolver(state) or preserved).strip()
    except Exception:  # noqa: BLE001 - optional node should be defensive
        return preserved


def _planning_context(state: dict[str, Any]) -> str:
    values: list[str] = []
    preserved = str(state.get("planning_input_context") or "").strip()
    if preserved:
        values.append(preserved)
    for message in list(state.get("messages") or [])[-4:]:
        text = message_text(message).strip()
        if text:
            values.append(text)
    return "\n\n".join(values)[-MAX_SCRATCHPAD_CHARS:]


def _answer_context(state: dict[str, Any]) -> str:
    values: list[str] = []
    for message in list(state.get("messages") or []):
        role = str(getattr(message, "type", "") or "").lower()
        if role in {"tool", "function"}:
            text = message_text(message).strip()
            if text:
                values.append(text)
    for key in ("subgoal_results", "plan", "web_search_results"):
        value = state.get(key)
        if value:
            values.append(_compact_value(value))
    return "\n\n".join(values)[-MAX_SCRATCHPAD_CHARS:]


def _latest_answer(state: dict[str, Any]) -> str:
    for message in reversed(list(state.get("messages") or [])):
        role = str(getattr(message, "type", "") or "").lower()
        if role in {"ai", "assistant"} and not getattr(message, "tool_calls", None):
            text = message_text(message).strip()
            if text:
                return text
    return ""


def _result_records(
    state: dict[str, Any], *, planning_run_id: int | None = None
) -> dict[str, Any]:
    raw = state.get("subgoal_results")
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}
    records: dict[str, Any] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping) or not item.get("id"):
                continue
            if planning_run_id is not None and int(item.get("planning_run_id", -1)) != planning_run_id:
                continue
            records[str(item["id"])] = item
    item = state.get("subgoal_result")
    if (
        isinstance(item, Mapping)
        and item.get("id")
        and (
            planning_run_id is None
            or int(item.get("planning_run_id", -1)) == planning_run_id
        )
    ):
        records[str(item["id"])] = item
    return records


def _break_dependency_cycles(plan: list[dict[str, Any]]) -> None:
    by_id = {item["id"]: item for item in plan}
    for item in plan:
        path: list[str] = []
        current = item["id"]
        while current in by_id:
            if current in path:
                predecessor = path[-1]
                by_id[predecessor]["dependencies"] = [
                    dep for dep in by_id[predecessor].get("dependencies", []) if dep != current
                ]
                break
            path.append(current)
            deps = by_id[current].get("dependencies") or []
            current = str(deps[0]) if deps else ""


def _append_scratchpad(existing: str, note: str) -> str:
    value = "\n".join(part for part in (existing.strip(), note.strip()) if part)
    return value[-MAX_SCRATCHPAD_CHARS:]


def _compact_value(value: Any) -> str:
    text = str(value)
    return text[-MAX_SCRATCHPAD_CHARS:]


def _plan_context(plan: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for item in plan:
        status = str(item.get("status") or "pending")
        description = str(item.get("description") or "").strip()
        result = str(item.get("result") or "").strip()
        if description:
            lines.append(f"[{status}] {description}: {result}" if result else f"[{status}] {description}")
    return "\n".join(lines)[-MAX_SCRATCHPAD_CHARS:]


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _critique_dict(
    correctness: float,
    groundedness: float,
    completeness: float,
    notes: str,
    suggestions: str,
) -> dict[str, Any]:
    return {
        "correctness_score": _score(correctness),
        "groundedness_score": _score(groundedness),
        "completeness_score": _score(completeness),
        "critique_notes": notes[:MAX_SCRATCHPAD_CHARS],
        "revision_suggestions": suggestions[:MAX_SCRATCHPAD_CHARS],
    }


__all__ = [
    "AnswerCritique",
    "MAX_PLAN_SUBGOALS",
    "MAX_REFLECTION_RETRIES",
    "PlanResult",
    "PlanSubgoal",
    "answer_self_critique_node",
    "normalize_plan",
    "planner_node",
    "reflection_revise_node",
    "route_after_self_critique",
    "route_after_subgoal_aggregation",
    "route_subgoals",
    "subgoal_aggregator_node",
    "subgoal_dispatcher_node",
    "subgoal_worker_node",
]
