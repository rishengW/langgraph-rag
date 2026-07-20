from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .common import host_authority_class, registrable_domain
from .query_constraints import STATUS_INTENT, detect_query_intents

StatusValue = Literal["open", "planned"]
ConsensusState = Literal["not_applicable", "supported", "conflicting", "insufficient"]

_OPEN_STATUS_RE = re.compile(
    r"(?:(?:现)?已(?:经)?(?:正式)?(?:开通|上线)|正式(?:开通|上线)|投入运营|正式运营|全线通车|"
    r"\bis\s+(?:now\s+)?open\b|\bopened\b|"
    r"\boperational\b|\blaunched\b|\breleased\b)",
    re.I,
)
_PLANNED_STATUS_RE = re.compile(
    r"(?:尚未开通|还未开通|未开通|预计|计划|拟于|将于|有望|试运行|"
    r"\bnot\s+yet\b|\bplanned\b|\bexpected\b|\bwill\s+(?:open|launch)\b|"
    r"\btrial\s+operation\b)",
    re.I,
)


@dataclass(frozen=True)
class StatusConsensus:
    state: ConsensusState
    categorical_allowed: bool
    supporting_domains: tuple[str, ...] = ()
    prompt_instruction: str = ""


def assess_status_consensus(pages: Sequence[Any], query: str) -> StatusConsensus:
    """Require authority or independent agreement for current-status claims."""

    if STATUS_INTENT not in detect_query_intents(query):
        return StatusConsensus(state="not_applicable", categorical_allowed=True)

    claims: dict[StatusValue, set[str]] = {"open": set(), "planned": set()}
    official_claims: dict[StatusValue, set[str]] = {"open": set(), "planned": set()}
    for page in pages:
        url = str(getattr(page, "url", "") or "")
        domain = registrable_domain(url) or url
        page_claims = _detect_page_statuses(
            " ".join(
                part
                for part in (
                    str(getattr(page, "title", "") or ""),
                    str(getattr(page, "text", "") or ""),
                )
                if part
            )
        )
        for value in page_claims:
            claims[value].add(domain)
            if host_authority_class(url) != "standard":
                official_claims[value].add(domain)

    official_states = {value for value, domains in official_claims.items() if domains}
    if len(official_states) == 1:
        value = next(iter(official_states))
        return StatusConsensus(
            state="supported",
            categorical_allowed=True,
            supporting_domains=tuple(sorted(official_claims[value])),
        )
    if len(official_states) > 1:
        return _conflicting_consensus(official_claims)

    observed_states = {value for value, domains in claims.items() if domains}
    if len(observed_states) > 1:
        return _conflicting_consensus(claims)
    if len(observed_states) == 1:
        value = next(iter(observed_states))
        domains = tuple(sorted(claims[value]))
        if len(domains) >= 2:
            return StatusConsensus(
                state="supported",
                categorical_allowed=True,
                supporting_domains=domains,
            )
        return StatusConsensus(
            state="insufficient",
            categorical_allowed=False,
            supporting_domains=domains,
            prompt_instruction=(
                "Only one non-authoritative domain supports the current-status claim. "
                "State that the status is unverified and do not present it as categorical fact."
            ),
        )

    return StatusConsensus(
        state="insufficient",
        categorical_allowed=False,
        prompt_instruction=(
            "The sources do not contain a clear current-status claim. Say that the current "
            "status could not be verified and do not infer it from plans or background text."
        ),
    )


def _detect_page_statuses(text: str) -> frozenset[StatusValue]:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    values: set[StatusValue] = set()
    if _OPEN_STATUS_RE.search(normalized):
        values.add("open")
    if _PLANNED_STATUS_RE.search(normalized):
        values.add("planned")
    return frozenset(values)


def _conflicting_consensus(
    claims: dict[StatusValue, set[str]],
) -> StatusConsensus:
    domains = tuple(sorted(set().union(*claims.values())))
    return StatusConsensus(
        state="conflicting",
        categorical_allowed=False,
        supporting_domains=domains,
        prompt_instruction=(
            "The retrieved sources conflict about the current status. Explicitly describe the "
            "conflict, cite both sides, and do not choose a definitive status unless a clearly "
            "responsible official source resolves it."
        ),
    )


__all__ = ["StatusConsensus", "assess_status_consensus"]
