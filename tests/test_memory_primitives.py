"""Tests for the pure memory primitives: relevance scoring and secret screening."""

from __future__ import annotations

import pytest

from src.memory.models import MAX_QUERY_TERMS, MIN_TERM_CHARS, MemoryRecord
from src.memory.relevance import (
    derive_query_terms,
    normalize_content,
    rank_records,
    relevance_score,
    searchable_text,
)
from src.memory.secrets import SECRET_PATTERNS, find_secret_in_any, find_secret_match

TS = "2026-07-01T00:00:00+00:00"


def record(
    rid: str,
    content: str,
    *,
    tags: tuple[str, ...] = (),
    updated_at: str = TS,
    scope: str = "global",
    scope_id: str | None = None,
    category: str = "fact",
    created_at: str = TS,
    last_recalled_at: str | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=rid,
        scope=scope,  # type: ignore[arg-type]
        scope_id=scope_id,
        category=category,  # type: ignore[arg-type]
        content=content,
        tags=tags,
        created_at=created_at,
        updated_at=updated_at,
        last_recalled_at=last_recalled_at,
    )


# ---- normalization ---------------------------------------------------------


def test_normalize_content_casefolds_and_collapses_whitespace():
    assert normalize_content("  Hello\t\tWORLD \n again ") == "hello world again"
    assert normalize_content("") == ""
    assert normalize_content("   ") == ""


# ---- query term derivation -------------------------------------------------


def test_derive_query_terms_drops_short_and_duplicate_terms():
    assert derive_query_terms("I do like Metric units, I do") == (
        "do",
        "like",
        "metric",
        "units,",
    )


def test_derive_query_terms_skips_single_character_tokens():
    assert all(len(term) >= MIN_TERM_CHARS for term in derive_query_terms("a b cd e fg"))
    assert derive_query_terms("a b cd e fg") == ("cd", "fg")


def test_derive_query_terms_is_bounded():
    query = " ".join(f"term{i:03d}" for i in range(MAX_QUERY_TERMS + 25))
    assert len(derive_query_terms(query)) == MAX_QUERY_TERMS


def test_derive_query_terms_on_blank_query_is_empty():
    assert derive_query_terms("") == ()
    assert derive_query_terms("   \t ") == ()


# ---- scoring ---------------------------------------------------------------


def test_relevance_score_counts_distinct_terms_in_content():
    rec = record("a" * 32, "Prefers metric units and dark mode")
    assert relevance_score(rec, ("metric", "units")) == 2
    assert relevance_score(rec, ("metric", "metric")) == 1
    assert relevance_score(rec, ("kilograms",)) == 0


def test_relevance_score_matches_tags_as_well_as_content():
    rec = record("b" * 32, "Prefers concise answers", tags=("style", "brevity"))
    assert relevance_score(rec, ("brevity",)) == 1
    assert searchable_text(rec) == "prefers concise answers style brevity"


def test_relevance_score_with_no_terms_is_zero():
    rec = record("c" * 32, "anything")
    assert relevance_score(rec, ()) == 0


def test_relevance_score_matches_cjk_substrings():
    """CJK text has no whitespace boundaries, so matching is substring-based."""

    rec = record("d" * 32, "\u7528\u6237\u559c\u6b22\u516c\u5236\u5355\u4f4d")
    assert relevance_score(rec, ("\u516c\u5236",)) == 1


# ---- ranking ---------------------------------------------------------------


def test_rank_records_orders_by_score_then_recency_then_id():
    high = record("2" * 32, "metric units preferred", updated_at=TS)
    older = record("1" * 32, "metric only", updated_at="2026-06-01T00:00:00+00:00")
    newer = record("3" * 32, "metric only", updated_at="2026-07-05T00:00:00+00:00")

    ranked = rank_records([older, high, newer], ("metric", "units"))

    # high scores 2; newer and older score 1 and break the tie on recency.
    assert [r.id for r in ranked] == [high.id, newer.id, older.id]


def test_rank_records_breaks_recency_ties_on_ascending_id():
    first = record("1" * 32, "metric", updated_at=TS)
    second = record("2" * 32, "metric", updated_at=TS)

    ranked = rank_records([second, first], ("metric",))

    assert [r.id for r in ranked] == [first.id, second.id]


def test_rank_records_excludes_zero_scores_and_applies_top_k():
    hit = record("1" * 32, "metric units")
    miss = record("2" * 32, "unrelated entirely")

    assert [r.id for r in rank_records([hit, miss], ("metric",))] == [hit.id]
    assert rank_records([hit, miss], ("metric",), top_k=0) == []


def test_rank_records_is_stable_across_repeated_calls():
    records = [record(f"{i:032x}", "metric units", updated_at=TS) for i in range(10)]
    terms = ("metric",)

    first = [r.id for r in rank_records(records, terms)]
    second = [r.id for r in rank_records(list(reversed(records)), terms)]

    assert first == second


def test_rank_records_tolerates_unparseable_timestamp():
    good = record("1" * 32, "metric", updated_at=TS)
    bad = record("2" * 32, "metric", updated_at="not-a-timestamp")

    ranked = rank_records([bad, good], ("metric",))

    # The unparseable timestamp sorts oldest instead of raising.
    assert [r.id for r in ranked] == [good.id, bad.id]


# ---- secret screening -----------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("-----BEGIN RSA PRIVATE KEY-----", "pem_private_key"),
    ("-----begin private key-----", "pem_private_key"),
    ("my key is sk-abcdefghijklmnopqrstuv", "sk_token"),
    ("AKIAIOSFODNN7EXAMPLE", "aws_access_key_id"),
    ("ASIAIOSFODNN7EXAMPLE", "aws_access_key_id"),
    ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz", "bearer_token"),
    ("password=hunter2hunter2", "assigned_secret"),
    ("API_KEY: 0123456789abcdef", "assigned_secret"),
    ("db_passwd = s3cret-value", "assigned_secret"),
    ("my token: abcdefghijkl", "assigned_secret"),
])
def test_find_secret_match_detects_each_pattern(text, expected):
    assert find_secret_match(text) == expected


@pytest.mark.parametrize("text", [
    "",
    "Prefers metric units",
    "My name is Ada and I like short answers",
    "sk-short",
    "bearer token",
    "the password is on a sticky note",
])
def test_find_secret_match_allows_ordinary_text(text):
    assert find_secret_match(text) is None


def test_find_secret_match_never_returns_the_secret():
    secret = "sk-abcdefghijklmnopqrstuv"
    name = find_secret_match(f"remember {secret}")

    assert name == "sk_token"
    assert secret not in name
    assert all(name != secret for name, _ in SECRET_PATTERNS)


def test_find_secret_in_any_screens_collections():
    assert find_secret_in_any(["safe", "AKIAIOSFODNN7EXAMPLE"]) == "aws_access_key_id"
    assert find_secret_in_any(["safe", "also safe"]) is None
    assert find_secret_in_any(()) is None
    assert find_secret_in_any(None) is None
    assert find_secret_in_any("password=hunter2hunter2") == "assigned_secret"
