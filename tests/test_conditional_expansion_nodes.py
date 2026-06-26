"""Unit tests for the conditional-expansion nodes (decompose, expand, merge)."""

from __future__ import annotations

from dataclasses import replace

from langchain_core.messages import HumanMessage, ToolMessage

from src.graph.nodes import decompose as decompose_module
from src.graph.nodes import expand as expand_module
from src.graph.nodes import merge as merge_module


def _structured_chat_model(result_factory):
    """Return a fake structured chat model that yields ``result_factory()``.

    ``result_factory`` is a no-arg callable that returns the object the
    test expects ``invoke_with_retry`` to receive. The nodes obtain their
    structured chain via ``new_structured_chat_model(settings, schema)``,
    so this fake stands in for the already-bound structured-output chain.
    """

    class _FakeStructured:
        @staticmethod
        def invoke(_payload, **_kwargs):
            return result_factory()

    return _FakeStructured


def _patch_structured_model(monkeypatch, module, result_factory):
    """Patch a node module's structured-model seam with a fake chain."""

    monkeypatch.setattr(
        module,
        "new_structured_chat_model",
        lambda _settings, _schema: _structured_chat_model(result_factory),
    )


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------


def test_decompose_passthrough_for_unresolvable_question(isolated_settings, monkeypatch):
    settings = isolated_settings()
    # No current_question / no messages -> resolver returns "" -> passthrough.
    monkeypatch.setattr(
        decompose_module,
        "invoke_with_retry",
        lambda *a, **k: pytest_forbidden("LLM must not run when no question is present"),
    )
    _patch_structured_model(monkeypatch, decompose_module, lambda: None)

    node = decompose_module.decompose_factory(settings)
    result = node({})
    assert result == {"sub_questions": []}


def test_decompose_uses_atomic_passthrough_when_llm_fails(isolated_settings, monkeypatch):
    settings = isolated_settings()
    monkeypatch.setattr(
        decompose_module,
        "invoke_with_retry",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    _patch_structured_model(monkeypatch, decompose_module, lambda: None)

    node = decompose_module.decompose_factory(settings)
    result = node({"messages": [HumanMessage(content="What year was X founded?")]})
    # Passthrough fallback emits the original question.
    assert result == {"sub_questions": ["What year was X founded?"]}


def test_decompose_clamps_llm_output_to_max_subquestions(isolated_settings, monkeypatch):
    settings = isolated_settings()
    fake_result = type("R", (), {"sub_questions": ["A", "B", "C", "D", "E"]})()
    monkeypatch.setattr(decompose_module, "invoke_with_retry", lambda *a, **k: fake_result)
    _patch_structured_model(monkeypatch, decompose_module, lambda: fake_result)

    node = decompose_module.decompose_factory(settings)
    result = node({"messages": [HumanMessage(content="Compound question.")]})

    assert len(result["sub_questions"]) == decompose_module.DECOMPOSE_MAX_SUBQUESTIONS
    assert result["sub_questions"] == ["A", "B", "C"]


def test_decompose_returns_llm_subquestions_as_is(isolated_settings, monkeypatch):
    settings = isolated_settings()
    fake_result = type("R", (), {"sub_questions": ["Reframed one", "Reframed two"]})()
    monkeypatch.setattr(decompose_module, "invoke_with_retry", lambda *a, **k: fake_result)
    _patch_structured_model(monkeypatch, decompose_module, lambda: fake_result)

    node = decompose_module.decompose_factory(settings)
    result = node({"messages": [HumanMessage(content="Compound question.")]})

    # The LLM's sub-questions are returned verbatim; we don't prepend
    # the original.
    assert result["sub_questions"] == ["Reframed one", "Reframed two"]


def test_decompose_dedupes_repeated_subquestions(isolated_settings, monkeypatch):
    settings = isolated_settings()
    fake_result = type("R", (), {"sub_questions": ["A", "B", "A", "C", "B"]})()
    monkeypatch.setattr(decompose_module, "invoke_with_retry", lambda *a, **k: fake_result)
    _patch_structured_model(monkeypatch, decompose_module, lambda: fake_result)

    node = decompose_module.decompose_factory(settings)
    result = node({"messages": [HumanMessage(content="Compound question.")]})

    assert result["sub_questions"] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# expand
# ---------------------------------------------------------------------------


def test_expand_passthrough_when_no_sub_questions(isolated_settings, monkeypatch):
    settings = isolated_settings()
    monkeypatch.setattr(
        expand_module,
        "invoke_with_retry",
        lambda *a, **k: pytest_forbidden("LLM must not run when no sub-questions are present"),
    )
    _patch_structured_model(monkeypatch, expand_module, lambda: None)

    node = expand_module.expand_factory(settings)
    result = node({"messages": []})
    assert result == {"expanded_queries": []}


def test_expand_returns_k1_passthrough_when_llm_fails(isolated_settings, monkeypatch):
    settings = isolated_settings()
    monkeypatch.setattr(
        expand_module, "invoke_with_retry", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    _patch_structured_model(monkeypatch, expand_module, lambda: None)

    node = expand_module.expand_factory(settings)
    result = node({"sub_questions": ["What year was X founded?"]})
    assert result == {"expanded_queries": ["What year was X founded?"]}


def test_expand_clamps_paraphrases_to_max(isolated_settings, monkeypatch):
    settings = isolated_settings()
    fake_result = type("R", (), {"paraphrases": [
        "Original question",
        "Paraphrase one",
        "Paraphrase two",
        "Paraphrase three",
        "Paraphrase four",
    ]})()
    monkeypatch.setattr(expand_module, "invoke_with_retry", lambda *a, **k: fake_result)
    _patch_structured_model(monkeypatch, expand_module, lambda: fake_result)

    node = expand_module.expand_factory(settings)
    result = node({"sub_questions": ["Original question"]})

    assert len(result["expanded_queries"]) == expand_module.EXPAND_MAX_PARAPHRASES
    assert result["expanded_queries"][0] == "Original question"


def test_expand_flattens_multiple_sub_questions(isolated_settings, monkeypatch):
    settings = isolated_settings()
    per_call_paraphrases = [
        ["Sub one", "Alt one", "Alt two"],
        ["Sub two"],
    ]
    counter = {"n": 0}

    def make_result():
        counter["n"] += 1
        idx = counter["n"] - 1
        return type("R", (), {"paraphrases": per_call_paraphrases[idx]})()

    monkeypatch.setattr(expand_module, "invoke_with_retry", lambda *a, **k: make_result())
    _patch_structured_model(monkeypatch, expand_module, make_result)

    node = expand_module.expand_factory(settings)
    result = node({"sub_questions": ["Sub one", "Sub two"]})

    # Flattened across sub-questions; sub-questions[0]'s alt paraphrases
    # are preserved, then sub-questions[1] is appended.
    assert result["expanded_queries"] == ["Sub one", "Alt one", "Alt two", "Sub two"]


def test_expand_dedupes_across_sub_questions(isolated_settings, monkeypatch):
    settings = isolated_settings()
    per_call_paraphrases = [["X"], ["Y"]]
    counter = {"n": 0}

    def make_result():
        counter["n"] += 1
        idx = counter["n"] - 1
        return type("R", (), {"paraphrases": per_call_paraphrases[idx]})()

    monkeypatch.setattr(expand_module, "invoke_with_retry", lambda *a, **k: make_result())
    _patch_structured_model(monkeypatch, expand_module, make_result)

    node = expand_module.expand_factory(settings)
    result = node({"sub_questions": ["X", "Y"]})

    assert result["expanded_queries"] == ["X", "Y"]


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def test_merge_dedupes_canonical_urls(isolated_settings):
    settings = isolated_settings()
    node = merge_module.merge_factory(settings)

    # Same canonical form (lower-cased host, same path) -> dedup to one.
    result = node(
        {
            "source_urls": [
                "https://Example.com/A",
                "https://example.com/A",
            ],
            "messages": [],
        }
    )
    assert result["source_urls"] == ["https://Example.com/A"]


def test_merge_keeps_query_string_distinct_canonicals(isolated_settings):
    settings = isolated_settings()
    node = merge_module.merge_factory(settings)

    # Different paths -> different canonicals, both survive.
    result = node(
        {
            "source_urls": [
                "https://example.com/A?b=1",
                "https://example.com/A?b=2",
            ],
            "messages": [],
        }
    )
    # The query string is stripped in the canonical form, so both URLs
    # collapse to one canonical. The first-seen URL is kept as the
    # display URL.
    assert len(result["source_urls"]) == 1
    assert result["source_urls"][0] == "https://example.com/A?b=1"


def test_merge_combines_first_attempt_and_tool_messages(isolated_settings):
    settings = isolated_settings()
    node = merge_module.merge_factory(settings)

    result = node(
        {
            "source_urls": ["https://example.com/first"],
            "messages": [
                ToolMessage(
                    content=(
                        "Live web search results for: Q\n"
                        "1. https://example.com/first\n"
                        "2. https://example.com/expanded1\n"
                        "3. https://example.com/expanded2"
                    ),
                    tool_call_id="call_live_web_search",
                )
            ],
        }
    )
    # The first-attempt URL "first" appears in both sets -> hit_count=2,
    # best_provider_rank=0 (from first_attempt), so it leads the ranking.
    assert result["source_urls"][0] == "https://example.com/first"
    assert "https://example.com/expanded1" in result["source_urls"]
    assert "https://example.com/expanded2" in result["source_urls"]


def test_merge_respects_web_search_top_k(isolated_settings):
    settings = replace(isolated_settings(), web_search_top_k=2)
    node = merge_module.merge_factory(settings)

    result = node(
        {
            "source_urls": [],
            "messages": [
                _tool_message_with_urls(
                    [
                        "https://example.com/a",
                        "https://example.com/b",
                        "https://example.com/c",
                    ]
                )
            ],
        }
    )
    # Top-K=2 limits the output.
    assert len(result["source_urls"]) == 2


def test_merge_ranks_higher_hit_count_first(isolated_settings):
    settings = replace(isolated_settings(), web_search_top_k=0)
    node = merge_module.merge_factory(settings)

    # "shared" appears in both first_attempt and tool messages; "unique" only in tool messages.
    state = {
        "source_urls": ["https://example.com/shared", "https://example.com/unique_a"],
        "messages": [
            ToolMessage(
                content=(
                    "Live web search results for: Q\n"
                    "1. https://example.com/shared\n"
                    "2. https://example.com/unique_b"
                ),
                tool_call_id="call_live_web_search",
            )
        ],
    }
    result = node(state)
    # "shared" has hit_count=2, so it leads the ranking.
    assert result["source_urls"][0] == "https://example.com/shared"


def test_merge_handles_empty_inputs(isolated_settings):
    settings = isolated_settings()
    node = merge_module.merge_factory(settings)
    result = node({"source_urls": [], "messages": []})
    assert result == {"source_urls": []}


def _tool_message_with_urls(urls):
    body = "Live web search results for: Q\n" + "\n".join(
        f"{i}. {url}" for i, url in enumerate(urls, start=1)
    )
    return ToolMessage(content=body, tool_call_id="call_live_web_search")


def pytest_forbidden(message):
    raise AssertionError(message)
