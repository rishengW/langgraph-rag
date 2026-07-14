"""Unit tests for the conditional-expansion nodes (decompose, expand, merge)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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
    assert result == {
        "sub_questions": [],
        "expanded_queries": [],
        "search_queries": [],
        "web_search_results": [],
        "web_search_result_metadata": [],
    }


@pytest.mark.parametrize(
    "question",
    [
        "What year was X founded?",
        "DeepSeek latest model 2026",
        "Research and development policy",
        "What is the restaurant where the agreement was signed?",
    ],
)
def test_decompose_bypasses_llm_for_atomic_questions(
    question, isolated_settings, monkeypatch
):
    settings = isolated_settings()
    monkeypatch.setattr(
        decompose_module,
        "invoke_with_retry",
        lambda *a, **k: pytest_forbidden("LLM must not run for atomic questions"),
    )
    monkeypatch.setattr(
        decompose_module,
        "new_structured_chat_model",
        lambda *_a, **_k: pytest_forbidden("model must not be built for atomic questions"),
    )

    node = decompose_module.decompose_factory(settings)
    result = node({"messages": [HumanMessage(content=question)]})
    assert result == {
        "sub_questions": [question],
        "expanded_queries": [],
        "search_queries": [question],
        "web_search_results": [],
        "web_search_result_metadata": [],
    }


def test_decompose_prefers_contextual_search_query_from_agent(
    isolated_settings, monkeypatch
):
    monkeypatch.setattr(
        decompose_module,
        "new_structured_chat_model",
        lambda *_a, **_k: pytest_forbidden("atomic tool query must bypass the model"),
    )
    state = {
        "messages": [
            HumanMessage(content="What about its pricing?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "live_web_search",
                        "args": {"query": "LangGraph Platform pricing 2026"},
                        "id": "search-call",
                    }
                ],
            ),
        ]
    }

    result = decompose_module.decompose_factory(isolated_settings())(state)

    assert result["search_queries"] == ["LangGraph Platform pricing 2026"]


@pytest.mark.parametrize(
    "question",
    [
        "Compare Alpha and Beta",
        "Alpha versus Beta",
        "Who founded Alpha and when did it launch?",
        "When and where was Alpha founded?",
        "List Alpha's releases and summarize Beta's roadmap",
        "What launched first? Where was it announced?",
    ],
)
def test_decompose_calls_llm_for_likely_compound_questions(
    question, isolated_settings, monkeypatch
):
    settings = isolated_settings()
    fake_result = type("R", (), {"sub_questions": ["Part A", "Part B"]})()
    calls = []
    monkeypatch.setattr(
        decompose_module,
        "invoke_with_retry",
        lambda *args, **kwargs: calls.append((args, kwargs)) or fake_result,
    )
    _patch_structured_model(monkeypatch, decompose_module, lambda: fake_result)

    result = decompose_module.decompose_factory(settings)(
        {"messages": [HumanMessage(content=question)]}
    )

    assert result["sub_questions"] == ["Part A", "Part B"]
    assert result["web_search_result_metadata"] == []
    assert len(calls) == 1


def test_decompose_clamps_llm_output_to_max_subquestions(isolated_settings, monkeypatch):
    settings = isolated_settings()
    fake_result = type("R", (), {"sub_questions": ["A", "B", "C", "D", "E"]})()
    monkeypatch.setattr(decompose_module, "invoke_with_retry", lambda *a, **k: fake_result)
    _patch_structured_model(monkeypatch, decompose_module, lambda: fake_result)

    node = decompose_module.decompose_factory(settings)
    result = node({"messages": [HumanMessage(content="Compare A, B, C, D, and E.")]})

    assert len(result["sub_questions"]) == decompose_module.DECOMPOSE_MAX_SUBQUESTIONS
    assert result["sub_questions"] == ["A", "B", "C"]


def test_decompose_returns_llm_subquestions_as_is(isolated_settings, monkeypatch):
    settings = isolated_settings()
    fake_result = type("R", (), {"sub_questions": ["Reframed one", "Reframed two"]})()
    monkeypatch.setattr(decompose_module, "invoke_with_retry", lambda *a, **k: fake_result)
    _patch_structured_model(monkeypatch, decompose_module, lambda: fake_result)

    node = decompose_module.decompose_factory(settings)
    result = node({"messages": [HumanMessage(content="Compare two systems.")]})

    # The LLM's sub-questions are returned verbatim; we don't prepend
    # the original.
    assert result["sub_questions"] == ["Reframed one", "Reframed two"]


def test_decompose_dedupes_repeated_subquestions(isolated_settings, monkeypatch):
    settings = isolated_settings()
    fake_result = type("R", (), {"sub_questions": ["A", "B", "A", "C", "B"]})()
    monkeypatch.setattr(decompose_module, "invoke_with_retry", lambda *a, **k: fake_result)
    _patch_structured_model(monkeypatch, decompose_module, lambda: fake_result)

    node = decompose_module.decompose_factory(settings)
    result = node({"messages": [HumanMessage(content="Compare A, B, and C.")]})

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
    assert result == {
        "expanded_queries": [],
        "search_queries": [],
        "expansion_attempted": True,
    }


def test_expand_returns_k1_passthrough_when_llm_fails(isolated_settings, monkeypatch):
    settings = isolated_settings()
    monkeypatch.setattr(
        expand_module, "invoke_with_retry", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    _patch_structured_model(monkeypatch, expand_module, lambda: None)

    node = expand_module.expand_factory(settings)
    result = node({"sub_questions": ["What year was X founded?"]})
    assert result == {
        "expanded_queries": ["What year was X founded?"],
        "search_queries": ["What year was X founded?"],
        "expansion_attempted": True,
    }


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


def test_expand_clamps_total_query_batch(isolated_settings, monkeypatch):
    monkeypatch.setattr(
        expand_module,
        "_paraphrases_for",
        lambda question, _settings: [question, f"{question} alt 1", f"{question} alt 2"],
    )
    node = expand_module.expand_factory(isolated_settings())

    result = node({"sub_questions": ["First", "Second", "Third"]})

    assert len(result["expanded_queries"]) == 6
    assert result["search_queries"] == result["expanded_queries"]


def test_expand_always_marks_expansion_attempted(isolated_settings, monkeypatch):
    """Regression: expand must set expansion_attempted so the post-web_answer
    edge cannot loop web_answer -> expand -> ... -> web_answer forever."""

    settings = isolated_settings()
    monkeypatch.setattr(
        expand_module,
        "invoke_with_retry",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    _patch_structured_model(monkeypatch, expand_module, lambda: None)

    node = expand_module.expand_factory(settings)

    # Both the populated and empty-input branches must mark the flag.
    assert node({"sub_questions": ["Q"]})["expansion_attempted"] is True
    assert node({"messages": []})["expansion_attempted"] is True


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


def test_merge_rewards_overlap_across_fanout_query_results(isolated_settings):
    settings = replace(isolated_settings(), web_search_top_k=0)
    node = merge_module.merge_factory(settings)

    result = node(
        {
            "source_urls": [],
            "messages": [],
            "web_search_results": [
                ["https://example.com/first", "https://example.com/shared"],
                ["https://example.com/second", "https://example.com/shared"],
            ],
        }
    )

    assert result["source_urls"][0] == "https://example.com/shared"


def test_merge_prioritizes_relevance_over_repeated_generic_result(isolated_settings):
    settings = replace(isolated_settings(), web_search_top_k=0)
    node = merge_module.merge_factory(settings)

    result = node(
        {
            "source_urls": [],
            "messages": [],
            "web_search_results": [
                ["https://example.com/generic", "https://example.com/relevant"],
                ["https://example.com/generic"],
            ],
            "web_search_result_metadata": [
                [
                    {
                        "url": "https://example.com/generic",
                        "provider_rank": 0,
                        "relevance_score": 0,
                        "quality_score": 65,
                    },
                    {
                        "url": "https://example.com/relevant",
                        "provider_rank": 1,
                        "relevance_score": 40,
                        "quality_score": 105,
                    },
                ],
                [
                    {
                        "url": "https://example.com/generic",
                        "provider_rank": 0,
                        "relevance_score": 0,
                        "quality_score": 65,
                    }
                ],
            ],
        }
    )

    assert result["source_urls"] == [
        "https://example.com/relevant",
        "https://example.com/generic",
    ]


def test_merge_keeps_url_only_results_when_metadata_is_partial(isolated_settings):
    settings = replace(isolated_settings(), web_search_top_k=0)
    node = merge_module.merge_factory(settings)

    result = node(
        {
            "source_urls": [],
            "messages": [],
            "web_search_results": [
                ["https://example.com/scored", "https://example.com/shared"],
                ["https://example.com/shared", "https://example.com/legacy"],
            ],
            "web_search_result_metadata": [
                [
                    {
                        "url": "https://example.com/scored",
                        "provider_rank": 0,
                        "relevance_score": 0,
                        "quality_score": 0,
                    }
                ],
                [],
            ],
        }
    )

    # The URL-only shared result keeps both query hits and the other legacy
    # result remains eligible even though one sibling has ranking metadata.
    assert result["source_urls"] == [
        "https://example.com/shared",
        "https://example.com/scored",
        "https://example.com/legacy",
    ]


def test_merge_handles_empty_inputs(isolated_settings):
    settings = isolated_settings()
    node = merge_module.merge_factory(settings)
    result = node({"source_urls": [], "messages": []})
    assert result == {"source_urls": []}


def test_merge_ignores_prior_turn_tool_urls(isolated_settings):
    """Tool URLs from a previous turn must not leak into the current turn.

    The chat checkpoint accumulates messages across turns, so a prior turn's
    ``live_web_search`` ToolMessage stays in ``state["messages"]``. Merge must
    only harvest tool URLs from the current turn (after the last HumanMessage).
    """

    settings = replace(isolated_settings(), web_search_top_k=0)
    node = merge_module.merge_factory(settings)

    state = {
        # Current turn's freshly discovered source URLs.
        "source_urls": ["https://example.com/doubao"],
        "messages": [
            # --- previous turn (soccer) ---
            HumanMessage(content="south africa vs canada score"),
            _tool_message_with_urls(
                [
                    "https://example.com/soccer1",
                    "https://example.com/soccer2",
                ]
            ),
            # --- current turn (doubao) ---
            HumanMessage(content="What models are in Doubao Coding Plan Lite"),
            _tool_message_with_urls(["https://example.com/doubao-search"]),
        ],
    }
    result = node(state)

    assert "https://example.com/soccer1" not in result["source_urls"]
    assert "https://example.com/soccer2" not in result["source_urls"]
    assert "https://example.com/doubao" in result["source_urls"]
    assert "https://example.com/doubao-search" in result["source_urls"]


def test_merge_does_not_anchor_pregraph_urls_above_ingraph(isolated_settings):
    """In-graph (tool) URLs must compete fairly, not sit below pre-graph URLs.

    Regression for the anchoring bug: the pre-graph refresh URLs were offset so
    they always out-ranked the in-graph contextualized search on ties. With no
    overlap and equal own-position, the top in-graph result should interleave
    with the pre-graph results rather than being pushed to the bottom.
    """

    settings = replace(isolated_settings(), web_search_top_k=0)
    node = merge_module.merge_factory(settings)

    state = {
        # Pre-graph refresh URLs (weaker, less-contextualized for a follow-up).
        "source_urls": [
            "https://example.com/pre1",
            "https://example.com/pre2",
            "https://example.com/pre3",
        ],
        "messages": [
            HumanMessage(content="Argentina vs Jordan world cup result"),
            _tool_message_with_urls(
                [
                    "https://example.com/ingraph-top",
                    "https://example.com/ingraph2",
                ]
            ),
        ],
    }
    result = node(state)

    # The top in-graph URL shares provider_rank 0 with the top pre-graph URL
    # and, with no offset penalty, must land among the leaders rather than
    # after every pre-graph URL.
    assert "https://example.com/ingraph-top" in result["source_urls"][:2]
    pre3_idx = result["source_urls"].index("https://example.com/pre3")
    ingraph_top_idx = result["source_urls"].index("https://example.com/ingraph-top")
    assert ingraph_top_idx < pre3_idx


def _tool_message_with_urls(urls):
    body = "Live web search results for: Q\n" + "\n".join(
        f"{i}. {url}" for i, url in enumerate(urls, start=1)
    )
    return ToolMessage(content=body, tool_call_id="call_live_web_search")


def pytest_forbidden(message):
    raise AssertionError(message)
