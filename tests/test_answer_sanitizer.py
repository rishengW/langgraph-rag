from __future__ import annotations

from src.llm.sanitize import CitationArtifactFilter, strip_citation_artifacts

_MARKER = "\u3010199\u2020L91-L126\u3011"
_SHORT_MARKER = "\u3010199\u2020L31-L32\u3011"


def test_strips_source_index_and_line_range_markers():
    answer = (
        f"Spain defeated Argentina 1-0 in the final held in New Jersey{_MARKER}. "
        f'The same source confirms that "Spain are champions of the world again"'
        f"{_SHORT_MARKER}"
    )

    cleaned = strip_citation_artifacts(answer)

    assert "\u2020" not in cleaned
    assert "199" not in cleaned
    assert cleaned == (
        "Spain defeated Argentina 1-0 in the final held in New Jersey. "
        'The same source confirms that "Spain are champions of the world again"'
    )


def test_strips_ascii_bracket_and_tool_marker_variants():
    assert strip_citation_artifacts("Result [199\u2020L4-L9] here") == "Result here"
    assert strip_citation_artifacts("Result [oaicite:0]") == "Result"
    assert strip_citation_artifacts("Result \u3010oaicite:12\u3011") == "Result"
    assert strip_citation_artifacts("Result citeturn3search2 done") == "Result done"


def test_preserves_urls_markdown_links_and_ordinary_brackets():
    answer = (
        "Spain won (https://football360.com.au/news/final). "
        "See the [match report](https://example.com/report) and note [1] for 2026 [sic]."
    )

    assert strip_citation_artifacts(answer) == answer


def test_preserves_a_lone_dagger_and_returns_input_untouched_when_clean():
    assert strip_citation_artifacts("The dagger \u2020 is a footnote symbol.") == (
        "The dagger \u2020 is a footnote symbol."
    )
    assert strip_citation_artifacts("") == ""


def test_cleanup_removes_the_space_a_marker_leaves_before_punctuation():
    assert strip_citation_artifacts(f"Spain won the final {_MARKER}.") == "Spain won the final."


def test_streaming_filter_removes_a_marker_split_across_chunks():
    chunks = ["Spain won", "\u3010199", "\u2020L91-", "L126\u3011", " in ", "New Jersey."]
    token_filter = CitationArtifactFilter()

    streamed = "".join(token_filter.feed(chunk) for chunk in chunks) + token_filter.flush()

    assert streamed == "Spain won in New Jersey."


def test_streaming_filter_releases_text_that_is_not_a_marker():
    chunks = ["Read the ", "[docs]", "(https://a.test/d)", " for more."]
    token_filter = CitationArtifactFilter()

    streamed = "".join(token_filter.feed(chunk) for chunk in chunks) + token_filter.flush()

    assert streamed == "Read the [docs](https://a.test/d) for more."


def test_streaming_filter_flushes_an_unterminated_marker_prefix():
    token_filter = CitationArtifactFilter()

    emitted = token_filter.feed("Spain won \u3010199\u2020L91")
    flushed = token_filter.flush()

    assert emitted == "Spain won "
    # An unterminated prefix is released rather than swallowed: dropping text
    # the model actually produced would be worse than showing a stray marker.
    assert "\u3010199" in flushed


def test_streaming_filter_preserves_a_complete_answer_character_for_character():
    answer = "Spain beat Argentina 1-0 (https://example.com/final) on 19 July 2026."
    token_filter = CitationArtifactFilter()

    streamed = "".join(token_filter.feed(char) for char in answer) + token_filter.flush()

    assert streamed == answer


def test_web_answer_strips_markers_from_the_grounded_answer(monkeypatch, isolated_settings):
    """The answer node is the last place a fabricated marker can be removed."""

    import sys
    from types import ModuleType, SimpleNamespace

    from langchain_core.messages import AIMessage, HumanMessage

    from src.graph.nodes import web_answer as web_answer_module
    from src.web_search.content_fetcher import is_readable_page, is_readable_text
    from src.web_search.page_structure import PageStructure

    settings = isolated_settings(source_urls=[])
    page = SimpleNamespace(
        url="https://football360.com.au/news/final",
        title="Spain are champions of the world again",
        text=" ".join(["Spain beat Argentina in the final."] * 30),
        structure=PageStructure(shape="article", content_words=150, measured=True),
    )

    content_fetcher = ModuleType("src.web_search.content_fetcher")
    content_fetcher.FetchedPage = SimpleNamespace
    content_fetcher.fetch_pages = lambda _urls, **_kwargs: [page]
    content_fetcher.is_readable_page = is_readable_page
    content_fetcher.is_readable_text = is_readable_text
    prompt_builder = ModuleType("src.web_search.prompt_builder")
    prompt_builder.build_web_search_prompt = lambda *_args, **_kwargs: "prompt"
    monkeypatch.setitem(sys.modules, "src.web_search.content_fetcher", content_fetcher)
    monkeypatch.setitem(sys.modules, "src.web_search.prompt_builder", prompt_builder)

    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _settings: "fake-model")
    monkeypatch.setattr(
        web_answer_module,
        "invoke_with_retry",
        lambda *_args, **_kwargs: AIMessage(content=f"Spain defeated Argentina 1-0{_MARKER}."),
    )

    node = web_answer_module.web_answer_factory(settings)
    result = node(
        {
            "messages": [HumanMessage(content="Spain Argentina final")],
            "source_urls": [page.url],
        }
    )

    assert result["messages"][0].content == "Spain defeated Argentina 1-0."


def test_token_stream_drops_a_marker_split_across_token_events():
    from langchain_core.messages import AIMessageChunk

    from src.graph.executor import GraphExecutor

    chunks = ["Spain won", "\u3010199", "\u2020L91-", "L126\u3011", " in New Jersey."]

    class StubGraph:
        def invoke(self, inputs, config=None):
            return {}

        def stream(self, _inputs, **_kwargs):
            for text in chunks:
                yield (
                    "messages",
                    (
                        AIMessageChunk(content=text),
                        {"langgraph_node": "web_answer"},
                    ),
                )
            yield "updates", {"web_answer": {"messages": []}}

    executor = GraphExecutor(StubGraph())
    tokens = [
        event.token
        for event in executor.stream({}, stream_tokens=True)
        if getattr(event, "type", "") == "token"
    ]

    assert "".join(tokens) == "Spain won in New Jersey."


def test_web_search_prompt_forbids_invented_reference_markers():
    from types import SimpleNamespace

    from src.web_search.prompt_builder import build_web_search_prompt

    page = SimpleNamespace(
        url="https://example.com/final",
        title="Final result",
        text="Spain beat Argentina.",
        publication_date=None,
    )

    prompt = build_web_search_prompt("who won", [page])

    assert "full URL in parentheses" in prompt
    assert "never invent bracketed reference" in prompt.lower()
