from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "chat" / "static" / "script.js"
INDEX = ROOT / "src" / "chat" / "static" / "index.html"


def _script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _index() -> str:
    return INDEX.read_text(encoding="utf-8")


def test_script_contains_no_literal_nul_bytes() -> None:
    assert b"\x00" not in SCRIPT.read_bytes()


def test_math_tokens_restore_escaped_text_not_raw_content() -> None:
    script = _script()

    assert "mathTokens.push(escapeHtml(m));" in script
    assert "mathTokens.push(m);" not in script
    assert "renderMath(bubble);" in script


def test_markdown_links_are_protected_before_html_escaping() -> None:
    script = _script()
    link_token_index = script.index("const linkTokens = [];")
    escape_index = script.index("value = escapeHtml(value);", link_token_index)

    assert link_token_index < escape_index
    assert '.replace(/&amp;/gi, "&")' in script
    assert "${escapeHtml(label)}</a>" in script
    assert "value.replaceAll(`\\u0000LINK${index}\\u0000`, () => html);" in script


def test_amap_config_is_cached_and_loaded_lazily_from_public_contract() -> None:
    script = _script()

    assert 'apiGet("/chat/config")' in script
    assert "chatConfigCache" in script
    assert "chatConfigPromise" in script
    assert "amapLoadPromise" in script
    assert "function loadAMap()" in script
    assert "window._AMapSecurityConfig" in script
    assert "absoluteSameOriginServiceHost" in script
    assert 'src.searchParams.set("key", jsApiKey);' in script
    assert "webapi.amap.com/maps" in script
    assert "AMAP_WEB_SERVICE_KEY" not in script
    assert "AMAP_API_KEY" not in script


def test_artifacts_are_validated_capped_deduped_and_restored_from_history() -> None:
    script = _script()

    assert "const MAX_ARTIFACTS_PER_TURN = 4;" in script
    assert "const MAX_MARKERS_PER_ARTIFACT = 12;" in script
    assert "const MAX_POLYLINE_POINTS = 500;" in script
    assert "function normalizeAMapArtifact" in script
    assert "function artifactDedupeKey" in script
    assert 'raw.provider !== AMAP_PROVIDER' in script
    assert "payload.artifacts" in script
    assert "payload.artifact" in script
    assert 'eventType === "artifact"' in script
    assert 'eventType === "done"' in script
    assert "collectArtifactsFromPayload(payload, responseArtifacts);" in script
    assert "appendTurn(turn.role, turn.content, { artifacts: turn.artifacts });" in script


def test_route_cards_accept_canonical_markers_positions_and_optional_polyline() -> None:
    script = _script()

    assert "function normalizeEndpointPosition" in script
    assert "const rawMarkers = Array.isArray(raw.markers)" in script
    assert "const rawPositions = Array.isArray(raw.positions)" in script
    assert "polyline: polyline.length >= 2 ? polyline : []" in script
    assert "if (artifact.polyline.length >= 2)" in script
    assert "fitMapView(map, overlays);" in script


def test_streaming_defers_markdown_katex_and_artifact_rendering_until_final() -> None:
    script = _script()
    token_branch = script.split('eventType === "token"', maxsplit=1)[1].split(
        'eventType === "artifact"', maxsplit=1
    )[0]

    assert "el.textContent = answer;" in token_branch
    assert "renderMarkdownPreview" not in token_branch
    assert "renderMath" not in token_branch
    assert "renderFinalAssistantBubble(bubble, answer" in script
    assert "renderArtifactStack(turn, artifacts);" in script


def test_amap_cards_use_dom_text_and_safe_fallback_links() -> None:
    script = _script()

    assert "function buildAMapArtifactCard" in script
    assert "document.createElement" in script
    assert "title.textContent" in script
    assert "title.textContent = artifact.label;" in script
    assert "fallback.textContent" in script
    assert "fallback.href = artifact.fallbackUrl;" in script
    assert "safeAmapFallbackUrl" in script
    assert "url.origin === AMAP_FALLBACK_ORIGIN" in script
    assert "!url.username" in script
    assert "!url.password" in script


def test_file_artifacts_use_validated_download_cards() -> None:
    script = _script()
    index = _index()

    assert "function normalizeFileArtifact" in script
    assert "function buildFileArtifactCard" in script
    assert 'artifact.type === "file"' in script
    assert "link.download = artifact.filename" in script
    assert "safeDownloadHref" in script
    assert ".file-card" in index
    assert ".file-card__download" in index


def test_maps_are_destroyed_on_transcript_clear_new_chat_and_pagehide() -> None:
    script = _script()

    assert "const liveMapInstances = new Set();" in script
    assert "function destroyMapInstances()" in script
    assert "destroyMapInstances();\n    transcript.innerHTML = \"\";" in script
    assert 'window.addEventListener("pagehide", destroyMapInstances);' in script


def test_index_has_responsive_amap_styles_and_bumped_script_cache_version() -> None:
    index = _index()

    assert ".amap-card" in index
    assert ".amap-card__map" in index
    assert "@media (max-width: 640px)" in index
    assert '<script src="/static/script.js?v=8"></script>' in index
