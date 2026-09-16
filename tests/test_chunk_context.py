from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from src.backend.rag import chroma_retriever as chroma_module
from src.backend.rag.chunk_context import (
    CHUNK_CONTEXT_CACHE_FILENAME,
    ChunkContextCache,
    ChunkContextConfig,
    apply_chunk_context,
)
from src.config.loader import load_settings, load_yaml_config
from src.config.settings import Settings

CHUNK_CONTEXT_SETTING_NAMES = (
    "chunk_context_enabled",
    "chunk_context_document_excerpt_chars",
    "chunk_context_max_prefix_chars",
    "chunk_context_max_concurrency",
)


def _chunk(text: str, source: str = "https://example.test/doc") -> Document:
    return Document(page_content=text, metadata={"source": source, "title": "Doc"})


class FakeChatModel:
    """Records prompts and returns a canned reply (or raises)."""

    model = "fake-model"

    def __init__(self, reply: str = "PREFIX", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.reply)


def _enabled_config(**overrides) -> ChunkContextConfig:
    return dataclasses.replace(ChunkContextConfig(enabled=True, max_retries=1), **overrides)


def test_apply_chunk_context_disabled_is_noop():
    chunks = [_chunk("alpha"), _chunk("beta")]
    model = FakeChatModel()

    result = apply_chunk_context(chunks, config=ChunkContextConfig(enabled=False), chat_model=model)

    assert [doc.page_content for doc in result] == ["alpha", "beta"]
    assert model.calls == []


def test_apply_chunk_context_prepends_prefix_and_preserves_metadata():
    original = _chunk("原文内容")
    model = FakeChatModel(reply="本段讨论检索流程。")

    result = apply_chunk_context([original], config=_enabled_config(), chat_model=model)

    assert result[0].page_content == "本段讨论检索流程。\n\n原文内容"
    assert result[0].metadata == original.metadata
    assert result[0].metadata is not original.metadata
    assert len(model.calls) == 1


def test_apply_chunk_context_skips_empty_chunks():
    chunks = [_chunk("  "), _chunk("real text")]
    model = FakeChatModel()

    result = apply_chunk_context(chunks, config=_enabled_config(), chat_model=model)

    assert result[0].page_content == "  "
    assert result[1].page_content == "PREFIX\n\nreal text"
    assert len(model.calls) == 1


def test_apply_chunk_context_failure_keeps_original_chunk():
    chunks = [_chunk("alpha"), _chunk("beta")]
    model = FakeChatModel(error=ConnectionError("SSL EOF"))

    result = apply_chunk_context(chunks, config=_enabled_config(), chat_model=model)

    assert [doc.page_content for doc in result] == ["alpha", "beta"]
    assert len(model.calls) == 2


def test_apply_chunk_context_truncates_long_prefix():
    model = FakeChatModel(reply="x" * 500)

    result = apply_chunk_context(
        [_chunk("body")],
        config=_enabled_config(max_prefix_chars=50),
        chat_model=model,
    )

    prefix, _, body = result[0].page_content.partition("\n\n")
    assert len(prefix) == 50
    assert body == "body"


def test_apply_chunk_context_uses_per_source_excerpt():
    chunks = [
        _chunk("A1 第一段", source="https://a.test"),
        _chunk("A2 第二段", source="https://a.test"),
        _chunk("B1 另一篇", source="https://b.test"),
    ]
    model = FakeChatModel()

    apply_chunk_context(chunks, config=_enabled_config(), chat_model=model)

    assert len(model.calls) == 3
    prompt_for_a2 = model.calls[1]
    assert "A1 第一段" in prompt_for_a2  # excerpt from the same source head
    prompt_for_b1 = model.calls[2]
    assert "B1 另一篇" in prompt_for_b1
    assert "A1 第一段" not in prompt_for_b1  # excerpts never cross sources


def test_chunk_context_cache_avoids_repeat_llm_calls(tmp_path):
    cache = ChunkContextCache(tmp_path / CHUNK_CONTEXT_CACHE_FILENAME)
    chunks = [_chunk("alpha"), _chunk("beta")]

    first_model = FakeChatModel()
    apply_chunk_context(chunks, config=_enabled_config(), chat_model=first_model, cache=cache)
    assert len(first_model.calls) == 2
    assert (tmp_path / CHUNK_CONTEXT_CACHE_FILENAME).exists()

    second_model = FakeChatModel()
    result = apply_chunk_context(
        chunks, config=_enabled_config(), chat_model=second_model, cache=cache
    )
    assert second_model.calls == []
    assert all(doc.page_content.startswith("PREFIX\n\n") for doc in result)


def test_chunk_context_cache_round_trip_from_disk(tmp_path):
    path = tmp_path / CHUNK_CONTEXT_CACHE_FILENAME
    ChunkContextCache(path).set("key-1", "前缀一")

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw == {"key-1": "前缀一"}
    assert ChunkContextCache(path).get("key-1") == "前缀一"
    assert ChunkContextCache(path).get("missing") is None


def test_chunk_context_cache_tolerates_corrupt_file(tmp_path):
    path = tmp_path / CHUNK_CONTEXT_CACHE_FILENAME
    path.write_text("{not json", encoding="utf-8")

    cache = ChunkContextCache(path)
    assert cache.get("key-1") is None


def test_load_settings_accepts_chunk_context_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("CHUNK_CONTEXT_ENABLED", "true")
    monkeypatch.setenv("CHUNK_CONTEXT_DOCUMENT_EXCERPT_CHARS", "4000")
    monkeypatch.setenv("CHUNK_CONTEXT_MAX_PREFIX_CHARS", "120")
    monkeypatch.setenv("CHUNK_CONTEXT_MAX_CONCURRENCY", "0")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.chunk_context_enabled is True
    assert settings.chunk_context_document_excerpt_chars == 4000
    assert settings.chunk_context_max_prefix_chars == 120
    assert settings.chunk_context_max_concurrency == 1  # clamped to >= 1


def test_chunk_context_defaults_align_with_yaml():
    defaults = load_yaml_config("config/default.yaml")
    settings = Settings(dashscope_api_key="test-key")

    assert settings.chunk_context_enabled is False
    for name in CHUNK_CONTEXT_SETTING_NAMES:
        assert name in defaults, f"{name} missing from config/default.yaml"
        assert defaults[name] == getattr(settings, name), name


def test_chunk_context_settings_are_documented():
    env_example = Path(".env.example").read_text(encoding="utf-8")
    yaml_text = Path("config/default.yaml").read_text(encoding="utf-8")

    for name in CHUNK_CONTEXT_SETTING_NAMES:
        assert name.upper() in env_example, f"{name.upper()} missing from .env.example"
        assert name in yaml_text, f"{name} missing from config/default.yaml"


def test_embedding_config_flag_flip_forces_rebuild(tmp_path):
    base = Settings(dashscope_api_key="test-key", chroma_dir=tmp_path)
    enabled = dataclasses.replace(base, chunk_context_enabled=True)

    chroma_module._write_embedding_config(enabled)
    assert chroma_module._embedding_config_matches(enabled) is True
    assert chroma_module._embedding_config_matches(base) is False


def test_embedding_config_without_flag_is_treated_as_stale(tmp_path):
    """Stores written before this feature lack the flag and must rebuild once."""

    settings = Settings(dashscope_api_key="test-key", chroma_dir=tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / chroma_module.EMBEDDING_CONFIG_FILENAME).write_text(
        json.dumps(
            {
                "embedding_model": settings.embedding_model,
                "embedding_dimension": settings.embedding_dimension,
            }
        ),
        encoding="utf-8",
    )

    assert chroma_module._embedding_config_matches(settings) is False


def test_clear_chroma_store_preserves_context_cache(tmp_path):
    (tmp_path / "chroma.sqlite3").write_text("db", encoding="utf-8")
    (tmp_path / chroma_module.EMBEDDING_CONFIG_FILENAME).write_text("{}", encoding="utf-8")
    (tmp_path / CHUNK_CONTEXT_CACHE_FILENAME).write_text("{}", encoding="utf-8")
    (tmp_path / "chat").mkdir()
    (tmp_path / "chat" / "x").write_text("chat", encoding="utf-8")
    (tmp_path / "stale").mkdir()
    (tmp_path / "stale" / "x").write_text("stale", encoding="utf-8")

    chroma_module._clear_chroma_store(tmp_path)

    assert (tmp_path / CHUNK_CONTEXT_CACHE_FILENAME).exists()
    assert (tmp_path / "chat").is_dir()
    assert not (tmp_path / "chroma.sqlite3").exists()
    assert not (tmp_path / "stale").exists()
