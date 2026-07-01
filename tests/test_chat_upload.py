from __future__ import annotations

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.chat import api as chat_api
from src.tools import build_text_file_tool, build_word_tool

_DOCX_DOCUMENT_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/'
    'wordprocessingml/2006/main"><w:body>'
    "<w:p><w:r><w:t>Quarterly revenue rose 12 percent.</w:t></w:r></w:p>"
    "</w:body></w:document>"
)


def _docx_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", _DOCX_DOCUMENT_XML)
    return buffer.getvalue()


class _FakeGraph:
    def __init__(self) -> None:
        self.messages: list = []

    def get_state(self, config):
        return SimpleNamespace(values={"messages": list(self.messages)})

    def invoke(self, inputs, config):
        self.messages.extend(inputs["messages"])
        self.messages.append(AIMessage(content="ok"))
        return {"messages": list(self.messages)}


def _client(monkeypatch, isolated_settings, tmp_path: Path):
    settings = isolated_settings(
        source_urls=["https://chat-default.test"],
        web_search_enabled=False,
        file_read_enabled=True,
        file_read_root=str(tmp_path / "files"),
        file_read_max_bytes=1_000_000,
    )
    monkeypatch.setattr(chat_api, "load_settings", lambda: settings)
    monkeypatch.setattr(
        chat_api,
        "build_chat_graph",
        lambda *args, **kwargs: _FakeGraph(),
    )
    return TestClient(chat_api.create_app()), settings


def _start_thread(client: TestClient) -> str:
    start = client.post("/chat", json={"web_search": False})
    assert start.status_code == 200
    return start.json()["thread_id"]


def test_upload_injects_context_into_next_turn(monkeypatch, isolated_settings, tmp_path):
    client, _ = _client(monkeypatch, isolated_settings, tmp_path)
    with client:
        thread_id = _start_thread(client)
        client.post(
            f"/chat/{thread_id}/upload",
            files={"files": ("notes.txt", b"hello from upload", "text/plain")},
        )

        # The next turn should carry a SystemMessage announcing the upload path
        # so the LLM knows the exact path to pass to the file tools.
        registry = client.app.state.session_registry
        session = registry.get(thread_id)
        graph = session.graph

        message = client.post(
            f"/chat/{thread_id}/message", json={"message": "summarize notes.txt"}
        )
        assert message.status_code == 200

        system_texts = [
            getattr(m, "content", "")
            for m in graph.messages
            if m.__class__.__name__ == "SystemMessage"
        ]
        assert any(
            f"chat_uploads/{thread_id}/notes.txt" in text for text in system_texts
        )

        # The injected system note must not leak into the user-facing history.
        history = client.get(f"/chat/{thread_id}/history")
        roles = [turn["role"] for turn in history.json()["turns"]]
        assert "system" not in roles


def test_upload_context_not_repeated_on_later_turns(monkeypatch, isolated_settings, tmp_path):
    client, _ = _client(monkeypatch, isolated_settings, tmp_path)
    with client:
        thread_id = _start_thread(client)
        client.post(
            f"/chat/{thread_id}/upload",
            files={"files": ("notes.txt", b"hi", "text/plain")},
        )
        registry = client.app.state.session_registry
        graph = registry.get(thread_id).graph

        client.post(f"/chat/{thread_id}/message", json={"message": "first"})
        client.post(f"/chat/{thread_id}/message", json={"message": "second"})

        system_count = sum(
            1 for m in graph.messages if m.__class__.__name__ == "SystemMessage"
        )
        # Announced once on the first turn after upload, not again on the second.
        assert system_count == 1


def test_upload_text_file_then_tool_reads_it(monkeypatch, isolated_settings, tmp_path):
    client, settings = _client(monkeypatch, isolated_settings, tmp_path)
    with client:
        thread_id = _start_thread(client)

        response = client.post(
            f"/chat/{thread_id}/upload",
            files={"files": ("notes.txt", b"hello from upload", "text/plain")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["errors"] == []
        assert len(body["files"]) == 1
        saved = body["files"][0]
        assert saved["filename"] == "notes.txt"
        assert thread_id in saved["relative_path"]

        # The saved file must be readable by the file tool using the returned
        # relative path, proving the upload lands inside the file-read root.
        tool = build_text_file_tool(settings)
        result = tool.invoke({"path": saved["relative_path"]})
        assert "hello from upload" in result


def test_upload_docx_then_word_tool_extracts_text(monkeypatch, isolated_settings, tmp_path):
    client, settings = _client(monkeypatch, isolated_settings, tmp_path)
    with client:
        thread_id = _start_thread(client)

        response = client.post(
            f"/chat/{thread_id}/upload",
            files={
                "files": (
                    "report.docx",
                    _docx_bytes(),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

        assert response.status_code == 200
        saved = response.json()["files"][0]

        tool = build_word_tool(settings)
        result = tool.invoke({"path": saved["relative_path"]})
        assert "Quarterly revenue rose 12 percent." in result


def test_upload_rejects_disallowed_type(monkeypatch, isolated_settings, tmp_path):
    client, _ = _client(monkeypatch, isolated_settings, tmp_path)
    with client:
        thread_id = _start_thread(client)

        response = client.post(
            f"/chat/{thread_id}/upload",
            files={"files": ("evil.exe", b"MZ...", "application/octet-stream")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["files"] == []
        assert len(body["errors"]) == 1
        assert "unsupported file type" in body["errors"][0].lower()


def test_upload_unknown_thread_returns_404(monkeypatch, isolated_settings, tmp_path):
    client, _ = _client(monkeypatch, isolated_settings, tmp_path)
    with client:
        response = client.post(
            "/chat/missing-thread/upload",
            files={"files": ("notes.txt", b"data", "text/plain")},
        )

        assert response.status_code == 404


def test_delete_thread_removes_uploaded_files(monkeypatch, isolated_settings, tmp_path):
    client, _ = _client(monkeypatch, isolated_settings, tmp_path)
    with client:
        thread_id = _start_thread(client)
        client.post(
            f"/chat/{thread_id}/upload",
            files={"files": ("notes.txt", b"hello", "text/plain")},
        )
        upload_dir = tmp_path / "files" / "chat_uploads" / thread_id
        assert upload_dir.exists()

        deleted = client.delete(f"/chat/{thread_id}")
        assert deleted.status_code == 200
        assert not upload_dir.exists()
