from __future__ import annotations

from pathlib import Path

import pytest

from src.backend.tools import (
    build_go_edit_tools,
    build_groovy_edit_tools,
    build_haskell_edit_tools,
    build_julia_edit_tools,
    build_latex_edit_tools,
    build_log_edit_tools,
    build_lua_edit_tools,
    build_matlab_edit_tools,
    build_php_edit_tools,
    build_prolog_edit_tools,
    build_r_edit_tools,
    build_ruby_edit_tools,
    build_rust_edit_tools,
    build_shell_edit_tools,
    build_sql_edit_tools,
    build_swift_edit_tools,
)

# (builder, settings flag, sample filename, source line, edited line)
LANGUAGES = [
    (build_r_edit_tools, "r_edit_enabled", "plot.r", "x <- 1", "x <- 2"),
    (build_rust_edit_tools, "rust_edit_enabled", "main.rs", "fn main() {}", "fn main() { }"),
    (build_go_edit_tools, "go_edit_enabled", "main.go", "package main", "package main // app"),
    (build_sql_edit_tools, "sql_edit_enabled", "query.sql", "SELECT 1;", "SELECT 2;"),
    (build_php_edit_tools, "php_edit_enabled", "index.php", "<?php echo 1;", "<?php echo 2;"),
    (build_ruby_edit_tools, "ruby_edit_enabled", "app.rb", "puts 'hi'", "puts 'hello'"),
    # Note: the roundtrip sample avoids backslashes because the inspector
    # JSON-quotes lines (so \section would display as \\section).
    (build_latex_edit_tools, "latex_edit_enabled", "paper.tex", "% preamble", "% main"),
    (build_prolog_edit_tools, "prolog_edit_enabled", "family.pl", "parent(tom, bob).", "parent(tom, mary)."),
    (build_haskell_edit_tools, "haskell_edit_enabled", "math.hs", "factorial n = product [1 .. n]", "factorial n = product [2 .. n]"),
    (build_lua_edit_tools, "lua_edit_enabled", "script.lua", "local x = 1", "local x = 2"),
    (build_julia_edit_tools, "julia_edit_enabled", "analyze.jl", "x = [1, 2, 3]", "x = [1, 2, 3, 4]"),
    (build_shell_edit_tools, "shell_edit_enabled", "deploy.sh", "echo hello", "echo goodbye"),
    (build_matlab_edit_tools, "matlab_edit_enabled", "analysis.m", "a = 1;", "a = 2;"),
    (build_groovy_edit_tools, "groovy_edit_enabled", "script.groovy", "def x = 1", "def x = 2"),
    (build_swift_edit_tools, "swift_edit_enabled", "app.swift", "let x = 1", "let x = 2"),
    (build_log_edit_tools, "log_edit_enabled", "server.log", "INFO started", "INFO ready"),
]


def _settings(isolated_settings, tmp_path: Path, flag: str, enabled: bool):
    return isolated_settings(
        file_read_enabled=True,
        file_read_root=str(tmp_path / "files"),
        **{flag: enabled},
    )


def _session(tmp_path: Path) -> tuple[Path, Path, str]:
    session_root = tmp_path / "files" / "chat_uploads" / "thread-a"
    session_root.mkdir(parents=True, exist_ok=True)
    return tmp_path / "files", session_root, "thread-a"


@pytest.mark.parametrize("builder,flag,filename,line,replacement", LANGUAGES)
def test_registration_requires_flags_scope_and_exposes_names(
    builder, flag, filename, line, replacement, isolated_settings, tmp_path
):
    file_root, session_root, thread_id = _session(tmp_path)
    enabled = _settings(isolated_settings, tmp_path, flag, True)
    disabled = _settings(isolated_settings, tmp_path, flag, False)

    prefix = builder.__name__.removeprefix("build_").removesuffix("_edit_tools")
    expected_names = [
        f"create_{prefix}_file",
        f"inspect_{prefix}_file",
        f"edit_{prefix}_file",
    ]

    assert builder(disabled, session_root=session_root, thread_id=thread_id) == []
    assert builder(enabled, session_root=None, thread_id=thread_id) == []
    assert builder(enabled, session_root=session_root, thread_id="") == []
    assert [
        tool.name
        for tool in builder(enabled, session_root=session_root, thread_id=thread_id)
    ] == expected_names


@pytest.mark.parametrize("builder,flag,filename,line,replacement", LANGUAGES)
def test_create_inspect_edit_roundtrip_preserves_suffix(
    builder, flag, filename, line, replacement, isolated_settings, tmp_path
):
    file_root, session_root, thread_id = _session(tmp_path)
    enabled = _settings(isolated_settings, tmp_path, flag, True)
    (session_root / filename).write_bytes(f"{line}\nprint()\n".encode())

    create_tool, inspect_tool, edit_tool = builder(
        enabled, session_root=session_root, thread_id=thread_id
    )

    created = create_tool.invoke(
        {
            "type": "tool_call",
            "name": create_tool.name,
            "args": {"filename": f"new-{filename}", "content": f"{replacement}\n"},
            "id": f"call-create-{filename}",
        }
    )
    suffix = Path(filename).suffix
    assert created.artifact["filename"] == f"new-{filename}"
    assert (session_root / f"new-{filename}").read_bytes() == f"{replacement}\n".encode()

    inspected = inspect_tool.invoke({"path": filename})
    assert f"{filename}:" in inspected
    assert f'[0] "{line}"' in inspected

    edited = edit_tool.invoke(
        {
            "type": "tool_call",
            "name": edit_tool.name,
            "args": {
                "path": filename,
                "operations": [
                    {
                        "action": "replace_line",
                        "line_index": 0,
                        "expected_text": line,
                        "new_text": replacement,
                    }
                ],
            },
            "id": f"call-edit-{filename}",
        }
    )
    stem = Path(filename).stem
    assert edited.artifact["filename"] == f"{stem}.edited{suffix}"
    assert (
        session_root / f"{stem}.edited{suffix}"
    ).read_bytes() == f"{replacement}\nprint()\n".encode()
    # The upload is never modified.
    assert (session_root / filename).read_bytes() == f"{line}\nprint()\n".encode()


def test_cross_session_and_traversal_access_are_denied(
    isolated_settings, tmp_path
):
    file_root, session_root, thread_id = _session(tmp_path)
    other_root = file_root / "chat_uploads" / "thread-b"
    other_root.mkdir(parents=True)
    (other_root / "secret.sql").write_text("SELECT * FROM users;", encoding="utf-8")

    enabled = _settings(isolated_settings, tmp_path, "sql_edit_enabled", True)
    _create, _inspect, edit_tool = build_sql_edit_tools(
        enabled, session_root=session_root, thread_id=thread_id
    )
    operation = {
        "action": "append_line",
        "new_text": "DROP TABLE users;",
    }

    for path in ("../thread-b/secret.sql", str(other_root / "secret.sql")):
        result = edit_tool.invoke(
            {
                "type": "tool_call",
                "name": edit_tool.name,
                "args": {"path": path, "operations": [operation]},
                "id": "call-edit",
            }
        )
        assert "outside this chat session" in result.content
        assert "Could not edit SQL file" in result.content

    assert not (other_root / "secret.edited.sql").exists()


def test_jsonl_create_and_edit_reject_non_json_lines(isolated_settings, tmp_path):
    from src.backend.tools.jsonl_edit import build_jsonl_edit_tools

    file_root, session_root, thread_id = _session(tmp_path)
    enabled = isolated_settings(
        file_read_enabled=True,
        jsonl_edit_enabled=True,
        file_read_root=str(tmp_path / "files"),
    )
    (session_root / "events.jsonl").write_bytes(
        b'{"id": 1}\n{"id": 2}\n'
    )
    create_tool, _inspect, edit_tool = build_jsonl_edit_tools(
        enabled, session_root=session_root, thread_id=thread_id
    )

    invalid_create = create_tool.invoke(
        {
            "type": "tool_call",
            "name": "create_jsonl_file",
            "args": {"filename": "bad.jsonl", "content": '{"ok": true}\nnot json\n'},
            "id": "call-create",
        }
    )
    assert invalid_create.artifact is None
    assert "line 1 is not valid JSON" in invalid_create.content
    assert not (session_root / "bad.jsonl").exists()

    invalid_edit = edit_tool.invoke(
        {
            "type": "tool_call",
            "name": "edit_jsonl_file",
            "args": {
                "path": "events.jsonl",
                "operations": [
                    {
                        "action": "append_line",
                        "new_text": "{broken",
                    }
                ],
            },
            "id": "call-edit",
        }
    )
    assert invalid_edit.artifact is None
    assert "JSONL validation" in invalid_edit.content
    assert list(session_root.glob("*.jsonl")) == [session_root / "events.jsonl"]

    valid_edit = edit_tool.invoke(
        {
            "type": "tool_call",
            "name": "edit_jsonl_file",
            "args": {
                "path": "events.jsonl",
                "operations": [
                    {
                        "action": "replace_line",
                        "line_index": 1,
                        "expected_text": '{"id": 2}',
                        "new_text": '{"id": 3}',
                    }
                ],
            },
            "id": "call-edit-valid",
        }
    )
    assert valid_edit.artifact is not None
    assert valid_edit.artifact["filename"] == "events.edited.jsonl"
    assert (session_root / "events.edited.jsonl").read_bytes() == (
        b'{"id": 1}\n{"id": 3}\n'
    )
