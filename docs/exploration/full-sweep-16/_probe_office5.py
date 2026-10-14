"""Probe: confirm the pdf_edit session-scope escape, empirically.

Everything is created inside a fresh temp dir. Nothing is read from or written
to the repository.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / "src" / "backend" / "tools"


def import_tool(filename: str, name: str, extra_stubs: dict[str, str] | None = None):
    tmp = Path(tempfile.mkdtemp(prefix="probe-"))
    src = (TOOLS / filename).read_text(encoding="utf-8")
    kept = [
        line
        for line in src.splitlines(keepends=True)
        if "langchain_core" not in line
    ]
    shutil.copyfile(TOOLS / "_files.py", tmp / "_files.py")
    for stem, body in (extra_stubs or {}).items():
        (tmp / stem).write_text(body, encoding="utf-8")
    kept = [line.replace("from ._files", "from _files") for line in kept]
    (tmp / (name + ".py")).write_text("".join(kept), encoding="utf-8")
    sys.path.insert(0, str(tmp))
    spec = importlib.util.spec_from_file_location(name, tmp / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, tmp


print("### 1. pdf_edit scope escape")
pdf_edit, _ = import_tool("pdf_edit.py", "probe_pdf_edit")

root = Path(tempfile.mkdtemp(prefix="sweep16-root-"))
try:
    # Simulate two chat sessions plus a shared 'working tree' PDF.
    (root / "chat_uploads" / "session-A").mkdir(parents=True)
    (root / "chat_uploads" / "session-B").mkdir(parents=True)
    (root / "worktree").mkdir()

    import pypdf
    from pypdf import PdfWriter

    def make_pdf(path: Path, text: str) -> None:
        w = PdfWriter()
        w.add_blank_page(width=200, height=200)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            w.write(fh)

    make_pdf(root / "worktree" / "report.pdf", "x")
    make_pdf(root / "chat_uploads" / "session-A" / "mine.pdf", "x")
    (root / "credentials").write_text("super-secret\n", encoding="utf-8")

    print("   tree before:")
    for p in sorted(root.rglob("*")):
        if p.is_file():
            print(f"     {p.relative_to(root)}")

    # The tool is built with NO session_root at all (build_pdf_edit_tools
    # signature accepts one but ignores it).
    result = pdf_edit.edit_pdf_document(
        path="worktree/report.pdf",
        operations=[pdf_edit.PdfEditOperation(action="set_metadata", title="tampered")],
        file_root=root,
        max_bytes=5_000_000,
    )
    print("   result:", result)

    # Now read the file that belongs to an unrelated session.
    result2 = pdf_edit.edit_pdf_document(
        path="chat_uploads/session-A/mine.pdf",
        operations=[pdf_edit.PdfEditOperation(action="set_metadata", author="intruder")],
        file_root=root,
        max_bytes=5_000_000,
    )
    print("   result2:", result2)

    print("   tree after:")
    for p in sorted(root.rglob("*")):
        if p.is_file():
            print(f"     {p.relative_to(root)}  ({p.stat().st_size} bytes)")

    # Verify the metadata actually landed on the source-adjacent file.
    with pypdf.PdfReader(str(root / "worktree" / "report.edited.pdf")) as r:
        print("   edited metadata:", dict(r.metadata or {}))
finally:
    shutil.rmtree(root, ignore_errors=True)
