"""Probe: cross-session filename leak through the content-addressed cache,
plus the list_safe_directory blocklist gap.

Everything is created inside fresh temp dirs. Nothing touches the repository.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / "src" / "backend" / "tools"


def import_module(filename: str, name: str):
    tmp = Path(tempfile.mkdtemp(prefix="probe-"))
    src = (TOOLS / filename).read_text(encoding="utf-8")
    kept = [
        line.replace("from ._files", "from _files").replace("from ._file_cache", "from _file_cache")
        for line in src.splitlines(keepends=True)
        if "langchain_core" not in line
    ]
    shutil.copyfile(TOOLS / "_files.py", tmp / "_files.py")
    shutil.copyfile(TOOLS / "_file_cache.py", tmp / "_file_cache.py")
    (tmp / (name + ".py")).write_text("".join(kept), encoding="utf-8")
    sys.path.insert(0, str(tmp))
    spec = importlib.util.spec_from_file_location(name, tmp / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, tmp


files = import_module("_files.py", "probe_files")[0]
cache_mod = import_module("_file_cache.py", "probe_cache")[0]
excel_file, _ = import_module("excel_file.py", "probe_excel_file")

print("### 1. list_safe_directory does not apply _BLOCKED_NAMES/_BLOCKED_SUFFIXES")
root = Path(tempfile.mkdtemp(prefix="sweep16-leak-"))
try:
    for name, size in [
        ("credentials", b"AKIA-super-secret"),
        ("credentials.json", b'{"k":"v"}'),
        ("id_rsa", b"BEGIN RSA"),
        ("id_ed25519", b"BEGIN ED"),
        ("server.pem", b"BEGIN CERT"),
        ("id_dsa", b"BEGIN DSA"),
        ("id_ecdsa", b"BEGIN EC"),
        ("authorized_keys", b"ssh-rsa AAA"),
        ("npmrc", b"//x:_auth=abc"),
        ("git-credentials", b"https://u:p@h"),
        ("config.gpg", b"GPG"),
        ("service-account.json", b'{"type":"service_account"}'),
        ("secrets.json", b'{"s":"s"}'),
        ("tokens", b"tok_abc"),
        ("shadow", b"root:x:0:0"),
        ("notes.txt", b"hello there"),
        ("report.xlsx", b"PK\x03\x04fake"),
    ]:
        (root / name).write_bytes(size)
    for name in [".env", ".netrc", ".pgpass", ".htpasswd"]:
        (root / name).write_text("SECRET=1\n", encoding="utf-8")
    (root / "nested").mkdir()
    (root / "nested" / "id_ed25519").write_text("BEGIN", encoding="utf-8")
    (root / ".secret_dir").mkdir()
    (root / ".secret_dir" / "leak.txt").write_text("x", encoding="utf-8")

    print(files.list_safe_directory("", root=root, max_entries=500))
    print()
    print("--- list_files('nested') ---")
    print(files.list_safe_directory("nested", root=root, max_entries=500))
    print()
    print("--- list_files('.secret_dir')  [directly named dot dir] ---")
    print(files.list_safe_directory(".secret_dir", root=root, max_entries=500))
finally:
    shutil.rmtree(root, ignore_errors=True)

print()
print("### 2. cache key collision across sessions leaks another session's filename")
cache = cache_mod.ParsedFileCache()
leak_root = Path(tempfile.mkdtemp(prefix="sweep16-cache-"))
try:
    # Two XLSX files with byte-identical content but different names.
    from openpyxl import Workbook

    import io as _io

    _wb = Workbook()
    _ws = _wb.active
    _ws.title = "Sheet-Q1"
    _ws["A1"] = "confidential header"
    _ws["B1"] = 42
    _buf = _io.BytesIO()
    _wb.save(_buf)
    same_bytes = _buf.getvalue()
    a_dir = leak_root / "session-A"
    b_dir = leak_root / "session-B"
    a_dir.mkdir(parents=True)
    b_dir.mkdir(parents=True)
    f_a = a_dir / "confidential-finance-2024.xlsx"
    f_b = b_dir / "my-boring-report.xlsx"
    f_a.write_bytes(same_bytes)
    f_b.write_bytes(same_bytes)

    for label, path, root_ in [
        ("session-A reads confidential-finance-2024.xlsx", f_a, a_dir),
        ("session-B reads my-boring-report.xlsx", f_b, b_dir),
    ]:
        out = excel_file.read_excel_spreadsheet(
            path=path.name, root=root_, max_bytes=1_000_000, max_rows=10
        )
        print(f"--- {label} ---")
        print(out[:200].replace("\n", " | "))
finally:
    shutil.rmtree(leak_root, ignore_errors=True)

print()
print("### 3. ParsedFileCache.clear() call sites")
print("(grep result computed separately)")
