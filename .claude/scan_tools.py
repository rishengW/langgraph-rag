"""Recurring scan of src/backend/tools.

Run:  python .claude/scan_tools.py
Exit code 0 = all checks clean, 1 = at least one finding.

Read-only. Does not import anything from the test suite, never writes to the
project tree.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOOLS_DIR = ROOT / "src" / "backend" / "tools"
PYS = sorted(p for p in TOOLS_DIR.glob("*.py") if p.name != "__init__.py")
INIT = TOOLS_DIR / "__init__.py"

findings: list[str] = []


def note(level: str, msg: str) -> None:
    findings.append(f"[{level}] {msg}")
    print(f"[{level}] {msg}")


# ---------------------------------------------------------------- 1. syntax
# compile() only, so the scan never writes a .pyc into the tree.
for p in PYS:
    try:
        compile(p.read_text(encoding="utf-8", errors="replace"), p.name, "exec")
    except SyntaxError as e:
        note("SYNTAX", f"{p.name}:{e.lineno} {e.msg}")

# ---------------------------------------------------------------- 2. import
pkg = importlib.import_module("src.backend.tools")

# ---------------------------------------------------------------- 3. __all__
init_src = INIT.read_text(encoding="utf-8")
for name in pkg.__all__:
    if not hasattr(pkg, name):
        note("EXPORT", f"__init__.py __all__ lists {name!r} but never imports it -> `from src.backend.tools import {name}` raises ImportError")

# ---------------------------------------------------------------- 4. orphan pyc
cache = TOOLS_DIR / "__pycache__"
if cache.is_dir():
    stems = {p.stem for p in PYS}
    stems.add("__init__")
    for c in sorted(cache.glob("*.pyc")):
        if c.name.split(".")[0] not in stems:
            note("CACHE", f"orphan bytecode {c.name} (source module deleted/renamed)")

# ------------------------------------------------- 5. exported but not wired
outside = [
    p
    for p in (ROOT / "src").rglob("*.py")
    if "backend/tools" not in p.as_posix()
]
prod_src = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in outside)

build_names = sorted({n for n in pkg.__all__ if n.startswith("build_")})
referenced = {n for n in build_names if re.search(rf"(?<!\w){re.escape(n)}\b", prod_src)}
for n in sorted(set(build_names) - referenced):
    note("UNWIRED", f"{n} exported but never referenced outside src/backend/tools")

# --------------------------------------------- 6. settings flags referenced
prov = ROOT / "src" / "backend" / "mcp" / "providers.py"
flags = set(re.findall(r"\bsettings\.(\w+_enabled)", prov.read_text(encoding="utf-8")))
Settings = importlib.import_module("src.config.settings").Settings
if hasattr(Settings, "model_fields"):
    settings_fields = set(Settings.model_fields)
else:  # dataclass
    import dataclasses

    settings_fields = {f.name for f in dataclasses.fields(Settings)}
for f in sorted(flags - settings_fields):
    note("SETTINGS", f"providers.py references settings.{f} which does not exist on Settings")

# --------------------------------------- 7. smoke-build every tool factory
from langchain_core.tools import BaseTool


def smoke(label: str, cfg: object, *, session_root: pathlib.Path | None) -> tuple[int, list[str], dict[str, str]]:
    """Build every exported factory under one settings profile.

    Returns (tools built, factories that raised, name -> factory map).
    """
    seen: dict[str, str] = {}
    empty: list[str] = []
    built = 0
    for fn_name in sorted(pkg.__all__):
        factory = getattr(pkg, fn_name, None)
        if not fn_name.startswith("build_") or not callable(factory):
            continue
        # Inspect rather than try/except-TypeError: these factories default
        # session_root=None and return [] for it instead of raising, so the
        # session surface would silently go unexercised.
        sig = inspect.signature(factory)
        takes_session = "session_root" in sig.parameters
        if takes_session and session_root is None:
            note("BUILD", f"[{label}] {fn_name}: needs session_root, none available")
            continue
        try:
            if takes_session:
                result = factory(cfg, session_root=session_root, thread_id="scan")
            else:
                result = factory(cfg)
        except Exception as e:
            note("BUILD", f"[{label}] {fn_name}: raised {type(e).__name__}: {e}")
            continue
        if isinstance(result, BaseTool):
            result = [result]
        if not isinstance(result, (list, tuple)):
            note("BUILD", f"[{label}] {fn_name}: returned {type(result).__name__}, expected BaseTool or list")
            continue
        if not result:
            empty.append(fn_name)
            continue
        for tool in result:
            if not isinstance(tool, BaseTool):
                note("BUILD", f"[{label}] {fn_name}: contained non-BaseTool {type(tool).__name__}")
                continue
            if not tool.name or not tool.description:
                note("SCHEMA", f"[{label}] {fn_name} -> missing name/description: {getattr(tool, 'name', None)!r}")
            if not tool.args:
                note("SCHEMA", f"[{label}] {fn_name} -> {tool.name!r} has an empty argument schema")
            prior = seen.get(tool.name)
            # A list factory legitimately re-emits the names of the singletons
            # it composes (build_memory_tools -> save/recall/forget). Only a
            # collision between two prod-wired factories is a real ambiguity.
            if prior and fn_name in referenced and prior in referenced:
                note("DUPENAME", f"[{label}] tool name {tool.name!r} emitted by both {prior} and {fn_name}")
            seen[tool.name] = fn_name
        built += len(result)
    if empty:
        print(f"[EMPTY:{label}] {len(empty)} factories returned no tools: {', '.join(empty)}")
    return built, empty, seen


settings = Settings(dashscope_api_key="test-key")
built, empty, names_seen = smoke("default", settings, session_root=ROOT)

# Default settings gate every editor off, so the edit surface was never
# actually exercised. Rebuild once with every flag on.
all_on = {
    f.name: (True if f.name.endswith("_enabled") else getattr(settings, f.name))
    for f in dataclasses.fields(settings)
}
built_on, empty_on, _ = smoke("all-enabled", Settings(**all_on), session_root=ROOT)
print(f"[all-enabled] {built_on} tools instantiated, {len(empty_on)} factories still empty")
if empty_on:
    for e in empty_on:
        note("UNREACHABLE", f"{e}: returns [] even with every flag enabled")

print()
print(f"factories scanned: {len(build_names)} | tools instantiated: {built} (default) / {built_on} (all-enabled)")
print(f"findings: {len(findings)}")
sys.exit(1 if findings else 0)

