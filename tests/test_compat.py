from __future__ import annotations

import importlib
import warnings


def _reload_with_warnings(module_name: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module(module_name)
        module = importlib.reload(module)
    return module, caught


def test_core_compat_imports_warn_and_remain_functional():
    module, caught = _reload_with_warnings("src.backend.core.nodes")

    assert hasattr(module, "agent_factory")
    assert hasattr(module, "_question_tokens")
    assert any("src.backend.core.nodes" in str(item.message) for item in caught)


def test_chat_compat_imports_warn_and_remain_functional():
    module, caught = _reload_with_warnings("src.frontend.chat.sessions")

    assert hasattr(module, "ChatSessionRegistry")
    assert any("src.frontend.chat.sessions" in str(item.message) for item in caught)
