"""nous is an optional extra (``forge[nous]``) — these tests pin the no-nous behavior.

The ``import_without_nous`` fixture poisons ``sys.modules`` so any ``import nous_mcp``
raises ``ModuleNotFoundError`` even when the extra IS installed, then re-imports the
modules under test fresh so their guarded imports actually run against the poisoned
machinery. Everything it touches (module cache and parent-package attributes) is restored
on teardown.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterator
from types import ModuleType

import pytest

_NOUS_MODULES = (
    "nous_mcp",
    "nous_mcp.daemon_client",
    "nous_mcp.markdown",
    "nous_mcp.storage",
    "nous_mcp.workflow",
)
# Modules whose import-time (or lazy-import) behavior depends on nous being present; they
# must be re-imported fresh under the poisoned machinery, not served from the cache.
_FRESH_ON_USE = ("forge.task_worker.nous_client",)
_MISSING = object()


@pytest.fixture
def import_without_nous() -> Iterator[Callable[[str], ModuleType]]:
    """Yield a fresh-importer that runs against a venv-without-nous simulation."""
    touched: dict[str, object] = {}

    def remember(key: str) -> None:
        touched.setdefault(key, sys.modules.get(key, _MISSING))

    def fresh(name: str) -> ModuleType:
        remember(name)
        sys.modules.pop(name, None)
        return importlib.import_module(name)

    poison = set(_NOUS_MODULES)
    poison |= {k for k in sys.modules if k == "nous_mcp" or k.startswith("nous_mcp.")}
    for key in poison:
        remember(key)
        sys.modules[key] = None  # a None entry makes ``import`` raise ModuleNotFoundError
    for key in _FRESH_ON_USE:
        remember(key)
        sys.modules.pop(key, None)

    yield fresh

    # Two passes: restore the module cache first, then re-sync parent-package attributes
    # (a fresh import rebinds e.g. ``nous_client`` on the ``forge.task_worker`` package).
    for key, value in touched.items():
        if value is _MISSING:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = value  # type: ignore[assignment]
    for key, value in touched.items():
        parent_name, _, child = key.rpartition(".")
        parent = sys.modules.get(parent_name) if parent_name else None
        if not isinstance(parent, ModuleType):
            continue
        if isinstance(value, ModuleType):
            setattr(parent, child, value)
        elif value is _MISSING and hasattr(parent, child):
            delattr(parent, child)


def test_task_store_imports_without_nous(import_without_nous):
    mod = import_without_nous("forge.shared.task_store")
    assert callable(mod.get_task_store)


def test_github_emit_path_imports_without_nous(import_without_nous):
    # The work-deployable path: the adapter and the emit types it round-trips.
    mod = import_without_nous("forge.shared.github_task_store")
    assert hasattr(mod, "GitHubTaskStore")
    emit_mod = import_without_nous("forge.shared.forge_emit")
    assert hasattr(emit_mod, "EmitSpec")


def test_github_backend_selection_works_without_nous(import_without_nous, monkeypatch):
    store_mod = import_without_nous("forge.shared.task_store")
    gh_mod = importlib.import_module("forge.shared.github_task_store")
    monkeypatch.setattr(gh_mod.settings, "repo", "acme/widgets")  # required by the adapter
    monkeypatch.setattr(store_mod.settings, "backend", "github")
    store = store_mod.get_task_store()
    assert isinstance(store, gh_mod.GitHubTaskStore)


def test_forge_backend_selection_raises_install_hint(import_without_nous, monkeypatch):
    store_mod = import_without_nous("forge.shared.task_store")
    monkeypatch.setattr(store_mod.settings, "backend", "forge")
    with pytest.raises(ModuleNotFoundError, match=r"forge\[nous\]"):
        store_mod.get_task_store()


def test_forge_store_methods_raise_install_hint_not_nameerror(import_without_nous):
    # Direct construction bypasses get_task_store()'s gate — the store's own nous
    # touchpoints must still fail with the hint, never a bare NameError/import error.
    store_mod = import_without_nous("forge.shared.task_store")
    store = store_mod.ForgeTaskStore()
    with pytest.raises(ModuleNotFoundError, match=r"forge\[nous\]"):
        store.find_task("anything")
    with pytest.raises(ModuleNotFoundError, match=r"forge\[nous\]"):
        store.list_rows("Meta")
    with pytest.raises(ModuleNotFoundError, match=r"forge\[nous\]"):
        store.in_progress_titles("pipeline:")


def test_nous_client_imports_without_nous_and_fails_on_first_use(import_without_nous):
    client = import_without_nous("forge.task_worker.nous_client")
    # Pure helpers stay usable without the extra.
    assert client._gate_reason("Ready", "Auto-OK", []) == ""
    assert "not Ready" in client._gate_reason("Spec Needed", "Auto-OK", [])
    # Anything that touches Nous raises the hint on first use, not at import.
    with pytest.raises(ModuleNotFoundError, match=r"forge\[nous\]"):
        client.find_next_task([])
    with pytest.raises(ModuleNotFoundError, match=r"forge\[nous\]"):
        client.get_task_spec("Some Task")
    with pytest.raises(ModuleNotFoundError, match=r"forge\[nous\]"):
        client.update_task_status("Some Task", "Done")
