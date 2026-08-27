"""Summarizer cache behavior against a hand-built index — no tree-sitter, no router."""

from __future__ import annotations

import json
from pathlib import Path

from forge.cartographer.config import CartographerSettings
from forge.cartographer.models import (
    FileEntry,
    MapConfig,
    ModuleNode,
    RepoIndex,
    TargetConfig,
    TargetInfo,
)
from forge.cartographer.summarize import summarize_target


class CountingModel:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.reply: str = "a summary."

    def __call__(self, system: str, user: str, max_tokens: int) -> str:
        self.calls.append(user.splitlines()[0])
        return self.reply


def _settings() -> CartographerSettings:
    return CartographerSettings(_env_file=None)


def _write_target(config: MapConfig, target: TargetConfig, shas: dict[str, str]) -> None:
    """Mirror files + a matching index.json for one target."""
    mirror = config.mirror_dir(target)
    for rel in shas:
        (mirror / rel).parent.mkdir(parents=True, exist_ok=True)
        (mirror / rel).write_text(f"content of {rel}\n")
    index = RepoIndex(
        target=TargetInfo(name=target.name, host=target.host, path=target.path),
        modules=[
            ModuleNode(name="app", path="app", files=[r for r in shas if r.startswith("app/")]),
            ModuleNode(name="lib", path="lib", files=[r for r in shas if r.startswith("lib/")]),
        ],
        files=[
            FileEntry(
                path=rel,
                module=rel.split("/", 1)[0],
                blob_sha=sha,
                language="python",
                symbols=["def x"],
            )
            for rel, sha in shas.items()
        ],
    )
    out = config.out_dir(target)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(index.model_dump_json())


_SHAS = {"app/a.py": "a" * 40, "app/b.py": "b" * 40, "lib/c.py": "c" * 40}


def _config(tmp_path: Path) -> MapConfig:
    return MapConfig(
        targets=[
            TargetConfig(name="t1", host="vm-a.test", path="/src/repo"),
            TargetConfig(name="t2", host="vm-b.test", path="/src/repo"),
        ],
        output_root=tmp_path / "maps",
    )


def test_first_run_then_full_cache_hit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_target(config, config.targets[0], _SHAS)
    sweep, synth = CountingModel(), CountingModel()

    stats = summarize_target(config, _settings(), config.targets[0], sweep, synth)
    assert stats.summarized == 3 and len(sweep.calls) == 3
    assert stats.rollups_built == 2 and stats.architecture_built
    assert len(synth.calls) == 3  # 2 rollups + 1 architecture

    sweep2, synth2 = CountingModel(), CountingModel()
    stats2 = summarize_target(config, _settings(), config.targets[0], sweep2, synth2)
    assert stats2.llm_calls == 0, "second run over unchanged input must make zero LLM calls"
    assert stats2.cache_hits == 3
    assert not sweep2.calls and not synth2.calls


def test_touch_one_file_rebuilds_exactly_its_chain(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_target(config, config.targets[0], _SHAS)
    summarize_target(config, _settings(), config.targets[0], CountingModel(), CountingModel())

    changed = dict(_SHAS, **{"app/a.py": "d" * 40})  # new blob sha for one file
    _write_target(config, config.targets[0], changed)
    sweep, synth = CountingModel(), CountingModel()
    stats = summarize_target(config, _settings(), config.targets[0], sweep, synth)
    assert len(sweep.calls) == 1  # only the touched file
    assert stats.rollups_built == 1  # only its module ("app"); "lib" rollup cached
    assert stats.architecture_built  # arch digest changed → rebuilt
    assert stats.llm_calls == 3


def test_divergent_checkout_shares_the_cache(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_target(config, config.targets[0], _SHAS)
    summarize_target(config, _settings(), config.targets[0], CountingModel(), CountingModel())

    _write_target(config, config.targets[1], _SHAS)  # same blobs, different VM
    sweep, synth = CountingModel(), CountingModel()
    stats = summarize_target(config, _settings(), config.targets[1], sweep, synth)
    assert stats.llm_calls == 0, "identical blobs on another VM must be a pure cache read"


def test_empty_completion_is_not_cached(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_target(config, config.targets[0], {"app/a.py": "a" * 40})
    sweep = CountingModel()
    sweep.reply = ""  # model produced nothing
    stats = summarize_target(config, _settings(), config.targets[0], sweep, CountingModel())
    assert stats.failed == ["app/a.py"] and stats.summarized == 0

    retry = CountingModel()
    stats2 = summarize_target(config, _settings(), config.targets[0], retry, CountingModel())
    assert len(retry.calls) == 1, "a failed file must be retried on the next run"
    assert stats2.summarized == 1


def test_call_exception_degrades_to_recorded_error(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_target(config, config.targets[0], {"app/a.py": "a" * 40})

    def boom(system: str, user: str, max_tokens: int) -> str:
        raise RuntimeError("router down")

    stats = summarize_target(config, _settings(), config.targets[0], boom, CountingModel())
    assert stats.failed == ["app/a.py"]
    assert any("router down" in e for e in stats.errors)

    retry = CountingModel()
    stats2 = summarize_target(config, _settings(), config.targets[0], retry, CountingModel())
    assert stats2.summarized == 1, "nothing was cached by the failed run"


class SizingModel(CountingModel):
    """Counting model that also records the size of every user prompt it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt_sizes: list[int] = []

    def __call__(self, system: str, user: str, max_tokens: int) -> str:
        self.prompt_sizes.append(len(user))
        return super().__call__(system, user, max_tokens)


def test_oversized_rollup_is_reduced_to_fit_the_budget(tmp_path: Path) -> None:
    budget = 600
    config = _config(tmp_path)
    shas = {f"app/f{i}.py": f"{i:02d}" * 20 for i in range(8)}
    _write_target(config, config.targets[0], shas)
    sweep = SizingModel()
    sweep.reply = "s" * 200  # 8 sections of ~210 chars — far over a 600-char budget
    synth = SizingModel()

    settings = CartographerSettings(_env_file=None, synthesis_prompt_budget_chars=budget)
    stats = summarize_target(config, settings, config.targets[0], sweep, synth)
    assert stats.rollups_built == 1 and stats.architecture_built
    assert len(sweep.calls) > 8, "condense passes must run on the sweep tier"
    assert len(synth.calls) == 2, "synthesis must get only the final rollup + architecture"
    assert all(size <= budget for size in sweep.prompt_sizes + synth.prompt_sizes), (
        "every prompt (condense, rollup, architecture) must fit the budget"
    )

    synth2 = SizingModel()
    stats2 = summarize_target(config, settings, config.targets[0], CountingModel(), synth2)
    assert stats2.llm_calls == 0 and not synth2.calls, "the reduced rollup must still cache"


def test_condense_failure_fails_the_rollup_and_is_retried(tmp_path: Path) -> None:
    config = _config(tmp_path)
    shas = {f"app/f{i}.py": f"{i:02d}" * 20 for i in range(8)}
    _write_target(config, config.targets[0], shas)

    class SummarizeThenFail(CountingModel):
        """Sweep model that summarizes files fine but fails every condense call."""

        def __call__(self, system: str, user: str, max_tokens: int) -> str:
            if "interim digest" in system:
                return ""
            return "s" * 200

    settings = CartographerSettings(_env_file=None, synthesis_prompt_budget_chars=600)
    stats = summarize_target(
        config, settings, config.targets[0], SummarizeThenFail(), CountingModel()
    )
    assert "rollup:app" in stats.failed and stats.rollups_built == 0
    assert "architecture" in stats.failed and not stats.architecture_built, (
        "an architecture doc over a missing rollup would cache permanently — must fail closed"
    )

    retry = CountingModel()
    retry.reply = "s" * 200
    stats2 = summarize_target(config, settings, config.targets[0], retry, CountingModel())
    assert stats2.rollups_built == 1, "a failed reduction must be retried on the next run"


def test_manifest_lists_cache_paths(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_target(config, config.targets[0], _SHAS)
    summarize_target(config, _settings(), config.targets[0], CountingModel(), CountingModel())
    manifest = json.loads((config.out_dir(config.targets[0]) / "summaries.json").read_text())
    assert set(manifest["files"]) == set(_SHAS.values())
    assert all(v["available"] for v in manifest["files"].values())
    assert set(manifest["modules"]) == {"app", "lib"}
    assert manifest["architecture"]["file"].startswith("cache/arch/")
