"""Shared fixture: a mini workspace repo (two npm workspace packages + a python tool)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from forge.cartographer.models import MapConfig, TargetConfig


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
            "HOME": str(repo),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        },
    )


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    """A committed fixture repo: npm workspace with two packages, plus one python entry script.

    ``scratch.py`` stays untracked so the sha256 fallback path is exercised.
    """
    repo = tmp_path / "mini"
    repo.mkdir()
    (repo / "package.json").write_text(json.dumps({"name": "mini", "workspaces": ["packages/*"]}))
    app = repo / "packages" / "app"
    util = repo / "packages" / "util"
    for pkg, name in ((app, "app"), (util, "util")):
        pkg.mkdir(parents=True)
        (pkg / "package.json").write_text(json.dumps({"name": name}))
    (app / "index.js").write_text(
        'import { helper } from "util";\n'
        "export function start() { return helper(); }\n"
        "export class Server { boot() {} }\n"
    )
    (util / "util.js").write_text("export function helper() { return 1; }\n")
    (repo / "tool.py").write_text(
        "import json\n\n"
        "class Tool:\n    def run(self):\n        return json.dumps({})\n\n"
        'if __name__ == "__main__":\n    Tool().run()\n'
    )
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fixture")
    (repo / "scratch.py").write_text("def scratch():\n    pass\n")  # after the commit, deliberately
    return repo


@pytest.fixture
def config(tmp_path: Path, mini_repo: Path) -> MapConfig:
    return MapConfig(
        targets=[
            TargetConfig(name="mini-a", host="vm-a.test", path=str(mini_repo)),
            TargetConfig(name="mini-b", host="vm-b.test", path=str(mini_repo)),
        ],
        output_root=tmp_path / "maps",
    )
