"""Deterministic supply-chain pre-scan of a unified diff (pure, no LLM).

Flags the supply-chain-relevant surface of a PR — changed dependency manifests/lockfiles, install
and build hooks, CI/CD workflows, committed binaries, and obfuscation / network-egress / secret-
access patterns in added lines — and collects the diff of just those files. The LLM audit then
focuses on this surface (which is usually small even in a huge PR, so the audit scales without
chunking). Regexes are the trigger: this catches the mechanical signals reliably, not every
possible hidden logic-level exfiltration.
"""

from __future__ import annotations

import re

from forge.pr_review_ensemble.diffsplit import file_segments
from forge.pr_review_ensemble.models import SupplyChainScan, SupplyChainSignal

# Dependency manifests (a human-declared dependency edit) vs lockfiles (machine-resolved).
_DEP_BASENAMES = {
    "package.json",
    "pyproject.toml",
    "Pipfile",
    "Cargo.toml",
    "go.mod",
    "Gemfile",
    "composer.json",
    "build.gradle",
    "build.gradle.kts",
    "pom.xml",
    "setup.cfg",
}
_LOCK_BASENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "Cargo.lock",
    "go.sum",
    "Gemfile.lock",
    "composer.lock",
}
# Files whose mere presence in a build runs code at install/build time.
_HOOK_BASENAMES = {"setup.py", "build.rs", "Makefile", "binding.gyp", "conanfile.py", "noxfile.py"}

# Per-added-line patterns. (regex, category, note). Case-insensitive.
_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r'"(?:pre|post)install"\s*:'),
        "install-hook",
        "npm pre/postinstall lifecycle script",
    ),
    (re.compile(r"\b(?:eval|exec)\s*\("), "obfuscation", "dynamic eval/exec of a string"),
    (
        re.compile(r"\b(?:child_process|os\.system|subprocess|Function\s*\()\b|pty\.spawn"),
        "obfuscation",
        "spawns a process / dynamic code",
    ),
    (
        re.compile(r"atob\(|fromCharCode|b64decode|Buffer\.from\([^,]+,\s*['\"]base64"),
        "obfuscation",
        "base64/char-code decode",
    ),
    (re.compile(r"[A-Za-z0-9+/]{180,}={0,2}"), "obfuscation", "long base64-like blob"),
    # git/file-URL deps are more specific than generic network egress, so they win first.
    (
        re.compile(r"git\+(?:https?|ssh)://|github:[\w.-]+/|file:\.?/"),
        "dependency",
        "dependency from a git/file URL, not a registry",
    ),
    (
        re.compile(
            r"\b(?:curl|wget)\b|requests\.(?:get|post)|urllib|fetch\(|net\.connect|new\s+WebSocket|https?://[^\s'\"]+"
        ),
        "network",
        "network egress",
    ),
    (
        re.compile(
            r"process\.env|os\.environ|\.npmrc|\.aws/|id_rsa|GITHUB_TOKEN|NPM_TOKEN|AWS_SECRET|SSH_KEY"
        ),
        "secret",
        "reads env vars / credentials",
    ),
    (
        re.compile(r"pull_request_target|permissions:\s*write-all|secrets\.[A-Z_]+"),
        "ci",
        "CI privilege/secret exposure",
    ),
]

_MAX_SIGNALS_PER_FILE = 6


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _is_ci(path: str) -> bool:
    base = _basename(path)
    return (
        "/.github/workflows/" in f"/{path}"
        or path.startswith(".github/workflows/")
        or base in {".gitlab-ci.yml", "azure-pipelines.yml", "Jenkinsfile"}
        or "/.circleci/" in f"/{path}"
    )


def _added_lines(file_diff: str) -> list[str]:
    return [
        ln[1:] for ln in file_diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")
    ]


def _classify_file(path: str, file_diff: str) -> SupplyChainSignal | None:
    """A whole-file signal from the filename/kind, or None if the file isn't inherently notable."""
    base = _basename(path)
    is_requirements = base.startswith("requirements") and base.endswith(".txt")
    if base in _LOCK_BASENAMES:
        return SupplyChainSignal(
            file=path,
            category="lockfile",
            evidence=base,
            note="lockfile changed — verify it matches the manifest (resolved URLs / integrity)",
        )
    if base in _DEP_BASENAMES or is_requirements:
        return SupplyChainSignal(
            file=path,
            category="dependency",
            evidence=base,
            note="dependency manifest changed — verify added/bumped packages",
        )
    if base in _HOOK_BASENAMES or base.endswith(".gemspec"):
        return SupplyChainSignal(
            file=path,
            category="install-hook",
            evidence=base,
            note="build/install hook file — runs code at install/build time",
        )
    if _is_ci(path):
        return SupplyChainSignal(
            file=path, category="ci", evidence=base, note="CI/CD workflow changed"
        )
    if "GIT binary patch" in file_diff or "\nBinary files " in f"\n{file_diff}":
        return SupplyChainSignal(
            file=path,
            category="binary",
            evidence=base,
            note="binary file added/changed — opaque to review",
        )
    return None


def _scan_lines(path: str, added: list[str]) -> list[SupplyChainSignal]:
    signals: list[SupplyChainSignal] = []
    for line in added:
        stripped = line.strip()
        for rx, category, note in _PATTERNS:
            if rx.search(line):
                signals.append(
                    SupplyChainSignal(
                        file=path, category=category, evidence=stripped[:160], note=note
                    )
                )
                break  # one signal per line is enough
    return signals


def scan_supply_chain(diff_text: str) -> SupplyChainScan:
    """Flag the supply-chain-relevant surface of a diff and collect just those files' diffs."""
    signals: list[SupplyChainSignal] = []
    relevant_files: list[str] = []
    relevant_parts: list[str] = []

    for seg in file_segments(diff_text):
        path = seg.files[0] if seg.files else "(unknown)"
        file_signals: list[SupplyChainSignal] = []

        kind = _classify_file(path, seg.text)
        if kind is not None:
            file_signals.append(kind)

        # Lockfiles legitimately carry long base64 integrity hashes; line-scanning them only floods
        # noise, so the whole-file "lockfile" signal is enough.
        if kind is None or kind.category != "lockfile":
            file_signals.extend(_scan_lines(path, _added_lines(seg.text)))

        if file_signals:
            if len(file_signals) > _MAX_SIGNALS_PER_FILE:
                extra = len(file_signals) - _MAX_SIGNALS_PER_FILE
                file_signals = file_signals[:_MAX_SIGNALS_PER_FILE]
                file_signals.append(
                    SupplyChainSignal(
                        file=path,
                        category=file_signals[-1].category,
                        evidence=f"(+{extra} more matches in this file)",
                        note="",
                    )
                )
            signals.extend(file_signals)
            relevant_files.append(path)
            relevant_parts.append(seg.text)

    return SupplyChainScan(
        signals=signals,
        relevant_files=relevant_files,
        relevant_diff="".join(relevant_parts),
        full_diff_lines=diff_text.count("\n") + 1,
    )
