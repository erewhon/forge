"""Fanout — entry point. `forge fanout sweep …` runs the seats; `forge fanout judge DIR` opens
an interactive Claude session on the findings."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shlex
import string
import sys
from datetime import datetime
from pathlib import Path

from forge.fanout.config import settings
from forge.fanout.models import SeatRun, SeatSpec, SweepManifest
from forge.fanout.profiles import BUILTIN, Profile, builtin_names, load_profile
from forge.fanout.prompts import build_judge_prompt, wrap_sweep_prompt
from forge.fanout.runner import run_all

_MAX_SEATS = len(string.ascii_uppercase)


def parse_one_seat(raw: str, label: str) -> SeatSpec:
    """Parse a "[kind:]model" token. Bare or "claude:<id>" -> claude; "pi:<id>" -> pi.

    pi model ids are passed through untouched (some contain "/", e.g. "go/glm-5.3"); the provider
    goes on the command line separately.
    """
    kind, sep, rest = raw.partition(":")
    if sep and kind in ("claude", "pi"):
        model = rest.strip()
    else:
        kind, model = "claude", raw.strip()
    if not model:
        raise SystemExit(f"error: empty model in seat spec {raw!r}")
    display = f"pi:{settings.pi_provider}/{model}" if kind == "pi" else f"claude:{model}"
    return SeatSpec(label=label, kind=kind, model=model, display=display)  # type: ignore[arg-type]


def parse_seats(raw: str | None, defaults: list[str] | None = None) -> list[SeatSpec]:
    if raw is None:
        tokens = list(defaults) if defaults is not None else list(settings.default_models)
    else:
        tokens = raw.split(",")
    tokens = [t.strip() for t in tokens if t.strip()]
    if not 1 <= len(tokens) <= _MAX_SEATS:
        raise SystemExit(f"error: --models must list 1 to {_MAX_SEATS} seats, got {len(tokens)}")
    labels = string.ascii_uppercase[: len(tokens)]
    return [parse_one_seat(tok, lab) for lab, tok in zip(labels, tokens, strict=True)]


def default_out_dir(target: Path, now: datetime) -> Path:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", target.name) or "sweep"
    return settings.out_base / f"{name}-{now.strftime('%Y%m%d-%H%M%S')}"


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("error: provide --prompt TEXT, --prompt-file PATH, or pipe a prompt via stdin")


def write_manifest(manifest: SweepManifest) -> Path:
    path = manifest.out_dir / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_manifest(out_dir: Path) -> SweepManifest:
    path = out_dir / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"error: no manifest.json in {out_dir} (not a fanout sweep dir?)")
    return SweepManifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_judge_prompt(manifest: SweepManifest) -> Path:
    path = manifest.out_dir / "judge-prompt.md"
    path.write_text(build_judge_prompt(manifest), encoding="utf-8")
    return path


def judge_cmd(model: str, prompt: str) -> list[str]:
    return [settings.claude_binary, "--model", model, prompt]


def _split_tools(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [t.strip() for t in raw.split(",") if t.strip()]


def resolve_profile(args: argparse.Namespace) -> Profile:
    """The sweep's profile: built-in or file, with --claude-tools / --pi-tools overrides."""
    profile = load_profile(args.profile)
    overrides: dict[str, list[str]] = {}
    if (ct := _split_tools(getattr(args, "claude_tools", None))) is not None:
        overrides["claude_allowed_tools"] = ct
    if (pt := _split_tools(getattr(args, "pi_tools", None))) is not None:
        overrides["pi_tools"] = pt
    return profile.model_copy(update=overrides) if overrides else profile


async def _sweep(args: argparse.Namespace) -> int:
    user_prompt = _read_prompt(args)
    if not user_prompt.strip():
        raise SystemExit("error: prompt is empty")
    target = Path(args.target).resolve()
    if not target.is_dir():
        raise SystemExit(f"error: {target} is not a directory")
    profile = resolve_profile(args)
    seats = parse_seats(args.models, profile.default_models)
    now = datetime.now().astimezone()  # local time: the dir name is for a human
    if args.timeout is not None and args.timeout <= 0:
        raise SystemExit("error: --timeout must be a positive number of seconds")
    out_dir = Path(args.out).resolve() if args.out else default_out_dir(target, now)
    if out_dir.exists() and not out_dir.is_dir():
        raise SystemExit(f"error: {out_dir} exists and is not a directory")
    if out_dir.is_dir() and any(out_dir.iterdir()):
        raise SystemExit(f"error: {out_dir} exists and is not empty; pick a fresh --out")
    out_dir.mkdir(parents=True, exist_ok=True)

    user_prompt_path = out_dir / "question.md"
    user_prompt_path.write_text(user_prompt, encoding="utf-8")
    prompt = wrap_sweep_prompt(user_prompt, target, profile)
    prompt_path = out_dir / "prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    roster = ", ".join(f"{s.label}={s.display}" for s in seats)
    print(
        f"Fanout sweep [{profile.name}] @ {target} → {out_dir}\n  seats: {roster}",
        file=sys.stderr,
    )

    # The manifest is checkpointed after every seat (and in the finally on Ctrl-C), so a sweep
    # that is interrupted or crashes mid-way still leaves a judgeable directory behind.
    done: dict[str, SeatRun] = {}

    def manifest_now() -> SweepManifest:
        seats_now = [done.get(s.label) or _pending(s) for s in seats]
        return SweepManifest(
            target=target,
            out_dir=out_dir,
            profile=profile,
            timestamp=now,
            user_prompt_path=user_prompt_path,
            prompt_path=prompt_path,
            seats=seats_now,
        )

    def checkpoint(run: SeatRun) -> None:
        done[run.label] = run
        _print_seat(run)
        write_manifest(manifest_now())

    try:
        await run_all(
            specs=seats,
            prompt=prompt,
            repo=target,
            findings_dir=out_dir / "findings",
            timeout=args.timeout,
            on_done=checkpoint,
            profile=profile,
        )
    finally:
        manifest = manifest_now()
        write_manifest(manifest)
        judge_path = write_judge_prompt(manifest)
        print(f"Manifest: {out_dir / 'manifest.json'}\nJudge prompt: {judge_path}", file=sys.stderr)

    with_findings = [r for r in manifest.seats if r.findings_path is not None]
    clean = [r for r in manifest.seats if r.status == "ok"]
    if not with_findings:
        print("No seat produced findings; not judging.", file=sys.stderr)
        return 1
    if not clean:
        print("No seat finished cleanly; only partial findings are on disk.", file=sys.stderr)
    if args.judge and clean:
        _exec_judge(manifest, args.judge_model, print_only=False)
    else:
        print(f"Next: forge fanout judge {shlex.quote(str(out_dir))}", file=sys.stderr)
    return 0 if clean else 1


def _pending(spec: SeatSpec) -> SeatRun:
    return SeatRun(
        label=spec.label,
        kind=spec.kind,
        model=spec.model,
        display=spec.display,
        status="pending",
        cmd=[],
        error_message="seat did not finish (sweep interrupted or crashed)",
    )


def _print_seat(run: SeatRun) -> None:
    line = f"  {run.label} ({run.display}): {run.status}"
    if run.latency_ms is not None:
        line += f" — {run.latency_ms / 1000:.0f}s"
    if run.findings_path is not None:
        line += f" — {run.findings_chars} chars → {run.findings_path.name}"
    if run.error_message:
        line += f" — {run.error_message}"
    print(line, file=sys.stderr)
    if run.status != "ok" and run.stderr_tail.strip():
        for t in run.stderr_tail.strip().splitlines()[-3:]:
            print(f"      {t}", file=sys.stderr)


def _exec_judge(manifest: SweepManifest, model: str, *, print_only: bool) -> int:
    prompt = build_judge_prompt(manifest)
    cmd = judge_cmd(model, prompt)
    if print_only:
        print(f"cd {shlex.quote(str(manifest.target))} && {shlex.join(cmd)}")
        return 0
    print(f"Opening judge session ({model}) in {manifest.target}", file=sys.stderr)
    os.chdir(manifest.target)
    os.execvp(cmd[0], cmd)  # replaces this process; the session is the user's from here
    return 0  # unreachable


def _judge(args: argparse.Namespace) -> int:
    manifest = load_manifest(Path(args.dir).resolve())
    if not manifest.target.is_dir():
        raise SystemExit(f"error: target {manifest.target} from the manifest no longer exists")
    write_judge_prompt(manifest)
    return _exec_judge(manifest, args.model, print_only=args.print_only)


def _profiles(args: argparse.Namespace) -> int:
    for name in builtin_names():
        prof = BUILTIN[name]()  # type: ignore[operator]
        print(f"{name:8s} {prof.description}")
        print(f"         seats: {', '.join(prof.default_models)}")
        print(f"         claude tools: {' '.join(prof.claude_allowed_tools)}")
        print(f"         pi tools: {','.join(prof.pi_tools)}")
    print("\nA TOML file path also works: it overrides fields of its `base` (default code).")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge fanout",
        description=(
            "One analysis prompt → N agentic seats in parallel (claude -p / pi -p, read-only) → "
            "findings files → an interactive Claude judge that verifies and collates."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sweep = sub.add_parser("sweep", help="Run the seats and write findings + manifest to DIR")
    pg = sweep.add_mutually_exclusive_group()
    pg.add_argument("--prompt", help="The question, inline")
    pg.add_argument("--prompt-file", help="The question, from a file (stdin if neither)")
    sweep.add_argument(
        "--dir",
        "--repo",
        dest="target",
        default=".",
        help="Directory the seats work in: a repo, a folder of papers, ... (default: cwd)",
    )
    sweep.add_argument(
        "--profile",
        default="code",
        help="Kind of sweep: " + ", ".join(builtin_names()) + ", or a TOML profile file "
        "(see `forge fanout profiles`). Default: code",
    )
    sweep.add_argument(
        "--models",
        default=None,
        help="Comma-separated seats as '[kind:]model'. Bare or 'claude:<id>' runs claude -p; "
        "'pi:<id>' runs pi -p over the router (local models only). Default: the profile's.",
    )
    sweep.add_argument(
        "--claude-tools",
        default=None,
        help="Override the profile's claude --allowedTools (comma-separated)",
    )
    sweep.add_argument(
        "--pi-tools",
        default=None,
        help="Override the profile's pi --tools (comma-separated; e.g. add bash for web access)",
    )
    sweep.add_argument(
        "--out", default=None, help="Sweep directory (default: ~/fanout/<dir-name>-<ts>)"
    )
    sweep.add_argument(
        "--timeout", type=float, default=None, help="Per-seat timeout in seconds (default 1800)"
    )
    sweep.add_argument(
        "--judge",
        action="store_true",
        help="Open the interactive judge session when the sweep ends",
    )
    sweep.add_argument(
        "--judge-model", default=settings.judge_model, help="Claude model for the judge session"
    )
    sweep.set_defaults(func=lambda a: asyncio.run(_sweep(a)))

    judge = sub.add_parser("judge", help="Open an interactive Claude judge over a sweep DIR")
    judge.add_argument("dir", help="A sweep directory holding manifest.json")
    judge.add_argument("--model", default=settings.judge_model, help="Claude model for the judge")
    judge.add_argument(
        "--print-only",
        action="store_true",
        help="Write judge-prompt.md and print the command instead of exec'ing claude",
    )
    judge.set_defaults(func=_judge)

    profiles = sub.add_parser("profiles", help="List the built-in sweep profiles")
    profiles.set_defaults(func=_profiles)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
