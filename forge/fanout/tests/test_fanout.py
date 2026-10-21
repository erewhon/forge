"""Unit tests for fanout: seat parsing, command construction, output cleaning, the sweep runner
against fake seat binaries, manifest round-trip, and the judge prompt / command."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from forge.fanout import main as fanout_main
from forge.fanout.config import settings
from forge.fanout.models import SeatRun, SeatSpec, SweepManifest
from forge.fanout.profiles import builtin_names, load_profile
from forge.fanout.prompts import build_judge_prompt, wrap_sweep_prompt
from forge.fanout.runner import (
    build_cmd,
    clean_output,
    findings_filename,
    run_all,
    run_seat,
    seat_env,
)

# ------------------------------------------------------------------------------ seat parsing


def test_bare_token_is_claude():
    spec = fanout_main.parse_one_seat("opus", "A")
    assert spec == SeatSpec(label="A", kind="claude", model="opus", display="claude:opus")


def test_explicit_claude_prefix():
    spec = fanout_main.parse_one_seat("claude:claude-opus-4-8", "B")
    assert (spec.kind, spec.model) == ("claude", "claude-opus-4-8")


def test_pi_token_keeps_model_id_and_shows_provider():
    spec = fanout_main.parse_one_seat("pi:qwen3.6-plus", "A")
    assert spec.kind == "pi"
    assert spec.model == "qwen3.6-plus"
    assert spec.display == "pi:llm-router/qwen3.6-plus"


def test_pi_model_with_slash_is_not_mangled():
    spec = fanout_main.parse_one_seat("pi:go/glm-5.3", "A")
    assert spec.model == "go/glm-5.3"


def test_parse_seats_labels_and_single_seat_allowed():
    specs = fanout_main.parse_seats("claude:opus,pi:deepseek-v4-flash")
    assert [s.label for s in specs] == ["A", "B"]
    assert len(fanout_main.parse_seats("pi:glm-5.3")) == 1


def test_parse_seats_rejects_empty():
    with pytest.raises(SystemExit):
        fanout_main.parse_seats(" , ")


def test_default_seats_when_models_omitted():
    specs = fanout_main.parse_seats(None)
    assert [s.kind for s in specs] == ["claude", "pi"]
    assert specs[1].model == "gemma4-26b"


# ------------------------------------------------------------------------- command building


def test_claude_cmd_is_read_only_print_mode():
    spec = fanout_main.parse_one_seat("claude:opus", "A")
    cmd = build_cmd(spec, "Q")
    assert cmd[:3] == [settings.claude_binary, "-p", "Q"]
    assert cmd[cmd.index("--model") + 1] == "opus"
    i = cmd.index("--allowedTools")
    assert cmd[i + 1 : i + 1 + len(settings.claude_allowed_tools)] == settings.claude_allowed_tools
    j = cmd.index("--disallowedTools")
    assert "Edit" in cmd[j + 1 :] and "Write" in cmd[j + 1 :]
    assert cmd[-2:] == ["--output-format", "text"]


def test_pi_cmd_uses_provider_flag_and_read_only_tools():
    spec = fanout_main.parse_one_seat("pi:qwen3.6-plus", "B")
    cmd = build_cmd(spec, "- leading dash prompt")
    assert cmd[0] == settings.pi_binary
    assert "-p" in cmd and "--no-session" in cmd and "--no-skills" in cmd
    assert cmd[cmd.index("--provider") + 1] == "llm-router"
    assert cmd[cmd.index("--model") + 1] == "qwen3.6-plus"
    assert cmd[cmd.index("--tools") + 1] == "read,grep,find,ls"
    # the prompt is after `--` so a prompt beginning with a dash is not parsed as an option
    assert cmd[-2:] == ["--", "- leading dash prompt"]


def test_findings_filename_is_label_and_safe_slug():
    spec = fanout_main.parse_one_seat("pi:go/glm-5.3", "C")
    assert findings_filename(spec) == "C-pi-llm-router-go-glm-5.3.md"


def test_clean_output_strips_pi_stats_line():
    raw = "⏱ turn 0 · ended 09:16:38 · 1.4s\n# Findings\n\n## F1: x\n"
    assert clean_output("pi", raw) == "# Findings\n\n## F1: x\n"
    assert clean_output("claude", "   \n") == ""


def test_wrap_sweep_prompt_embeds_question_and_schema(tmp_path):
    text = wrap_sweep_prompt("  Is the cache thread-safe?  ", tmp_path)
    assert "Is the cache thread-safe?" in text
    assert str(tmp_path) in text
    assert "# Findings" in text and "# Not found" in text and "**Confidence:**" in text


# ------------------------------------------------------------------------------ sweep runner


def _fake_bin(dir: Path, name: str, script: str) -> Path:
    path = dir / name
    path.write_text("#!/bin/sh\n" + script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_run_all_writes_findings_and_classifies_statuses(tmp_path, monkeypatch):
    bins = tmp_path / "bin"
    bins.mkdir()
    # fake claude: echoes a findings doc; fake pi: prints the stats line then nothing useful
    _fake_bin(bins, "fake-claude", 'echo "# Findings"; echo "## F1: from $PWD"\n')
    _fake_bin(bins, "fake-pi", 'echo "⏱ turn 0 · 1ms"; exit 0\n')
    _fake_bin(bins, "fake-crash", 'echo "partial"; echo "boom" >&2; exit 3\n')
    monkeypatch.setattr(settings, "claude_binary", str(bins / "fake-claude"))
    monkeypatch.setattr(settings, "pi_binary", str(bins / "fake-pi"))

    repo = tmp_path / "repo"
    repo.mkdir()
    specs = fanout_main.parse_seats("claude:opus,pi:glm-5.3")
    crash = SeatSpec(label="C", kind="claude", model="x", display="claude:x")
    monkeypatch.setattr(
        "forge.fanout.runner.build_cmd",
        lambda spec, prompt, profile=None: (
            [str(bins / "fake-crash")] if spec.label == "C" else build_cmd(spec, prompt, profile)
        ),
    )
    runs = asyncio.run(
        run_all(
            specs=[*specs, crash],
            prompt="Q",
            repo=repo,
            findings_dir=tmp_path / "out" / "findings",
            timeout=30,
        )
    )
    by = {r.label: r for r in runs}
    assert by["A"].status == "ok"
    assert by["A"].findings_path is not None
    assert f"from {repo}" in by["A"].findings_path.read_text()  # ran with cwd=repo
    assert by["B"].status == "empty" and by["B"].findings_path is None
    assert by["C"].status == "error" and by["C"].returncode == 3
    assert by["C"].findings_path is not None  # partial stdout kept
    assert "boom" in by["C"].stderr_tail


def test_run_all_reports_missing_binary_as_error(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "pi_binary", str(tmp_path / "nope"))
    specs = fanout_main.parse_seats("pi:glm-5.3")
    runs = asyncio.run(
        run_all(specs=specs, prompt="Q", repo=tmp_path, findings_dir=tmp_path / "f", timeout=5)
    )
    assert runs[0].status == "error"
    assert "could not start" in (runs[0].error_message or "")


def test_timeout_keeps_partial_output_and_kills_the_process_group(tmp_path, monkeypatch):
    bins = tmp_path / "bin"
    bins.mkdir()
    # Prints a finding, then a grandchild inherits the pipes and sleeps: a plain kill of the
    # direct child would leave the pipe open and the drain hanging; the pgroup kill must not.
    _fake_bin(bins, "slow", 'echo "# Findings"; echo "## F1: early"; sleep 60 & wait\n')
    monkeypatch.setattr(settings, "claude_binary", str(bins / "slow"))
    specs = fanout_main.parse_seats("claude:opus")
    t0 = time.monotonic()
    runs = asyncio.run(
        run_all(specs=specs, prompt="Q", repo=tmp_path, findings_dir=tmp_path / "f", timeout=1)
    )
    assert time.monotonic() - t0 < 20
    assert runs[0].status == "timeout"
    assert runs[0].findings_path is not None
    assert "F1: early" in runs[0].findings_path.read_text()


def test_one_seat_crashing_does_not_lose_the_others(tmp_path, monkeypatch):
    bins = tmp_path / "bin"
    bins.mkdir()
    _fake_bin(bins, "fake-claude", 'echo "# Findings"\n')
    monkeypatch.setattr(settings, "claude_binary", str(bins / "fake-claude"))
    specs = fanout_main.parse_seats("claude:opus,pi:x")
    real = build_cmd

    def exploding(spec, prompt, profile=None):
        if spec.kind == "pi":
            raise RuntimeError("kaboom")
        return real(spec, prompt, profile)

    monkeypatch.setattr("forge.fanout.runner.build_cmd", exploding)
    runs = asyncio.run(
        run_all(specs=specs, prompt="Q", repo=tmp_path, findings_dir=tmp_path / "f", timeout=10)
    )
    assert runs[0].status == "ok"
    assert runs[1].status == "error" and "kaboom" in (runs[1].error_message or "")


# ---------------------------------------------------------------- manifest + judge prompt


def _manifest(tmp_path: Path) -> SweepManifest:
    out = tmp_path / "sweep"
    (out / "findings").mkdir(parents=True)
    fa = out / "findings" / "A-claude-opus.md"
    fa.write_text("# Findings\n")
    seats = [
        SeatRun(
            label="A",
            kind="claude",
            model="opus",
            display="claude:opus",
            status="ok",
            cmd=["claude"],
            findings_path=fa,
            findings_chars=11,
        ),
        SeatRun(
            label="B",
            kind="pi",
            model="glm-5.3",
            display="pi:llm-router/glm-5.3",
            status="error",
            cmd=["pi"],
            error_message="pi exited with code 1",
        ),
    ]
    return SweepManifest(
        target=tmp_path / "repo",
        out_dir=out,
        profile=load_profile("code"),
        timestamp=datetime(2026, 10, 20, 12, 0, tzinfo=UTC),
        user_prompt_path=out / "question.md",
        prompt_path=out / "prompt.md",
        seats=seats,
    )


def test_manifest_round_trips_through_json(tmp_path):
    m = _manifest(tmp_path)
    path = fanout_main.write_manifest(m)
    assert json.loads(path.read_text())["seats"][1]["status"] == "error"
    assert fanout_main.load_manifest(m.out_dir) == m


def test_judge_prompt_lists_findings_and_failed_seats(tmp_path):
    m = _manifest(tmp_path)
    text = build_judge_prompt(m)
    assert str(m.seats[0].findings_path) in text
    assert "B (pi:llm-router/glm-5.3): error — pi exited with code 1" in text
    assert str(m.out_dir / "collated.md") in text
    assert "CONFIRMED, FALSE, UNVERIFIABLE, or DUPLICATE-of" in text


def test_judge_print_only_writes_prompt_and_prints_command(tmp_path, capsys):
    m = _manifest(tmp_path)
    m.target.mkdir()
    fanout_main.write_manifest(m)
    rc = fanout_main.main(["judge", str(m.out_dir), "--model", "fable", "--print-only"])
    assert rc == 0
    assert (m.out_dir / "judge-prompt.md").is_file()
    out = capsys.readouterr().out
    assert out.startswith(f"cd {m.target} && ")
    assert "--model fable" in out


def test_judge_cmd_is_interactive_claude_with_positional_prompt():
    assert fanout_main.judge_cmd("opus", "P") == [settings.claude_binary, "--model", "opus", "P"]


def test_sweep_refuses_nonempty_out_dir(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        fanout_main.main(
            [
                "sweep",
                "--prompt",
                "Q",
                "--repo",
                str(tmp_path),
                "--out",
                str(out),
                "--models",
                "pi:x",
            ]
        )


def test_default_out_dir_uses_repo_name_and_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "out_base", tmp_path)
    d = fanout_main.default_out_dir(Path("/srv/my repo"), datetime(2026, 10, 20, 9, 5, 7))
    assert d == tmp_path / "my-repo-20261020-090507"


def test_sweep_end_to_end_with_fake_seats(tmp_path, monkeypatch, capsys):
    bins = tmp_path / "bin"
    bins.mkdir()
    _fake_bin(bins, "fake-claude", 'echo "# Findings"; echo "## F1: ok"\n')
    _fake_bin(bins, "fake-pi", 'echo "# Findings"; echo "## F1: also ok"\n')
    monkeypatch.setattr(settings, "claude_binary", str(bins / "fake-claude"))
    monkeypatch.setattr(settings, "pi_binary", str(bins / "fake-pi"))
    repo = tmp_path / "repo"
    repo.mkdir()
    out = tmp_path / "sweep"
    rc = fanout_main.main(
        [
            "sweep",
            "--prompt",
            "Q?",
            "--repo",
            str(repo),
            "--out",
            str(out),
            "--models",
            "claude:opus,pi:glm-5.3",
            "--timeout",
            "30",
        ]
    )
    assert rc == 0
    assert (out / "question.md").read_text() == "Q?"
    assert "Q?" in (out / "prompt.md").read_text()
    assert sorted(p.name for p in (out / "findings").iterdir()) == [
        "A-claude-opus.md",
        "B-pi-llm-router-glm-5.3.md",
    ]
    m = fanout_main.load_manifest(out)
    assert [s.status for s in m.seats] == ["ok", "ok"]
    assert (out / "judge-prompt.md").is_file()
    assert f"forge fanout judge {out}" in capsys.readouterr().err
    assert os.environ.get("PWD") != str(repo)  # the runner's PWD override did not leak


def test_cancellation_saves_partial_findings_and_reports_the_seat(tmp_path, monkeypatch):
    bins = tmp_path / "bin"
    bins.mkdir()
    _fake_bin(bins, "slow", 'echo "# Findings"; echo "## F1: before ctrl-c"; sleep 60 & wait\n')
    monkeypatch.setattr(settings, "claude_binary", str(bins / "slow"))
    spec = fanout_main.parse_seats("claude:opus")[0]
    seen: list[SeatRun] = []

    async def scenario():
        task = asyncio.create_task(
            run_seat(
                spec=spec,
                prompt="Q",
                repo=tmp_path,
                findings_dir=tmp_path / "f",
                timeout=60,
                on_done=seen.append,
            )
        )
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert len(seen) == 1 and seen[0].status == "cancelled"
    assert seen[0].findings_path is not None
    assert "before ctrl-c" in seen[0].findings_path.read_text()


def test_sweep_exit_code_requires_a_clean_seat_but_keeps_partials(tmp_path, monkeypatch, capsys):
    bins = tmp_path / "bin"
    bins.mkdir()
    _fake_bin(bins, "crashy", 'echo "# Findings"; echo "## F1: then died"; exit 2\n')
    monkeypatch.setattr(settings, "claude_binary", str(bins / "crashy"))
    repo = tmp_path / "repo"
    repo.mkdir()
    out = tmp_path / "sweep"
    rc = fanout_main.main(
        ["sweep", "--prompt", "Q", "--repo", str(repo), "--out", str(out), "--models", "claude:x"]
    )
    assert rc == 1
    m = fanout_main.load_manifest(out)
    assert m.seats[0].status == "error" and m.seats[0].findings_path is not None
    assert (out / "judge-prompt.md").is_file()
    assert "No seat finished cleanly" in capsys.readouterr().err


def test_sweep_validates_timeout_and_out_path(tmp_path):
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(SystemExit, match="not a directory"):
        fanout_main.main(["sweep", "--prompt", "Q", "--repo", str(tmp_path), "--out", str(f)])
    with pytest.raises(SystemExit, match="positive"):
        fanout_main.main(
            [
                "sweep",
                "--prompt",
                "Q",
                "--repo",
                str(tmp_path),
                "--out",
                str(tmp_path / "o"),
                "--timeout",
                "0",
            ]
        )


def test_pi_seat_env_lifts_the_guardrails_tool_budget(tmp_path):
    pi = fanout_main.parse_one_seat("pi:coder", "A")
    env = seat_env(pi, tmp_path)
    assert env["PI_GUARD_BUDGET"] == str(settings.pi_tool_budget)
    assert env["PWD"] == str(tmp_path)
    assert "PI_GUARD_BUDGET" not in seat_env(fanout_main.parse_one_seat("opus", "B"), tmp_path)


# ----------------------------------------------------------------------------------- profiles


def test_builtin_profiles_resolve_and_differ_in_tools_and_wording():
    assert builtin_names() == ["code", "papers", "refs", "raw"]
    code, papers, refs, raw = (load_profile(n) for n in builtin_names())
    assert raw.sweep_template == "{prompt}" and raw.system_prompt is None
    assert code.claude_allowed_tools == settings.claude_allowed_tools
    assert "WebFetch" in refs.claude_allowed_tools and "WebSearch" in refs.claude_allowed_tools
    assert "WebFetch" not in papers.claude_allowed_tools
    assert "**Where:**" in code.sweep_template and "**Source:**" in papers.sweep_template
    assert "**Resolves:**" in refs.sweep_template
    assert "MISREAD" in papers.judge_template and "SUSPICIOUS" in refs.judge_template
    assert papers.default_models[0] == "claude:fable"


def test_profile_sets_seat_tools_in_build_cmd():
    refs = load_profile("refs")
    cmd = build_cmd(fanout_main.parse_one_seat("claude:fable", "A"), "Q", refs)
    i = cmd.index("--allowedTools")
    assert "WebFetch" in cmd[i + 1 : cmd.index("--disallowedTools")]
    pi_cmd = build_cmd(fanout_main.parse_one_seat("pi:gemma4-26b", "B"), "Q", refs)
    assert pi_cmd[pi_cmd.index("--tools") + 1] == "read,grep,find,ls"


def test_wrap_sweep_prompt_uses_the_profile_template(tmp_path):
    text = wrap_sweep_prompt("What do the papers say about X?", tmp_path, load_profile("papers"))
    assert "documents under " + str(tmp_path) in text
    assert "**Quote:**" in text and "path:line" not in text


def test_judge_prompt_uses_the_manifest_profile(tmp_path):
    m = _manifest(tmp_path).model_copy(update={"profile": load_profile("papers")})
    text = build_judge_prompt(m)
    assert "reading sweep" in text and "MISREAD" in text
    assert str(m.seats[0].findings_path) in text


def test_unknown_profile_is_an_error():
    with pytest.raises(SystemExit, match="unknown profile"):
        load_profile("nope")


def test_toml_profile_overrides_its_base(tmp_path):
    f = tmp_path / "cites.toml"
    f.write_text(
        'base = "refs"\n'
        'description = "my cites"\n'
        'pi_tools = ["read", "bash"]\n'
        'sweep_template = "Target {target}. Q: {prompt}. Literal {{braces}} ok."\n'
    )
    prof = load_profile(str(f))
    assert prof.name == "cites"  # from the file stem
    assert prof.description == "my cites"
    assert prof.pi_tools == ["read", "bash"]
    assert "WebFetch" in prof.claude_allowed_tools  # inherited from refs
    assert "SUSPICIOUS" in prof.judge_template  # inherited
    assert (
        wrap_sweep_prompt("why", tmp_path, prof)
        == f"Target {tmp_path}. Q: why. Literal {{braces}} ok."
    )


def test_toml_profile_rejects_bad_base_and_bad_toml(tmp_path):
    f = tmp_path / "p.toml"
    f.write_text('base = "nope"\n')
    with pytest.raises(SystemExit, match="base must be one of"):
        load_profile(str(f))
    f.write_text("this is = not toml = at all\n")
    with pytest.raises(SystemExit, match="not valid TOML"):
        load_profile(str(f))


def test_sweep_records_profile_and_honours_tool_overrides(tmp_path, monkeypatch, capsys):
    bins = tmp_path / "bin"
    bins.mkdir()
    # the fake seat echoes its argv so the test can see which tools it was given
    _fake_bin(bins, "fake-claude", 'echo "# Findings"; echo "argv: $*" | cut -c1-400\n')
    monkeypatch.setattr(settings, "claude_binary", str(bins / "fake-claude"))
    docs = tmp_path / "papers"
    docs.mkdir()
    out = tmp_path / "sweep"
    rc = fanout_main.main(
        [
            "sweep",
            "--prompt",
            "Q?",
            "--dir",
            str(docs),
            "--out",
            str(out),
            "--profile",
            "papers",
            "--models",
            "claude:fable",
            "--claude-tools",
            "Read,Glob",
        ]
    )
    assert rc == 0
    m = fanout_main.load_manifest(out)
    assert m.profile.name == "papers" and m.target == docs
    assert m.profile.claude_allowed_tools == ["Read", "Glob"]
    findings = (out / "findings" / "A-claude-fable.md").read_text()
    assert "--allowedTools Read Glob" in findings
    assert "**Quote:**" in (out / "prompt.md").read_text()
    assert "MISREAD" in (out / "judge-prompt.md").read_text()
    assert "[papers]" in capsys.readouterr().err


def test_profiles_subcommand_lists_builtins(capsys):
    assert fanout_main.main(["profiles"]) == 0
    out = capsys.readouterr().out
    for name in builtin_names():
        assert name in out


def test_raw_profile_passes_the_prompt_verbatim_and_no_system_prompt_by_default(tmp_path):
    raw = load_profile("raw")
    assert wrap_sweep_prompt("  exactly this  ", tmp_path, raw) == "exactly this"
    cmd = build_cmd(fanout_main.parse_one_seat("pi:gemma4-26b", "A"), "P", raw)
    assert "--system-prompt" not in cmd
    assert "--system-prompt" not in build_cmd(fanout_main.parse_one_seat("opus", "B"), "P", raw)


def test_system_prompt_override_reaches_both_clis(tmp_path):
    prof = load_profile("raw").model_copy(update={"system_prompt": "You are terse."})
    claude_cmd = build_cmd(fanout_main.parse_one_seat("opus", "A"), "P", prof)
    assert claude_cmd[claude_cmd.index("--system-prompt") + 1] == "You are terse."
    pi_cmd = build_cmd(fanout_main.parse_one_seat("pi:gemma4-26b", "B"), "P", prof)
    i = pi_cmd.index("--system-prompt")
    assert pi_cmd[i + 1] == "You are terse." and pi_cmd[i + 2 :] == ["--", "P"]


def test_sweep_system_prompt_flag_reads_at_file_and_is_recorded(tmp_path, monkeypatch):
    bins = tmp_path / "bin"
    bins.mkdir()
    _fake_bin(bins, "fake-claude", 'echo "argv: $*" | cut -c1-600\n')
    monkeypatch.setattr(settings, "claude_binary", str(bins / "fake-claude"))
    sysf = tmp_path / "sys.md"
    sysf.write_text("Answer in Latin.")
    docs = tmp_path / "d"
    docs.mkdir()
    out = tmp_path / "sweep"
    rc = fanout_main.main(
        [
            "sweep",
            "--profile",
            "raw",
            "--prompt",
            "Why?",
            "--dir",
            str(docs),
            "--out",
            str(out),
            "--models",
            "claude:fable",
            "--system-prompt",
            f"@{sysf}",
        ]
    )
    assert rc == 0
    m = fanout_main.load_manifest(out)
    assert m.profile.name == "raw" and m.profile.system_prompt == "Answer in Latin."
    assert (out / "prompt.md").read_text() == "Why?"  # verbatim, no wrapper
    findings = (out / "findings" / "A-claude-fable.md").read_text()
    assert "--system-prompt Answer in Latin." in findings
    assert "no wrapper" in (out / "judge-prompt.md").read_text()
