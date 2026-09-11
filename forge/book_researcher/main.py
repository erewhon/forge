from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from forge.book_researcher.brief import effective_policy, render_chapter_brief
from forge.book_researcher.config import settings
from forge.book_researcher.importer import import_research
from forge.book_researcher.models import BookConfig, SourcePolicy, SprintFindings
from forge.book_researcher.planner import create_sprint
from forge.book_researcher.probe import load_reachability
from forge.book_researcher.renderer import render_knowledge_summary, render_verification
from forge.book_researcher.researcher import execute_sprint, sprint_has_content
from forge.book_researcher.scaffold import DEFAULT_FILENAME, write_skeleton
from forge.book_researcher.verifier import verify_sprint
from forge.shared.llm import RESEARCH_FAILED_PREFIX

# Failed attempts on one chapter before the planner is told to go elsewhere. Low on purpose: the
# point is coverage across the outline, and a chapter that has missed the threshold twice is
# usually blocked by something a third identical attempt will not fix (unreachable sources, a
# question that cannot be satisfied as posed) rather than by insufficient effort.
MAX_ATTEMPTS_PER_CHAPTER = 2


def _load_book_config(config_path: str) -> BookConfig:
    """Load book configuration from YAML or JSON file."""
    path = Path(config_path).expanduser().resolve()
    text = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    return BookConfig.model_validate(data)


def _policy_for(book: BookConfig) -> SourcePolicy:
    """The YAML source policy merged with the last `forge book probe` (if any)."""
    return effective_policy(book, load_reachability(settings.project_dir))


def _scan_existing_knowledge() -> dict[int, list[str]]:
    """Scan knowledge directory to understand what's already been researched.

    Returns a mapping of chapter number to list of questions already covered.
    """
    knowledge: dict[int, list[str]] = {}
    knowledge_dir = settings.knowledge_dir

    if not knowledge_dir.exists():
        return knowledge

    for chapter_dir in sorted(knowledge_dir.iterdir()):
        if not chapter_dir.is_dir() or not chapter_dir.name.startswith("chapter-"):
            continue

        try:
            chapter_num = int(chapter_dir.name.split("-")[1])
        except (IndexError, ValueError):
            continue

        questions: list[str] = []
        for json_file in sorted(chapter_dir.glob("sprint-*.json")):
            try:
                data = json.loads(json_file.read_text())
                findings = SprintFindings.model_validate(data)
                questions.extend(f.question for f in findings.findings)
            except Exception:
                continue

        if questions:
            knowledge[chapter_num] = questions

    return knowledge


def _get_chapter_context(chapter_num: int) -> str:
    """Load existing research for a chapter as context for the researcher.

    Reads the structured JSON rather than the rendered markdown so unsourced and failed findings
    can be filtered out. That filter matters: a fabricated court docket entered chapter 2 in
    sprint 1 with ZERO sources, and because the whole markdown was fed forward verbatim as
    "existing research context", sprint 2 restated it at HIGH confidence with sources gathered for
    adjacent facts — laundering an invention into apparent fact across four further sprints.

    Surviving claims are labelled unverified so the model re-verifies rather than treating the
    harness's own memory as authority.
    """
    chapter_dir = settings.knowledge_dir / f"chapter-{chapter_num:02d}"
    if not chapter_dir.exists():
        return ""

    context_parts: list[str] = []
    for json_file in sorted(chapter_dir.glob("sprint-*.json")):
        try:
            sf = SprintFindings.model_validate(json.loads(json_file.read_text()))
        except Exception:
            continue
        for f in sf.findings:
            if f.answer.startswith(RESEARCH_FAILED_PREFIX) or not f.sources:
                continue  # never feed an unsourced or failed claim forward
            context_parts.append(f"### {f.question}\n{f.answer}\nSources: {', '.join(f.sources)}\n")

    if not context_parts:
        return ""

    header = (
        "The following are UNVERIFIED CLAIMS from earlier sprints on this chapter, not established "
        "facts. They may contain errors. Do not restate any of them as settled — if you rely on "
        "one, verify it against a source you retrieve yourself this session and cite that source. "
        "Never treat this section as itself a source.\n\n"
    )
    full_context = header + "\n---\n".join(context_parts)
    # Truncate to avoid overwhelming the researcher's context
    max_chars = settings.max_findings_tokens * 4
    if len(full_context) > max_chars:
        full_context = full_context[:max_chars] + "\n\n[... earlier research truncated ...]"

    return full_context


def _count_existing_sprints() -> int:
    """Count how many sprints have already been run."""
    sprints_dir = settings.sprints_dir
    if not sprints_dir.exists():
        return 0
    return len(list(sprints_dir.glob("sprint-[0-9]*.json")))


def run(config_path: str, *, max_sprints: int | None = None, dry_run: bool = False) -> None:
    """Run the research sprint cycle."""
    # 1. Load book config
    book_config = _load_book_config(config_path)
    policy = _policy_for(book_config)
    print(f"Book: {book_config.title}")
    print(f"Chapters: {len(book_config.chapters)}")
    if policy.blocked:
        print(f"Blocked hosts (from sources/probe): {', '.join(policy.blocked)}")
    print()

    # Ensure project directories exist
    settings.project_dir.mkdir(parents=True, exist_ok=True)
    settings.sprints_dir.mkdir(parents=True, exist_ok=True)
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)

    # 2. Scan existing knowledge
    existing_knowledge = _scan_existing_knowledge()
    if existing_knowledge:
        print("Existing research coverage:")
        for ch_num, questions in sorted(existing_knowledge.items()):
            print(f"  Chapter {ch_num}: {len(questions)} questions covered")
        print()
    else:
        print("No existing research found. Starting fresh.")
        print()

    sprint_limit = max_sprints if max_sprints is not None else settings.max_sprints_per_run
    sprint_offset = _count_existing_sprints()
    follow_up_feedback: str | None = None
    # Attempts per chapter this run. A failing sprint's feedback pushes the planner back at the
    # same chapter, so without a cap one unpassable chapter absorbs every sprint — observed live,
    # six of six sprints on chapter 2 while chapters 3-10 got nothing. After the cap the chapter is
    # declared exhausted and the planner is told to go elsewhere.
    chapter_attempts: dict[int, int] = {}
    exhausted_chapters: set[int] = set()

    # 3. Sprint cycle
    for i in range(sprint_limit):
        sprint_number = sprint_offset + i + 1
        print(f"{'=' * 60}")
        print(f"SPRINT {sprint_number}")
        print(f"{'=' * 60}")
        print()

        # a. Plan
        print("--- Planning ---")
        contract = create_sprint(
            book_config,
            existing_knowledge,
            sprint_number,
            exhausted_chapters=exhausted_chapters,
            follow_up_feedback=follow_up_feedback,
            policy=policy,
        )
        print(f"  Target: Chapter {contract.chapter}")
        print(f"  Questions: {len(contract.questions)}")
        print(f"  Priority: {contract.priority}")
        print()

        if dry_run:
            print("  [DRY RUN] Skipping research and verification.")
            print()
            follow_up_feedback = None
            continue

        # b. Research
        print("--- Researching ---")
        chapter_context = _get_chapter_context(contract.chapter)
        chapter = book_config.chapter(contract.chapter)
        brief = render_chapter_brief(book_config, chapter, policy) if chapter else ""
        findings = execute_sprint(contract, chapter_context=chapter_context, brief=brief)
        print(f"  Collected {len(findings.findings)} findings")
        print()

        # Abort a dead sprint before running the verifier panel on emptiness. Every finding
        # empty/failed means the tool proxy or its web egress is down, so further sprints fail
        # the same way. Stop loud rather than grind the cap. See Forge task 41c3a3b3.
        if not sprint_has_content(findings):
            print(
                "  All findings this sprint are empty/failed — aborting.\n"
                "  The router tool proxy or its web egress is likely down; check it, then re-run."
            )
            print()
            break

        # c. Verify
        print("--- Verifying ---")
        result = verify_sprint(contract, findings)
        print(render_verification(result))
        print()

        # d/e. Decide next action
        chapter_attempts[contract.chapter] = chapter_attempts.get(contract.chapter, 0) + 1
        if result.passed:
            print(f"  PASSED (score: {result.scores.overall}/10)")
            follow_up_feedback = None
            # Update knowledge index
            existing_knowledge = _scan_existing_knowledge()
        else:
            print(
                f"  FAILED (score: {result.scores.overall}/10, "
                f"threshold: {settings.score_threshold})"
            )
            follow_up_feedback = (
                f"Previous sprint {contract.sprint_id} scored {result.scores.overall}/10.\n"
                f"Feedback: {result.feedback}\n"
                f"Follow-up questions: {', '.join(result.follow_up_questions)}"
            )
            if chapter_attempts[contract.chapter] >= MAX_ATTEMPTS_PER_CHAPTER:
                exhausted_chapters.add(contract.chapter)
                print(
                    f"  Chapter {contract.chapter} has now failed "
                    f"{chapter_attempts[contract.chapter]} attempts — moving on so the remaining "
                    f"sprints reach other chapters. Its findings are kept; re-run to revisit it."
                )
                if len(exhausted_chapters) >= len(book_config.chapters):
                    print("  Every chapter has hit its attempt limit — stopping.")
                    print()
                    break

        print()

    # 4. Final summary
    print(f"{'=' * 60}")
    print("RESEARCH SUMMARY")
    print(f"{'=' * 60}")
    print()
    summary = render_knowledge_summary(book_config, settings.knowledge_dir)
    print(summary)


def print_summary(config_path: str) -> None:
    """Print current knowledge summary without running sprints."""
    book_config = _load_book_config(config_path)
    summary = render_knowledge_summary(book_config, settings.knowledge_dir)
    print(summary)


def _init(argv: list[str]) -> int:
    """`meta book init [path]` — write a skeleton book config to edit."""
    parser = argparse.ArgumentParser(
        prog="forge book init",
        description="Write a skeleton book config YAML to fill in",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help=f"Where to write it (default: ./{DEFAULT_FILENAME}; a directory gets "
        f"{DEFAULT_FILENAME} appended)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the file if it already exists",
    )
    args = parser.parse_args(argv)

    try:
        target = write_skeleton(args.path, force=args.force)
    except FileExistsError as e:
        print(f"error: {e} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: could not write skeleton: {e}", file=sys.stderr)
        return 1

    print(f"Wrote book config skeleton to {target}")
    print(f"Edit it, then run:  meta book {target} --max-sprints 10")
    return 0


def _import_research(argv: list[str]) -> int:
    """`forge book import-research <slug> --chapter N` — pull a research topic into a chapter."""
    parser = argparse.ArgumentParser(
        prog="forge book import-research",
        description=(
            "Transplant a `forge research` topic's findings into a book chapter's knowledge dir, "
            "so `forge book` treats them as prior context instead of re-researching them."
        ),
    )
    parser.add_argument(
        "slug",
        help="Research topic slug (the dir name under GENERAL_RESEARCHER_PROJECT_DIR)",
    )
    parser.add_argument(
        "--chapter", type=int, required=True, help="Target chapter number in this book"
    )
    args = parser.parse_args(argv)

    try:
        result = import_research(args.slug, args.chapter)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    dropped = f" ({result.dropped} empty/failed skipped)" if result.dropped else ""
    print(
        f"Imported {result.imported} finding(s){dropped} from research topic "
        f"'{args.slug}' into chapter {args.chapter}."
    )
    print(f"  source: {result.source_dir}")
    print(f"  wrote:  {result.json_path}")
    print("Run `forge book <config> --summary` to see it in the knowledge summary.")
    return 0


def _lint(argv: list[str]) -> int:
    """`forge book lint <config>` — deterministic outline checks, optional model critic."""
    from forge.book_researcher.lint import (
        critique_chapter,
        critiques_to_findings,
        has_errors,
        lint_book,
        render_lint_report,
    )

    parser = argparse.ArgumentParser(
        prog="forge book lint",
        description=(
            "Check an outline for what the verifier panel will dock — no counter-narrative "
            "question, compound questions, unnamed subjects, blocked hosts, placeholders — "
            "before spending a run on it. Errors exit 1; warnings are advice."
        ),
    )
    parser.add_argument("config", help="Path to book config YAML/JSON")
    parser.add_argument(
        "--critic",
        action="store_true",
        help="Also ask the outline model pool to grade each question against the verifier "
        "rubric (one call per chapter; uses BOOK_RESEARCHER_OUTLINE_MODELS)",
    )
    parser.add_argument(
        "--chapter", type=int, action="append", help="Only critique these chapters (repeatable)"
    )
    parser.add_argument("--strict", action="store_true", help="Exit 1 on warnings too")
    args = parser.parse_args(argv)

    try:
        book = _load_book_config(args.config)
    except Exception as e:  # noqa: BLE001 — a schema error IS the lint result here
        print(f"error: {args.config} does not load as a book config: {e}", file=sys.stderr)
        return 1
    policy = _policy_for(book)
    findings = lint_book(book, policy)

    if args.critic:
        from forge.book_researcher.outline import outline_pool

        pool = outline_pool()
        wanted = set(args.chapter or [])
        critiques = []
        for ch in book.chapters:
            if wanted and ch.number not in wanted:
                continue
            print(f"  critiquing chapter {ch.number} ({len(ch.research_questions)} questions)...")
            c = critique_chapter(
                book,
                ch,
                policy,
                pool=pool,
                max_tokens=settings.outline_max_tokens,
                timeout=settings.outline_timeout,
            )
            if c is not None:
                critiques.append(c)
        findings.extend(critiques_to_findings(critiques, threshold=settings.score_threshold))
        print()

    print(render_lint_report(book, findings))
    if has_errors(findings):
        return 1
    if args.strict and findings:
        return 1
    return 0


def _probe(argv: list[str]) -> int:
    """`forge book probe <config>` — test which source hosts the proxy egress can fetch."""
    from forge.book_researcher.probe import hosts_in_config, probe_book, render_reachability

    parser = argparse.ArgumentParser(
        prog="forge book probe",
        description=(
            "Fetch https://<host>/ for every host the outline names (source policy, chapter "
            "sources, hosts in question text) through the researcher's egress, and record which "
            "answer, which 401/403, and which never respond. No model. Writes reachability.json "
            "in the project dir; the planner and researcher read it on every run."
        ),
    )
    parser.add_argument("config", help="Path to book config YAML/JSON")
    parser.add_argument(
        "--host", action="append", default=[], help="Extra host to probe (repeatable)"
    )
    parser.add_argument(
        "--proxy",
        default=settings.check_sources_proxy,
        help="Egress proxy URL (default: BOOK_RESEARCHER_CHECK_SOURCES_PROXY)",
    )
    parser.add_argument("--timeout", type=float, default=settings.check_sources_timeout)
    args = parser.parse_args(argv)

    book = _load_book_config(args.config)
    hosts = hosts_in_config(book) | set(args.host)
    if not hosts:
        print(
            "No hosts to probe: name repositories under sources:, chapter sources:, or in "
            "questions, or pass --host."
        )
        return 0
    print(f"Probing {len(hosts)} host(s)" + (f" via {args.proxy}" if args.proxy else " directly"))
    reach = probe_book(
        book,
        settings.project_dir,
        extra_hosts=args.host,
        timeout=args.timeout,
        proxy=args.proxy,
    )
    print(render_reachability(reach, only=hosts))
    print(f"Wrote {settings.project_dir / 'reachability.json'}")
    if not args.proxy:
        print(
            "NOTE: probed directly, not through the tool proxy's egress — set "
            "BOOK_RESEARCHER_CHECK_SOURCES_PROXY to the proxy's egress so verdicts match what "
            "the researcher can actually fetch."
        )
    return 0


def _revise(argv: list[str]) -> int:
    """`forge book revise <config>` — propose outline edits from the verifier reviews."""
    from forge.book_researcher.lint import has_errors, lint_book, render_lint_report
    from forge.book_researcher.revise import (
        apply_edits,
        gather_evidence,
        policy_for,
        proposal_paths,
        propose_revision,
        render_evidence,
        render_revision_report,
        resolve_edits,
        save_proposal_json,
        unified_diff,
    )

    parser = argparse.ArgumentParser(
        prog="forge book revise",
        description=(
            "Read the sprint reviews, coverage, and probe results for this book and propose "
            "edits to the outline (rewrite/add/remove questions, add guidance, block hosts). "
            "Writes <name>.proposed.yaml + <name>.revision.md and prints the diff; book.yaml "
            "itself is untouched unless --apply."
        ),
    )
    parser.add_argument("config", help="Path to book config YAML/JSON")
    parser.add_argument(
        "--report", action="store_true", help="Print the evidence digest only (no model call)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the applicable edits into the config in place (refused if lint errors)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    book = _load_book_config(str(config_path))
    ev = gather_evidence(book, settings.project_dir, max_reviews=settings.max_reviews_per_chapter)
    if args.report:
        print(render_evidence(book, ev))
        return 0
    if ev.total_sprints == 0:
        print(
            f"No sprints found under {settings.project_dir} — nothing to revise from. Run "
            "`forge book <config>` first, or check BOOK_RESEARCHER_PROJECT_DIR."
        )
        return 1

    from forge.book_researcher.outline import outline_pool

    print(f"Gathered evidence from {ev.total_sprints} sprint(s); proposing edits...")
    try:
        proposal = propose_revision(
            book,
            ev,
            pool=outline_pool(),
            max_tokens=settings.outline_max_tokens,
            timeout=settings.outline_timeout,
        )
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    saved = save_proposal_json(settings.project_dir, proposal)

    resolved = resolve_edits(book, proposal.edits)
    before = config_path.read_text()
    if config_path.suffix not in (".yaml", ".yml"):
        print(
            "error: --apply / proposed output needs a YAML config (JSON edits are not supported)",
            file=sys.stderr,
        )
        return 1
    after = apply_edits(before, resolved)
    diff = unified_diff(before, after, config_path.name)
    report = render_revision_report(book, proposal, resolved, diff)

    proposed_path, report_path = proposal_paths(config_path)
    proposed_path.write_text(after)
    report_path.write_text(report)
    print(report)
    print(f"Proposal saved: {saved}")
    print(f"Proposed outline: {proposed_path}")
    print(f"Report: {report_path}")

    if not args.apply:
        print(
            f"\nReview the diff, then `forge book revise {args.config} --apply` to write it, "
            "or edit the proposed file and copy it over by hand."
        )
        return 0

    try:
        new_book = BookConfig.model_validate(yaml.safe_load(after))
    except Exception as e:  # noqa: BLE001
        print(f"error: proposed outline does not validate — not applied: {e}", file=sys.stderr)
        return 1
    findings = lint_book(new_book, policy_for(new_book, ev))
    if has_errors(findings):
        print(render_lint_report(new_book, findings))
        print("error: proposed outline has lint errors — not applied.", file=sys.stderr)
        return 1
    config_path.write_text(after)
    print(f"Applied {sum(1 for r in resolved if r.applicable)} edit(s) to {config_path}")
    return 0


def _decompose(argv: list[str]) -> int:
    """`forge book decompose <recon-slug>` — frame (human gate) then decompose an outline."""
    from forge.book_researcher.decompose import (
        DecomposeError,
        FramingExistsError,
        approve_framing,
        decompose,
        load_framing,
        load_human_framing,
        load_recon,
        persist_framing,
        propose_framing,
        render_framing,
        write_decomposed,
    )
    from forge.book_researcher.lint import lint_book, render_lint_report
    from forge.book_researcher.outline import outline_pool

    parser = argparse.ArgumentParser(
        prog="forge book decompose",
        description=(
            "Derive an outline from a `forge research` recon run in two gated steps. First run "
            "proposes a FRAMING (thesis, slicing options, chapter sketch, guidance) into the "
            "project dir for you to read and edit; `--approve` opens the gate and DECOMPOSES "
            "the approved framing into book.yaml, validated by the linter. "
            "`--framing <file>` skips the model framing with one you wrote yourself."
        ),
    )
    parser.add_argument(
        "slug", help="Research topic slug (dir under GENERAL_RESEARCHER_PROJECT_DIR)"
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_FILENAME,
        help=f"Where to write the outline (default ./{DEFAULT_FILENAME})",
    )
    parser.add_argument(
        "--framing", help="A human-written framing YAML/JSON (approved by construction)"
    )
    parser.add_argument("--approve", action="store_true", help="Approve the stored framing")
    parser.add_argument(
        "--reframe", action="store_true", help="Re-propose the framing, overwriting the stored one"
    )
    parser.add_argument("--chapters", type=int, default=None, help="Target chapter count")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing outline file")
    args = parser.parse_args(argv)

    try:
        recon = load_recon(args.slug)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    reach = load_reachability(settings.project_dir)
    framing_path = settings.framing_file

    if args.approve:
        try:
            framing = approve_framing(framing_path)
        except DecomposeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"Approved framing at {framing_path}; decomposing it now.")
    elif args.framing:
        framing = load_human_framing(Path(args.framing).expanduser())
        persist_framing(framing, framing_path, force=True)
        print(f"Using your framing from {args.framing} (stored at {framing_path})")
    else:
        framing = None if args.reframe else load_framing(framing_path)
        if framing is None:
            print(f"Framing {args.slug} ({recon.sprint_count} recon sprints)...")
            try:
                framing = propose_framing(
                    recon,
                    pool=outline_pool(),
                    reachability=reach,
                    max_tokens=settings.outline_max_tokens,
                    timeout=settings.outline_timeout,
                )
                persist_framing(framing, framing_path, force=args.reframe)
            except FramingExistsError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            except DecomposeError as e:
                print(f"error: {e}", file=sys.stderr)
                if e.raw:
                    print(f"last raw output:\n{e.raw[:1500]}", file=sys.stderr)
                return 1
            print(render_framing(framing))
            print(f"Framing written to {framing_path} (+ .md).")
            print(
                "Read it, edit the JSON if you disagree with the slice, then run\n"
                f"  forge book decompose {args.slug} --approve   # approves, then decomposes"
            )
            return 0

    if not framing.approved:
        print(
            f"Framing at {framing_path} is not approved. Read {framing_path.with_suffix('.md')}, "
            f"then `forge book decompose {args.slug} --approve`.",
            file=sys.stderr,
        )
        return 1

    out_path = Path(args.out).expanduser()
    if out_path.is_dir():
        out_path = out_path / DEFAULT_FILENAME
    if out_path.exists() and not args.force:
        print(f"error: {out_path} already exists (use --force to overwrite)", file=sys.stderr)
        return 1

    print(f"Decomposing approved framing ({len(framing.chapter_sketch)} chapters sketched)...")
    try:
        book = decompose(
            framing,
            recon,
            pool=outline_pool(),
            reachability=reach,
            chapter_count=args.chapters,
            max_tokens=settings.outline_max_tokens,
            timeout=settings.outline_timeout,
        )
    except DecomposeError as e:
        print(f"error: {e}", file=sys.stderr)
        if e.raw:
            print(f"last raw output:\n{e.raw[:1500]}", file=sys.stderr)
        return 1
    written = write_decomposed(book, out_path, recon, framing, force=args.force)
    print(f"Wrote {written}")
    print(render_lint_report(book, lint_book(book, book.sources)))
    print(
        f"\nNext: `forge book {written} --dry-run`, then `forge book {written} --max-sprints 10`."
    )
    return 0


_SUBCOMMANDS = {
    "init": _init,
    "import-research": _import_research,
    "lint": _lint,
    "probe": _probe,
    "revise": _revise,
    "decompose": _decompose,
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in _SUBCOMMANDS:
        return _SUBCOMMANDS[argv[0]](argv[1:])

    parser = argparse.ArgumentParser(
        description="Book research harness using generator-evaluator sprint cycles",
        epilog=(
            "Subcommands: `init [path]` writes a skeleton config; `lint <config>` checks it; "
            "`probe <config>` tests source-host reachability; `revise <config>` proposes "
            "edits from the reviews; `decompose <recon-slug>` derives an outline from a "
            "`forge research` run; `import-research <slug> --chapter N` seeds a chapter."
        ),
    )
    parser.add_argument("config", help="Path to book config YAML/JSON file")
    parser.add_argument(
        "--max-sprints",
        type=int,
        default=None,
        help=f"Max research sprints to run (default: {settings.max_sprints_per_run})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan sprints without executing research or verification",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Just print current knowledge summary and exit",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Privacy lane: verify with the all-self-hosted panel — no findings leave the "
        "homelab. Trades family purity for privacy (see config); default panel stays vetted-Zen.",
    )
    parser.add_argument(
        "--privacy",
        choices=["local", "zdr", "any"],
        default=None,
        help="X-Router-Privacy tier for every router call this run (default: derived from the "
        "lane — local → local, otherwise zdr). 'any' admits Zen seats on the allowlist's say-so "
        "alone, i.e. the pre-header behaviour.",
    )
    args = parser.parse_args(argv)
    if args.local:
        settings.panel_lane = "local"
    if args.privacy:
        settings.panel_privacy = args.privacy

    if args.summary:
        print_summary(args.config)
    else:
        run(args.config, max_sprints=args.max_sprints, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
