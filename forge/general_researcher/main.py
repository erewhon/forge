"""Iterative research harness for scoped, focused topics.

Pattern: plan -> research -> verify, looping until a sprint passes the
quality threshold (or `--max-sprints` is exhausted). After the loop, a
synthesizer combines all sprint findings into a single coherent report.

Each topic lives in its own directory under `project_dir/<slug>/`. Re-running
with the same question (or `--slug`) resumes from where the last run stopped.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from forge.general_researcher.config import settings
from forge.general_researcher.models import (
    SprintFindings,
    TopicConfig,
    VerificationResult,
)
from forge.general_researcher.planner import create_sprint
from forge.general_researcher.renderer import (
    render_findings_context,
    render_findings_summary,
    render_sprint_findings,
    render_synthesis,
    render_verification,
)
from forge.general_researcher.researcher import execute_sprint
from forge.general_researcher.synthesizer import synthesize
from forge.general_researcher.verifier import verify_sprint


def _slugify(text: str, max_len: int = 60) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "untitled"


def _load_topic_config(arg: str) -> TopicConfig:
    path = Path(arg).expanduser()
    if path.is_file():
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return TopicConfig.model_validate(data)
    return TopicConfig(question=arg)


def _topic_dir(topic: TopicConfig) -> Path:
    slug = topic.slug or _slugify(topic.question)
    return settings.project_dir / slug


def _scan_existing(topic_dir: Path) -> tuple[list[SprintFindings], list[VerificationResult]]:
    findings_dir = topic_dir / "findings"
    sprints_dir = topic_dir / "sprints"
    all_findings: list[SprintFindings] = []
    all_reviews: list[VerificationResult] = []

    if findings_dir.exists():
        for jf in sorted(findings_dir.glob("sprint-*.json")):
            try:
                all_findings.append(SprintFindings.model_validate(json.loads(jf.read_text())))
            except Exception:
                continue

    if sprints_dir.exists():
        for jf in sorted(sprints_dir.glob("sprint-*-review.json")):
            try:
                all_reviews.append(VerificationResult.model_validate(json.loads(jf.read_text())))
            except Exception:
                continue

    return all_findings, all_reviews


def _count_existing_sprints(topic_dir: Path) -> int:
    sprints_dir = topic_dir / "sprints"
    if not sprints_dir.exists():
        return 0
    return len(
        [p for p in sprints_dir.glob("sprint-*.json") if not p.name.endswith("-review.json")]
    )


def run(
    topic: TopicConfig,
    *,
    max_sprints: int | None = None,
    always_deepen: bool = False,
    dry_run: bool = False,
) -> None:
    topic_dir = _topic_dir(topic)
    topic_dir.mkdir(parents=True, exist_ok=True)
    sprints_dir = topic_dir / "sprints"
    findings_dir = topic_dir / "findings"
    sprints_dir.mkdir(exist_ok=True)
    findings_dir.mkdir(exist_ok=True)

    # Persist the topic config so re-runs are deterministic
    (topic_dir / "topic.yaml").write_text(
        yaml.safe_dump(topic.model_dump(exclude_none=True), sort_keys=False)
    )

    print(f"Topic: {topic.question}")
    print(f"Directory: {topic_dir}")
    print()

    all_findings, all_reviews = _scan_existing(topic_dir)
    if all_findings:
        print(f"Resuming: {len(all_findings)} prior sprint(s) found.")
        print()

    sprint_offset = _count_existing_sprints(topic_dir)
    sprint_limit = max_sprints if max_sprints is not None else settings.max_sprints_per_run
    follow_up_feedback: str | None = None
    passed_any = any(r.passed for r in all_reviews)

    for i in range(sprint_limit):
        sprint_number = sprint_offset + i + 1
        print(f"{'=' * 60}")
        print(f"SPRINT {sprint_number}")
        print(f"{'=' * 60}")
        print()

        # Plan
        print("--- Planning ---")
        summary = render_findings_summary(all_findings, max_chars=2000)
        contract = create_sprint(
            topic,
            summary,
            sprint_number,
            follow_up_feedback=follow_up_feedback,
        )
        contract_path = sprints_dir / f"sprint-{contract.sprint_id}.json"
        contract_path.write_text(contract.model_dump_json(indent=2))
        print(f"  Questions: {len(contract.questions)}")
        if contract.rationale:
            print(f"  Rationale: {contract.rationale[:120]}")
        print()

        if dry_run:
            print("  [DRY RUN] Skipping research and verification.")
            print()
            follow_up_feedback = None
            continue

        # Research
        print("--- Researching ---")
        prior_context = render_findings_context(
            all_findings,
            max_chars=settings.max_findings_tokens * 4,
        )
        findings = execute_sprint(contract, prior_context=prior_context)
        (findings_dir / f"sprint-{contract.sprint_id}.json").write_text(
            findings.model_dump_json(indent=2)
        )
        (findings_dir / f"sprint-{contract.sprint_id}.md").write_text(
            render_sprint_findings(findings)
        )
        print(f"  Collected {len(findings.findings)} findings")
        print()

        # Verify
        print("--- Verifying ---")
        result = verify_sprint(topic, contract, findings)
        (sprints_dir / f"sprint-{contract.sprint_id}-review.json").write_text(
            result.model_dump_json(indent=2)
        )
        print(render_verification(result))
        print()

        all_findings.append(findings)
        all_reviews.append(result)

        if result.passed:
            passed_any = True
            follow_up_feedback = None
            if not always_deepen:
                print(
                    f"PASSED with {result.scores.overall}/10. "
                    f"Stopping (use --always-deepen to keep going)."
                )
                break
            print(f"PASSED with {result.scores.overall}/10. Continuing (--always-deepen).")
        else:
            print(
                f"FAILED ({result.scores.overall}/10 < threshold). "
                f"Folding feedback into next sprint."
            )
            follow_up_feedback = (
                f"Previous sprint scored {result.scores.overall}/10.\n"
                f"Feedback: {result.feedback}\n"
                f"Follow-up questions: {', '.join(result.follow_up_questions)}"
            )
        print()

    if dry_run:
        return

    # Synthesis (always, even if no sprint passed — surface incompleteness in output)
    print(f"{'=' * 60}")
    print("SYNTHESIS")
    print(f"{'=' * 60}")
    print()
    if not all_findings:
        print("No findings produced. Skipping synthesis.")
        return
    synth = synthesize(topic, all_findings, all_reviews)
    (topic_dir / "synthesis.json").write_text(synth.model_dump_json(indent=2))
    synth_md = render_synthesis(synth, topic)
    (topic_dir / "synthesis.md").write_text(synth_md)
    print(f"Synthesis written to {topic_dir / 'synthesis.md'}")
    if not passed_any:
        print("(no sprint passed verification — synthesis is provisional)")


def print_summary(topic: TopicConfig) -> None:
    topic_dir = _topic_dir(topic)
    if not topic_dir.exists():
        print(f"No research directory at {topic_dir}")
        return
    all_findings, all_reviews = _scan_existing(topic_dir)
    print(f"Topic: {topic.question}")
    print(f"Directory: {topic_dir}")
    print(f"Sprints completed: {len(all_findings)}")
    print(f"Verifications: {len(all_reviews)}")
    if all_reviews:
        scores = [r.scores.overall for r in all_reviews]
        print(f"Best score: {max(scores)}/10  Worst: {min(scores)}/10")
        print(f"Passing: {sum(1 for r in all_reviews if r.passed)}/{len(all_reviews)}")
    synth_path = topic_dir / "synthesis.md"
    if synth_path.exists():
        print(f"Synthesis: {synth_path}")
    else:
        print("Synthesis: not yet produced")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Iterative research harness using plan->research->verify sprint cycles"
    )
    parser.add_argument(
        "topic",
        help="Research question (string) OR path to a topic YAML/JSON file",
    )
    parser.add_argument(
        "--max-sprints",
        type=int,
        default=None,
        help=f"Cap on sprints per run (default: {settings.max_sprints_per_run})",
    )
    parser.add_argument(
        "--always-deepen", action="store_true", help="Keep running sprints even after one passes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan sprints without executing research/verification/synthesis",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print existing research status and exit"
    )
    parser.add_argument("--slug", default=None, help="Override the auto-derived directory slug")
    args = parser.parse_args(argv)

    topic = _load_topic_config(args.topic)
    if args.slug:
        topic.slug = args.slug

    if args.summary:
        print_summary(topic)
        return 0

    run(
        topic,
        max_sprints=args.max_sprints,
        always_deepen=args.always_deepen or settings.always_deepen,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
