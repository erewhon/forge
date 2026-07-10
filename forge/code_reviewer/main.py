"""Nightly Code Reviewer — scan repos for recent commits and review via LLM."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from forge.code_reviewer.ai import generate_overall_summary, review_repo
from forge.code_reviewer.collectors.changes import collect_all
from forge.code_reviewer.config import settings
from forge.code_reviewer.models import NightlyReport
from forge.code_reviewer.renderer import render_markdown
from forge.code_reviewer.writer import append_to_daily_note


def run(ref_date: date | None = None, *, dry_run: bool = False) -> None:
    if ref_date is None:
        ref_date = date.today()

    date_str = ref_date.isoformat()
    print(f"Nightly Code Reviewer: scanning for {date_str}")

    # 1. Collect changes from all repos
    print(f"\nCollecting changes from {len(settings.repos)} repos...")
    all_changes = collect_all()

    if not all_changes:
        print("\nNo repos had changes in the lookback period. Nothing to review.")
        return

    print(f"\n{len(all_changes)} repo(s) with changes found.")

    # 2. Review each repo via LLM
    print("\nReviewing diffs...")
    reviews = []
    for changes in all_changes:
        review = review_repo(changes)
        reviews.append(review)

    # 3. Generate overall summary
    overall_summary = generate_overall_summary(reviews)

    # 4. Build report
    report = NightlyReport(
        date=date_str,
        repos_reviewed=len(settings.repos),
        repos_with_changes=len(all_changes),
        reviews=reviews,
        overall_summary=overall_summary,
    )

    # 5. Write markdown log for debugging
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"review-{date_str}.md"
    log_file.write_text(render_markdown(report))
    print(f"\nMarkdown log written to {log_file}")

    # 6. Write to Nous daily note (unless dry run)
    if dry_run:
        print("\n--- DRY RUN: Report not written to Nous ---")
        print(render_markdown(report))
    else:
        print("\nAppending to Nous daily note...")
        result = append_to_daily_note(report)
        if result:
            print(f"  Done: {result.get('blocksAdded', '?')} blocks appended")
        else:
            print("  Not appended (note missing or review already present)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nightly Code Reviewer")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to generate review for (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print report without writing to Nous",
    )
    args = parser.parse_args(argv)

    ref_date = date.fromisoformat(args.date) if args.date else None
    run(ref_date, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
