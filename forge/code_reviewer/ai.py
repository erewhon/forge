from __future__ import annotations

import json
import re

from agents.code_reviewer.config import settings
from agents.code_reviewer.models import RepoChanges, RepoReview, RepoScores, ReviewFinding


def _complete(system: str, user_message: str, max_tokens: int = 4096) -> str:
    """Call configured LLM backend."""
    if settings.llm_backend == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        for block in response.content:
            if block.type == "text":
                return block.text.strip()
        return ""
    else:
        import openai

        client = openai.OpenAI(
            base_url=settings.openai_base_url, api_key=settings.openai_api_key
        )
        response = client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )
        return (response.choices[0].message.content or "").strip()


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to find JSON in code blocks first
    code_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if code_match:
        text = code_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find any JSON object in the text
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


_REVIEW_SYSTEM_PROMPT = """\
You are a senior software engineer conducting a nightly code review. Review the following diff for:

1. **Bugs & Logic Errors**: Off-by-one errors, null/undefined handling, race conditions, resource leaks
2. **Security Issues**: Injection vulnerabilities, exposed secrets, missing auth checks, unsafe deserialization
3. **Performance Concerns**: N+1 queries, unnecessary allocations, missing indexes, blocking operations in async code
4. **Code Quality**: Dead code, duplicated logic, overly complex functions, missing error handling at system boundaries
5. **Positive Observations**: Well-structured code, good patterns, thorough error handling worth noting

Do NOT flag:
- Style/formatting issues (handled by linters)
- Missing documentation or comments (unless critical for understanding)
- Test coverage (separate concern)
- Minor naming preferences

## Scoring

Score each dimension from 1-10 based on the diff:

- **security**: Injection risks, auth gaps, secrets exposure, unsafe deserialization
- **correctness**: Logic errors, off-by-one, null handling, race conditions
- **error_handling**: Boundary validation, resource cleanup, graceful degradation
- **performance**: N+1 queries, blocking in async, unnecessary allocations
- **overall**: Weighted judgment across all dimensions

Calibration guide:
- 9-10: Exemplary, actively doing things well
- 7-8: Solid, no concerns
- 5-6: Minor issues worth noting
- 3-4: Significant concerns that should be addressed
- 1-2: Critical issues requiring immediate attention

Score conservatively. Most routine code should be 7-8. Reserve 9-10 for genuinely \
impressive patterns. Use 5-6 freely for anything worth a second look.

Return ONLY valid JSON:
{
  "scores": {
    "security": 8,
    "correctness": 7,
    "error_handling": 6,
    "performance": 8,
    "overall": 7
  },
  "findings": [
    {
      "severity": "critical|warning|info|positive",
      "file_path": "path/to/file.ext",
      "description": "Concise description of the finding"
    }
  ],
  "summary": "1-2 sentence overall assessment"
}

If the code looks solid with no issues, return an empty findings array and a positive summary. Be concise and actionable.\
"""

_SUMMARY_SYSTEM_PROMPT = """\
You are a senior engineering lead summarizing nightly code reviews across multiple repositories. \
Generate a 2-3 sentence overall summary of all the reviews. Focus on the most important findings \
and any cross-cutting concerns. Be concise and actionable.\
"""


def review_repo(changes: RepoChanges) -> RepoReview:
    """Send the diff to the LLM for review and return structured findings."""
    commits_text = "\n".join(f"  - {s}" for s in changes.commit_summaries)
    truncated_note = " (diff was truncated)" if changes.truncated else ""

    user_message = (
        f"Repository: {changes.repo_name}\n"
        f"VCS: {changes.vcs}\n"
        f"Commits ({changes.commit_count}):\n{commits_text}\n\n"
        f"Diff stat:\n{changes.diff_stat}\n\n"
        f"Diff{truncated_note}:\n{changes.diff_text}"
    )

    print(f"  Reviewing {changes.repo_name} ({changes.commit_count} commits)...")
    try:
        response_text = _complete(_REVIEW_SYSTEM_PROMPT, user_message)
    except Exception as e:
        print(f"  Warning: LLM review failed for {changes.repo_name}: {e}")
        return RepoReview(
            repo_name=changes.repo_name,
            findings=[],
            summary=f"Review skipped: LLM error ({type(e).__name__})",
        )
    data = _extract_json(response_text)

    findings: list[ReviewFinding] = []
    for f in data.get("findings", []):
        try:
            findings.append(
                ReviewFinding(
                    severity=f.get("severity", "info"),
                    file_path=f.get("file_path", "unknown"),
                    description=f.get("description", ""),
                )
            )
        except Exception:
            # Skip malformed findings
            continue

    summary = data.get("summary", "Review completed (no structured response from LLM).")

    scores: RepoScores | None = None
    raw_scores = data.get("scores")
    if isinstance(raw_scores, dict):
        try:
            scores = RepoScores(**raw_scores)
        except Exception:
            # LLM returned malformed scores — skip gracefully
            scores = None

    return RepoReview(
        repo_name=changes.repo_name,
        findings=findings,
        summary=summary,
        scores=scores,
    )


def generate_overall_summary(reviews: list[RepoReview]) -> str:
    """Ask the LLM to generate a brief summary of all reviews across repos."""
    if not reviews:
        return "No repositories had changes in the review period."

    parts: list[str] = []
    for review in reviews:
        score_text = ""
        if review.scores:
            s = review.scores
            score_text = (
                f"  Scores: security={s.security} correctness={s.correctness} "
                f"error_handling={s.error_handling} performance={s.performance} "
                f"overall={s.overall}"
            )
        findings_text = ""
        if review.findings:
            findings_text = "\n".join(
                f"  [{f.severity.upper()}] {f.file_path}: {f.description}"
                for f in review.findings
            )
        else:
            findings_text = "  No issues found."
        section = f"{review.repo_name}:\n  Summary: {review.summary}"
        if score_text:
            section += f"\n{score_text}"
        section += f"\n{findings_text}"
        parts.append(section)

    # Compute aggregate scores for repos that have them
    scored_reviews = [r for r in reviews if r.scores is not None]
    aggregate_text = ""
    if scored_reviews:
        n = len(scored_reviews)
        avg = {
            "security": sum(r.scores.security for r in scored_reviews) / n,
            "correctness": sum(r.scores.correctness for r in scored_reviews) / n,
            "error_handling": sum(r.scores.error_handling for r in scored_reviews) / n,
            "performance": sum(r.scores.performance for r in scored_reviews) / n,
            "overall": sum(r.scores.overall for r in scored_reviews) / n,
        }
        aggregate_text = (
            f"\n\nAggregate scores across {n} repos: "
            + ", ".join(f"{k}={v:.1f}" for k, v in avg.items())
        )

    user_message = (
        "Here are the nightly code reviews across all repositories:\n\n"
        + "\n\n".join(parts)
        + aggregate_text
    )

    print("  Generating overall summary...")
    try:
        return _complete(_SUMMARY_SYSTEM_PROMPT, user_message, max_tokens=512)
    except Exception as e:
        print(f"  Warning: overall summary generation failed: {e}")
        return f"Summary unavailable (LLM error: {type(e).__name__})"
