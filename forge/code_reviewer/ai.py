from __future__ import annotations

import json
import re

from agents.code_reviewer.config import settings
from agents.code_reviewer.models import RepoChanges, RepoReview, ReviewFinding


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

Return ONLY valid JSON:
{
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

    return RepoReview(
        repo_name=changes.repo_name,
        findings=findings,
        summary=summary,
    )


def generate_overall_summary(reviews: list[RepoReview]) -> str:
    """Ask the LLM to generate a brief summary of all reviews across repos."""
    if not reviews:
        return "No repositories had changes in the review period."

    parts: list[str] = []
    for review in reviews:
        findings_text = ""
        if review.findings:
            findings_text = "\n".join(
                f"  [{f.severity.upper()}] {f.file_path}: {f.description}"
                for f in review.findings
            )
        else:
            findings_text = "  No issues found."
        parts.append(f"{review.repo_name}:\n  Summary: {review.summary}\n{findings_text}")

    user_message = (
        "Here are the nightly code reviews across all repositories:\n\n"
        + "\n\n".join(parts)
    )

    print("  Generating overall summary...")
    try:
        return _complete(_SUMMARY_SYSTEM_PROMPT, user_message, max_tokens=512)
    except Exception as e:
        print(f"  Warning: overall summary generation failed: {e}")
        return f"Summary unavailable (LLM error: {type(e).__name__})"
