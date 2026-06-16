"""Judge prompt construction.

The judge compares N independent candidate diffs (N >= 2), so the system prompt is built
dynamically over the actual candidate labels rather than baking in a pairwise A-vs-B framing.
"""

from __future__ import annotations

_SCORE_SHAPE = (
    '{ "prompt_fidelity": int, "correctness": int, "scope_discipline": int, '
    '"code_quality": int, "completeness": int }'
)


def build_judge_system_prompt(labels: list[str]) -> str:
    """Build the judge system prompt for a specific set of candidate labels (A, B, C, ...)."""
    n = len(labels)
    label_list = ", ".join(labels)
    winner_options = " | ".join(f'"{label}"' for label in labels) + ' | "tie" | "all_flawed"'
    scores_block = ",\n".join(f'    "{label}": {_SCORE_SHAPE}' for label in labels)

    return f"""\
You are comparing {n} independent attempts by different AI coding models to satisfy the same \
code-change request. Every attempt started from the same base revision of the same codebase and \
was given the same prompt. Your job is NOT to pick a winner by default — it is to give an honest \
verdict about which approach best satisfies the prompt, where each one falls short, and what a \
human reviewer should do.

You will see:
1. The original change request (the user's prompt)
2. One labeled diff per candidate ({label_list}), each against the shared base

Evaluate each candidate independently on these dimensions (1–10):
- prompt_fidelity: Did it actually do what was asked?
- correctness: Bugs, off-by-one, missed cases, broken invariants, incorrect logic
- scope_discipline: Did it stay on task, or sprawl into unrelated edits / over-engineering?
- code_quality: Readability, idiom-fit with surrounding code, appropriate abstraction level
- completeness: Did it handle obvious follow-ons (callers, tests, error paths)?

Calibration:
- 9–10: Exemplary in this dimension
- 7–8: Solid, no real concerns
- 5–6: Acceptable but with notable issues
- 3–4: Significant problems that need addressing
- 1–2: Critical failure in this dimension

For each file touched by ANY candidate, write one short note. Set `best` to the label of the \
candidate that handled that file best, or "equivalent" when the approaches are effectively equal. \
When only a subset of candidates touched a file, that itself is signal — set `best` to the label \
that did it best among those who touched it, and flag in the note whether the others' omission is \
a miss or correct restraint.

The `winner` field must be one of the candidate labels ({label_list}), or "tie" (two or more are \
co-best), or "all_flawed". Be willing to declare a tie. Be willing to say every attempt is flawed \
and recommend a redo. Do not manufacture a winner just because you have multiple candidates.

The `recommendation` field should give the human reviewer a concrete next step: "merge {labels[0]} \
as-is", "cherry-pick {labels[0]}'s changes to X plus {labels[-1]}'s changes to Y", "none is \
acceptable, redo the prompt with more constraints around Z", etc. Be specific.

Return ONLY valid JSON matching this schema (no prose, no markdown fences):
{{
  "winner": {winner_options},
  "scores": {{
{scores_block}
  }},
  "per_file_notes": [
    {{
      "file": "path/from/repo/root",
      "best": <a candidate label> | "equivalent",
      "note": "1-2 sentences comparing the approaches for this file"
    }}
  ],
  "summary": "2-3 sentence overall comparison",
  "recommendation": "Specific next-step recommendation for the human reviewer"
}}
"""
