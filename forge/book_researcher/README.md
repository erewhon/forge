# Book Researcher

Generator-evaluator research harness for non-fiction book projects. Runs
sprint cycles of **plan → research → verify**, scoring findings against
gradable criteria and feeding failures back into the next sprint until
quality thresholds are met. All findings accumulate to disk, so runs are
resumable across invocations.

## Pattern

Three specialized roles, each a different LLM call:

- **Planner** — reads the book config and existing coverage, picks the
  most important gap, emits a sprint contract (chapter, 2-4 questions,
  success criteria, priority).
- **Researcher** — answers each question with sources and a confidence
  level. Runs one LLM call per question.
- **Verifier** — scores findings 1-10 across five dimensions, weighted to
  produce an overall score. Returns pass/fail plus feedback and follow-up
  questions.

If the verifier rejects a sprint, its feedback is folded into the *next*
sprint's planner prompt so the harness self-corrects rather than retrying
blind. Findings accumulate as files in the project directory; the planner
sees what's already covered when picking the next gap.

## Pipeline (per sprint)

1. **Plan** — planner LLM emits a `SprintContract`. Written to
   `sprints/sprint-NNN.json`.
2. **Research** — researcher LLM answers each question, optionally given
   prior chapter findings as context. Writes structured JSON
   (`knowledge/chapter-NN/sprint-NNN.json`) and readable markdown
   (`...sprint-NNN.md`).
3. **Verify** — verifier LLM scores five dimensions:
   `source_diversity`, `claim_verification`, `counter_narrative`,
   `depth`, `actionability`. `counter_narrative` and `depth` are
   weighted 1.5×; others 1.0×. Overall is the rounded weighted mean.
   Review written to `sprints/sprint-NNN-review.json`.
4. **Gate** — if `overall >= score_threshold` (default 7), pass and clear
   feedback. Otherwise, capture verifier feedback and pass it into the
   next sprint's planner prompt as `follow_up_feedback`.
5. **Loop** until `--max-sprints` is reached.

**LLM failures degrade gracefully:** planner falls back to the first
uncovered chapter, researcher records the error against the question,
verifier emits all-3s with the error as feedback. The run continues.

## Resumability

Every run scans `knowledge/` for existing sprint findings and `sprints/`
for prior contracts. Sprint numbering continues from the highest existing
sprint, and the planner is told which chapters and questions are already
covered. Re-running with the same config picks up where the last run left
off — there is no separate "init" step.

## Output layout

Under `BOOK_RESEARCHER_PROJECT_DIR` (default
`~/Projects/erewhon/meta/book-research`):

```
sprints/
  sprint-001.json              # planner contract
  sprint-001-review.json       # verifier review
  ...
knowledge/
  chapter-01/
    sprint-001.json            # structured findings
    sprint-001.md              # readable findings
    ...
  chapter-02/
    ...
```

## Usage

Scaffold a config to start (writes `./book.yaml`, refuses to clobber without
`--force`):

```bash
meta book init                 # or: meta book init path/to/book.yaml
```

The skeleton already validates, so you can `--dry-run` it immediately. Or write
one by hand (see `examples/sample-book.yaml`) — chapters with research questions
per chapter:

```yaml
title: "My Book"
description: "What it's about"
chapters:
  - number: 1
    title: "Introduction"
    description: "Overview and thesis"
    research_questions:
      - "What is the central argument?"
      - "What existing literature covers this topic?"
```

Then:

```bash
# Run the default number of sprints (3)
forge/book_researcher/run.sh path/to/your-book.yaml

# Bump it for a longer session
forge/book_researcher/run.sh path/to/your-book.yaml --max-sprints 10

# Plan sprints without spending tokens on research/verification
forge/book_researcher/run.sh path/to/your-book.yaml --dry-run

# Show what's already been collected
forge/book_researcher/run.sh path/to/your-book.yaml --summary
```

## Configuration

Environment variables (all prefixed `BOOK_RESEARCHER_`):

| Var | Purpose | Default |
|---|---|---|
| `PROJECT_DIR` | Where contracts and findings live | `~/Projects/erewhon/meta/book-research` |
| `LLM_BACKEND` | `openai` (router/local) or `anthropic` | `openai` |
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint | `http://localhost:4010/v1` |
| `OPENAI_API_KEY` | Router key | `<your-router-key>` |
| `RESEARCH_MODEL` | Model alias for the researcher | `research` |
| `SYNTHESIS_MODEL` | Model alias for planner + verifier | `coder` |
| `ANTHROPIC_MODEL` | Used when `LLM_BACKEND=anthropic` | `claude-sonnet-4-6` |
| `MAX_SPRINTS_PER_RUN` | Default sprint count if `--max-sprints` not set | `3` |
| `SCORE_THRESHOLD` | Min overall score (1-10) to accept a sprint | `7` |
| `MAX_FINDINGS_TOKENS` | Truncation budget for findings context | `4000` |

Defaults route through the local LLM router on Euclid, so the harness is
unaffected by Anthropic outages. Flip `BOOK_RESEARCHER_LLM_BACKEND=anthropic`
to use Claude instead.

## Files

- `config.py` — settings (Pydantic `BaseSettings`)
- `models.py` — `BookConfig`, `SprintContract`, `SprintFindings`,
  `VerificationScores`, `VerificationResult`
- `planner.py` — emits sprint contracts
- `researcher.py` — executes a sprint's questions
- `verifier.py` — scores findings, computes weighted overall
- `renderer.py` — markdown rendering for findings, reviews, summaries
- `main.py` — orchestrator + CLI
- `run.sh` — uv wrapper
- `examples/sample-book.yaml` — config template

## Notes

- **Web search comes from the LLM router's tool proxy, not this harness.**
  When `RESEARCH_MODEL` is a `tool_proxy: true` model (the default
  `research` alias is), the proxy auto-injects `web_search`,
  `tavily_search`, and `fetch_url` tools and handles the tool loop server
  side. The researcher prompt explicitly tells the model to use them. If
  you switch to a model without a tool proxy, the researcher will fall
  back to memory.
- **Anthropic backend has no search.** `LLM_BACKEND=anthropic` produces
  memory-only research today — the tool proxy only sits in front of the
  local-router models. Stick with the local backend if search matters.
- **No daemon / timer.** Each invocation runs N sprints and exits. Wrap
  in cron or `/loop` for continuous runs.
- **Verifier rigor depends on the synthesis model.** The default routes
  to the `coder` alias on the local router; the prompt is calibrated for
  honesty ("7+ means genuinely good research; 5 is mediocre") but
  weaker models tend to grade more leniently. Bump `SCORE_THRESHOLD` or
  swap to a stronger synthesis model.
