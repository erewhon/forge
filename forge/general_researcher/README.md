# General Researcher

Iterative research harness for **scoped, focused topics** — one question per
run rather than a whole book outline. Same generator-evaluator pattern as
`book_researcher` (plan → research → verify → loop), with two differences
that suit smaller problems:

- **Stop on first passing sprint** by default. Most scoped questions are
  "done" once one sprint clears the verification threshold; deepening
  further is opt-in via `--always-deepen`.
- **Synthesis step.** After the loop, a fourth role combines all sprint
  findings into a single coherent markdown report (`synthesis.md`) — the
  artifact you actually read.

The researcher prompt explicitly instructs the model to use the search
tools (`web_search`, `tavily_search`, `fetch_url`) that the local LLM
router's tool proxy auto-injects for the `research` alias. So research
draws on live retrieval rather than just model memory.

## Pattern

Four LLM-driven roles:

- **Planner** — picks 2-4 questions for the next sprint. Sprint 1 attacks
  the main question (or user-supplied `sub_questions`); later sprints
  address verifier follow-ups.
- **Researcher** — answers each question, expected to use search/fetch
  tools to ground answers in retrieved sources. One LLM call per question.
- **Verifier** — scores findings 1-10 across `source_diversity`,
  `claim_verification`, `counter_narrative`, `depth`, `actionability`.
  `counter_narrative` and `depth` weighted 1.5×; overall is the rounded
  weighted mean.
- **Synthesizer** — once the loop ends, combines all findings (across all
  sprints) into a single answer with inline citations and a list of open
  questions. If no sprint passed verification, the synthesis is marked
  provisional in its caveat.

Sprint contracts and reviews live alongside structured + readable
findings, so a partial run is always inspectable on disk.

## Pipeline (per sprint)

1. **Plan** — LLM emits questions + success criteria + rationale. Written
   to `sprints/sprint-NNN.json`.
2. **Research** — LLM answers each question with tool use. Writes
   `findings/sprint-NNN.json` (structured) and `.md` (readable).
3. **Verify** — LLM scores. Writes `sprints/sprint-NNN-review.json`.
4. **Gate** — if `overall >= score_threshold` (default 7), pass. Without
   `--always-deepen`, the loop stops here. Otherwise, feed the verifier's
   feedback into the next planner prompt.
5. **Loop** until pass or `--max-sprints` cap.
6. **Synthesize** — always runs once the loop ends (unless `--dry-run`).

LLM failures degrade gracefully: planner falls back to the main question,
researcher records the error in the finding, verifier emits all-3s with
the error as feedback, synthesizer emits a stub. The run continues.

## Resumability

Each topic gets a directory at `<project_dir>/<slug>/`. Slug is
auto-derived from the question (`_slugify`), or pass `--slug` to override.
Re-running with the same question reuses the same directory: the planner
sees prior findings, sprint numbering continues, and synthesis is
regenerated against the full accumulated record.

## Output layout

Under `GENERAL_RESEARCHER_PROJECT_DIR` (default
`~/Projects/erewhon/meta/research`):

```
<topic-slug>/
  topic.yaml                    # the config used
  sprints/
    sprint-001.json             # planner contract
    sprint-001-review.json      # verifier review
    ...
  findings/
    sprint-001.json             # structured findings
    sprint-001.md               # readable findings
    ...
  synthesis.json                # structured synthesis
  synthesis.md                  # final readable report
```

## Usage

**One-shot question:**

```bash
agents/general_researcher/run.sh \
    "What is the history of the IETF standards process?"
```

**Structured topic with context and sub-questions** (see
`examples/sample-topic.yaml`):

```yaml
question: "What is the history of the IETF standards process and how has it evolved?"
context: "Investigating governance models for technical standards organizations."
sub_questions:
  - "When was the IETF formalized?"
  - "How did the RFC process evolve?"
score_threshold: 7
```

```bash
agents/general_researcher/run.sh path/to/topic.yaml
```

**Useful flags:**

| Flag | Purpose |
|---|---|
| `--max-sprints N` | Cap sprints per run (default 5) |
| `--always-deepen` | Keep running after a sprint passes |
| `--dry-run` | Plan only — no research, verification, or synthesis |
| `--summary` | Print existing research status and exit |
| `--slug <name>` | Override auto-derived directory name |

## Configuration

Environment variables (all prefixed `GENERAL_RESEARCHER_`):

| Var | Purpose | Default |
|---|---|---|
| `PROJECT_DIR` | Where topic dirs live | `~/Projects/erewhon/meta/research` |
| `LLM_BACKEND` | `openai` (router/local) or `anthropic` | `openai` |
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint | `http://localhost:4010/v1` |
| `OPENAI_API_KEY` | Router key | `<your-router-key>` |
| `RESEARCH_MODEL` | Alias for the researcher (must be a `tool_proxy: true` model for search) | `research` |
| `SYNTHESIS_MODEL` | Alias for planner + verifier + synthesizer | `coder` |
| `ANTHROPIC_MODEL` | Used when `LLM_BACKEND=anthropic` | `claude-sonnet-4-6` |
| `MAX_SPRINTS_PER_RUN` | Default sprint cap | `5` |
| `SCORE_THRESHOLD` | Min overall score (1-10) to accept a sprint | `7` |
| `MAX_FINDINGS_TOKENS` | Truncation budget for context | `4000` |
| `ALWAYS_DEEPEN` | If `true`, never stop on pass | `false` |

## Files

- `config.py` — settings (Pydantic `BaseSettings`)
- `models.py` — `TopicConfig`, `SprintContract`, `SprintFindings`,
  `VerificationScores`, `VerificationResult`, `Synthesis`
- `planner.py` — emits sprint contracts
- `researcher.py` — executes a sprint's questions (tool-using)
- `verifier.py` — scores findings, computes weighted overall
- `synthesizer.py` — combines all findings into a final report
- `renderer.py` — markdown rendering
- `main.py` — orchestrator + CLI (resume + slug logic)
- `run.sh` — uv wrapper
- `examples/sample-topic.yaml` — config template

## Notes

- **Search depends on the model + proxy combination.** This harness
  doesn't implement search itself — it relies on the local router's tool
  proxy. The default `research` alias goes to a `tool_proxy: true` model,
  so search works automatically. If you switch `RESEARCH_MODEL` to a model
  without `tool_proxy: true`, the researcher will fall back to memory.
- **Anthropic backend has no search.** The Anthropic path doesn't have a
  tool proxy in front of it, so `LLM_BACKEND=anthropic` produces
  memory-only research today. Stick with the local backend if search
  matters.
- **Synthesis runs even on failure.** If no sprint passed verification, a
  synthesis is still written but flagged provisional, so partial work is
  never thrown away.
- **Verification rigor depends on the synthesis model.** Default is the
  `coder` alias on the local router. If verifier scores feel too lenient,
  raise `SCORE_THRESHOLD` or swap to a stronger synthesis model.
