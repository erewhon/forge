# Evals: Judgment Eval Harness

Frozen gold-set grading for the coding pipeline agents. Every step measures a **production prompt**
through the same LLM call path the agents use — no mocks, no synthetic shortcuts.

## Package map

```
agents/evals/
├── __init__.py          # package marker
├── main.py              # CLI front door: ``meta evals run | baseline | compare``
├── runner.py            # Scorecard runner — model → load gold sets → render prompts → call model → grade → aggregate
├── fixtures.py          # Gold-set loader — walks case directories, validates schema_version, reads inputs
├── steps.py             # Step adapters — one per graded prompt surface; render exact production system/user messages
├── graders/             # Deterministic graders — one module per step (or step pair)
│   ├── decomposition.py # grade_decompose, grade_boundedness
│   ├── replan.py        # grade (replan)
│   ├── review.py        # grade_findings, grade_confirm
│   └── testgap.py       # grade_find, grade_skeptic
├── models.py            # Pydantic models — GoldCase, GradeResult, CaseScore, StepScore, Scorecard
├── report.py            # Scorecard rendering (markdown) and persistence (JSON + markdown)
├── config.py            # EvalsSettings (goldsets_dir, runs_dir, router URL, repeats, temperature)
├── tests/               # Unit tests for loaders, graders, runner, CLI
├── goldsets/            # (external) Gold-set fixture root — populated from distill session
└── baselines/           # Saved baseline scorecards (``.json``)
```

## Gold-set directory layout

Each gold case lives in its own directory under a step directory in the gold-set root
(default ``agents/evals/goldsets`` inside the package, overridden by ``--goldsets`` or
``EVALS_GOLDSETS_DIR``).

```
goldsets/
└── <step-name>/         # directory name = step key (e.g. ``replan``, ``decompose``)
    └── <case-id>/       # one directory per case; the name is the case_id
        ├── case.yaml    # metadata + schema version + expected block
        ├── <input-file> # referenced by inputs; arbitrary format (json, yaml, patch, md, txt)
        └── ...
```

The step directory name **is** the step key and holds any number of case directories;
``case.yaml``'s ``step`` field must match its step directory.

### case.yaml fields

| Field | Required | Type | Description |
|---|---|---|---|
| ``schema_version`` | yes | ``int`` | Schema pin. Must equal ``1`` (``SUPPORTED_SCHEMA_VERSION``). Mismatch aborts the run. |
| ``step`` | yes | ``str`` | Must match the parent directory name. |
| ``holdout`` | no | ``bool`` | Default ``false``. Holdout cases are tracked separately in the scorecard (``holdout_pass_rate``) and excluded from the main pass-rate. |
| ``inputs`` | yes | ``dict[str, str]`` | Mapping of logical input name → filename. The loader resolves each filename relative to the case directory and reads the content as text. |
| ``expected`` | yes | ``dict`` | Grader-specific expected output. Keys vary by grader — see each grader module for the schema. |
| ``notes`` | no | ``str`` | Free-form description (human reference only). |

### Schema version rule

The fixture loader enforces ``schema_version == 1`` and **rejects** any case whose version does not
match. When the schema evolves, bump ``SUPPORTED_SCHEMA_VERSION`` in ``fixtures.py`` and migrate
existing cases at that time. Cases from older schemas are never silently accepted.

### Worked example: a minimal replan case

Directory: ``goldsets/replan/validation-failure/``

```
validation-failure/
├── case.yaml
├── framing.json        # task framing passed to the replan prompt
├── tree.json           # current TaskTree leaves
├── report.json         # WaveReport with confirmed findings
├── attempts.json       # prior attempt history
└── diff.patch          # the wave diff that caused the validation failure
```

``case.yaml``:

```yaml
schema_version: 1
step: replan
holdout: false
inputs:
  framing: framing.json
  tree: tree.json
  report: report.json
  attempts: attempts.json
  diff: diff.patch
expected:
  must:
    - kind: fixup
      finding_slug: seeded-bug-login
  forbid_kinds:
    - split_subtree
  forbid_targets:
    - "Legacy auth module"
  allow_extra: false
notes: "Captured from 3a6360cb — the replan validation failure that motivated the harness."
```

This case feeds ``framing.json``, ``tree.json``, ``report.json``, ``attempts.json``, and
``diff.patch`` into the ``replan`` step adapter, which renders the exact production ``REPLAN_SYSTEM``
prompt. The raw model output is then graded by ``graders/replan.py`` against the ``expected`` block
(6 deterministic checks: envelope validity, must-actions, no-forbidden, fixup-confirmed-only,
no-extras, leaf-floors).

## The 7 step keys

Each key maps to one production prompt surface and one grader module:

| Step key | Prompt surface | Grader module | What it measures |
|---|---|---|---|
| ``replan`` | ``REPLAN_SYSTEM`` (``architect.py``) | ``graders/replan.py`` | Action choice, JSON validity, forbidden actions, leaf floors |
| ``decompose`` | ``DECOMPOSE_SYSTEM`` (``architect.py``) | ``graders/decomposition.py`` | Tree structure, deps, file naming, auto-floors, rubric DSL |
| ``boundedness`` | ``BOUNDEDNESS_SYSTEM`` (``architect.py``) | ``graders/decomposition.py`` | ``worker_shaped`` verdict, criteria fields |
| ``review-findings`` | ``FINDINGS_SYSTEM`` (``verify.py``) | ``graders/review.py`` | Precision/recall against frozen reference findings |
| ``review-confirm`` | ``CONFIRM_SYSTEM`` (``verify.py``) | ``graders/review.py`` | Real/decoy skeptic verdict accuracy |
| ``testgap-find`` | ``finder_system`` (``prompts.py``) | ``graders/testgap.py`` | Recall, cry-wolf rate, severity ordering |
| ``testgap-skeptic`` | ``SKEPTIC_BASE`` (``prompts.py``) | ``graders/testgap.py`` | Real/decoy verdict accuracy |

## How to add a gold case

1. Create a case directory under the step directory whose **name matches the step key**:

   ```bash
   mkdir -p agents/evals/goldsets/replan/my-new-case
   ```

2. Gather the input files that the step adapter needs (the adapter imports from the agent modules
   and reads from ``case.yaml``'s ``inputs`` map). Copy or generate them into the case directory.

3. Write ``case.yaml`` with the required fields (``schema_version``, ``step``, ``inputs``,
   ``expected``). See the case.yaml schema above.

4. Validate the fixture:

   ```bash
   uv run python -c "
   from pathlib import Path
   from agents.evals.fixtures import load_goldsets
   print(load_goldsets(Path('agents/evals/goldsets')))"
   ```

   Any missing file, schema mismatch, or step name mismatch raises ``EvalFixtureError``.

5. Run a scorecard to verify the grader works on the new case:

   ```bash
   meta evals run --step replan
   ```

## CLI reference

```bash
# Run all steps against the coder model, print scorecard
meta evals run

# Run only specific steps
meta evals run --step replan --step decompose

# Override the gold-set root or model
meta evals run --goldsets /path/to/goldsets --model coder

# Persist a baseline (run + save)
meta evals baseline --force

# Compare a fresh run against the baseline
meta evals compare
```

The ``compare`` output includes a delta table (baseline vs fresh pass rate per step) and flags
regressions. Holdout deltas are shown when both baseline and fresh include holdout cases.

## Environment variables

Prefixed with ``EVALS_`` (loaded via ``pydantic-settings``):

| Variable | Default | Description |
|---|---|---|
| ``EVALS_GOLDSETS_DIR`` | ``agents/evals/goldsets`` (in-package) | Root directory for gold-set fixtures |
| ``EVALS_RUNS_DIR`` | ``eval-runs/`` (repo root) | Where scorecard JSON/Markdown outputs are written |
| ``EVALS_OPENAI_BASE_URL`` | ``http://localhost:4010/v1`` | LLM router endpoint |
| ``EVALS_OPENAI_API_KEY`` | ``sk-local-router`` | Router API key |
| ``EVALS_MODEL`` | ``coder`` | Default model identifier |
| ``EVALS_REPEATS`` | ``3`` | Number of repeats per case |
| ``EVALS_TEMPERATURE`` | ``0.0`` | Temperature for determinism |
| ``EVALS_TIMEOUT`` | ``240.0`` | Per-call timeout in seconds |
| ``EVALS_MAX_TOKENS`` | ``16000`` | Max tokens per call |
