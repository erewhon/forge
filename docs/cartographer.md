# forge map — the repo cartographer

Turns large codebases into precomputed structure so interactive questions read small maps and a
handful of targeted files instead of slogging through whole repos at question time. Built for the
local-only constraint: **inference goes exclusively to the on-prem router** (a non-local URL is
refused at startup — there is no cloud tier), and derived artifacts are treated as derived from
the source they map — they stay on the home network (an `output_root` inside a work tree with a
public remote is refused).

## Pipeline

```
forge map sync        rsync-over-SSH mirrors, one per (host, path) — divergent checkouts of the
                      same repo on different VMs stay independent targets
forge map structure   no-LLM tree-sitter pass → out/<key>/{map.md,index.json}
forge map summarize   guarded LLM stage: file summaries (sweep tier) → module rollups →
                      architecture doc (synthesis tier), all content-addressed in cache/
forge map render      assemble out/<key>/{modules/*.md,architecture.md} from the cache
forge map run         all of the above per target, failure-isolated, + report.md
```

The cache is keyed by git blob hash (content sha256 for untracked files), so unchanged files are
never re-summarized and two checkouts of the same repo share almost the whole cache. Touch one
file and exactly that file, its module rollup, and the architecture doc recompute.

## Config — `~/.config/forge/map.toml`

```toml
output_root = "/home/me/contract-maps"   # keep on the home network; never inside a public repo
concurrency = 4                          # sweep fan-out
max_file_bytes = 262144                  # larger files are listed as skipped, never silently

[[targets]]
name = "food-api"
host = "work-vm.m.example"               # omit for a local path
path = "/home/me/src/food-api"
ignore_globs = ["generated"]             # on top of the built-in ignores (node_modules, .git…)
```

Model seats and the router endpoint come from the standard forge env layers
(`CARTOGRAPHER_OPENAI_BASE_URL`, `CARTOGRAPHER_OPENAI_API_KEY`, `CARTOGRAPHER_SWEEP_MODEL`,
`CARTOGRAPHER_SYNTHESIS_MODEL`) — repo `.env`, `~/.config/forge/.env`, or `config.toml`'s
`router_url`/`api_key_env` fan-out. Token budgets default high (2048/3072/4096) because the local
seats are reasoning models: too small a budget returns `finish_reason=length` with zero content.

Rollup and architecture prompts are capped at `CARTOGRAPHER_SYNTHESIS_PROMPT_BUDGET_CHARS`
(default 48,000 — the same envelope as `max_prompt_chars`, which the sweep seat serves
reliably). A module whose file summaries exceed the cap is reduced map-reduce style —
batches → interim digests on the *sweep* seat → final rollup on the synthesis seat — so
module size never overflows a seat's context window; you only pay the extra condense calls
on modules that need them. Don't raise the budget without measuring both seats: the
synthesis seat has been observed to 502 on ~24k-token prefills.

Requires the `map` extra: `uv sync --extra map` (tree-sitter + language pack).

## Cost expectations

The **first** sweep of a big repo is an overnight batch job — prefill dominates whole-file
prompts on the local seats, and synthesis-tier decode may be slow depending on the seat.
That is what the nightly timer is for. Steady-state incremental runs finish in minutes; a run
over unchanged input makes **zero** LLM calls. `report.md` at the output root names everything
skipped, failed (auto-retried next run), or stale — no silent caps.

## Nightly timer

```sh
ln -s ~/Projects/erewhon/forge/forge/cartographer/systemd/forge-map.service \
      ~/Projects/erewhon/forge/forge/cartographer/systemd/forge-map.timer \
      ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now forge-map.timer
```

Overrides for the unit (router URL, seats) go in `~/.config/forge/map.env`.

## Using the output

Point an agent at `out/<key>/architecture.md` + `map.md` as orientation context, then let it read
specific files. Spot-check that earned its keep during development: "where does the summary cache
decide what to invalidate?" is answerable from `architecture.md` + `modules/forge.md` + one read
of `cache.py` — no repo slog.
