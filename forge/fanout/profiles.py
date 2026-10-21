"""Profiles: what kind of sweep this is.

A profile bundles everything that differs between auditing code, reading a stack of papers, and
checking a bibliography: the schema wrapper every seat gets, the judge's kickoff text, the
read-only tool allowlists, and the default seats. The runner does not change between profiles.

Built-ins: ``code`` (today's behaviour), ``papers``, ``refs``, and ``raw`` — no wrapper at all:
the seats get the prompt exactly as written. A TOML file passed to
``--profile`` overrides any field of the ``base`` it names (default ``code``), so a one-off sweep
can bring its own templates without a code change.

Template placeholders (``str.format``; write ``{{`` for a literal brace):
  sweep:  ``{target}`` ``{prompt}``
  judge:  ``{target}`` ``{n}`` ``{prompt_path}`` ``{user_prompt_path}`` ``{findings_list}``
          ``{failed_block}`` ``{manifest_path}`` ``{collated_path}``
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ValidationError

from forge.fanout.config import settings


class Profile(BaseModel):
    name: str
    description: str
    target_noun: str  # how the sweep talks about the directory it runs in
    sweep_template: str
    judge_template: str
    claude_allowed_tools: list[str]
    claude_disallowed_tools: list[str]
    pi_tools: list[str]
    default_models: list[str]
    # Replaces the seat CLI's own default system prompt (claude/pi ``--system-prompt``). None keeps
    # each tool's default. ``--system-prompt`` on the command line overrides this per sweep.
    system_prompt: str | None = None


# ---------------------------------------------------------------------------------------- code

_CODE_SWEEP = """\
You are one of several independent analysts examining the repository at {target}. Other analysts
are answering the same question in parallel; a reviewer will later verify every claim against
the code and collate the results, so precision and evidence matter more than volume.

Rules:
- Read-only. Do not create, edit, or delete files. Explore with the read/search tools you have.
- Every finding must cite a concrete location (path:line) and quote or paraphrase the evidence.
- Do not pad. If you checked something and it is fine, say so under "Not found".
- Rate your own confidence honestly; a low-confidence lead is useful, a confident guess is not.

Question:

{prompt}

When you are done exploring, reply with your findings as markdown in exactly this shape. The
reply itself is your deliverable; there is no file to write.

# Findings

## F1: <short title>
- **Where:** <path:line, or several>
- **Severity:** critical | high | medium | low
- **Confidence:** high | medium | low
- **Evidence:** <what the code does, quoting the relevant lines>
- **Why it matters:** <the consequence>
- **Suggested fix:** <one or two sentences; optional>

## F2: ...

# Not found
<what you checked and ruled out, one line each>

# Summary
<three to five sentences>
"""

_CODE_JUDGE = """\
You are the judge for a multi-model analysis sweep of the repository at {target}.

{n} independent seats were given the same question (the wrapped prompt is at {prompt_path};
the question as I wrote it is at {user_prompt_path}). Their findings are in:

{findings_list}
{failed_block}
Sweep manifest (models, timings, exit codes): {manifest_path}

Your job:
1. Read the question and every findings file in full.
2. For each distinct claim, verify it against the code here: open the cited lines, follow the
   call sites, and decide. Label each one CONFIRMED, FALSE, UNVERIFIABLE, or DUPLICATE-of-<id>.
   Note which seats raised it and whether they agree on severity.
3. Show me a compact table (claim, label, severity, seats, one-line reason) before writing
   anything, and flag anything you could not settle so I can weigh in.
4. Then write {collated_path}: confirmed findings grouped by severity with seat attribution and
   your verified evidence, a short list of rejected claims with the reason, and gaps — things the
   question asked about that no seat covered well.

Do not edit anything else in the repository. Start with step 1.
"""

# -------------------------------------------------------------------------------------- papers

_PAPERS_SWEEP = """\
You are one of several independent readers working through the documents under {target}
(papers, notes, transcripts — whatever is there). Other readers are answering the same question
in parallel; a reviewer will later check every claim against the source text and collate the
results, so fidelity to the sources matters more than volume.

Rules:
- Read-only. Do not create, edit, or delete files. Use the read/search tools you have.
- Every finding cites its source (file, and page or section) and quotes the passage it rests on.
- Keep what the source says separate from what you infer from it, and say which is which.
- Do not pad. If the documents are silent on part of the question, say so under "Not found".
- Rate your own confidence honestly.

Question:

{prompt}

When you are done reading, reply with your findings as markdown in exactly this shape. The reply
itself is your deliverable; there is no file to write.

# Findings

## F1: <short title>
- **Source:** <file, page or section>
- **Quote:** <the passage, verbatim, trimmed to what matters>
- **Importance:** high | medium | low
- **Confidence:** high | medium | low
- **Reading:** <what it says about the question; mark inference as inference>
- **Follow-up:** <what to read or check next; optional>

## F2: ...

# Not found
<parts of the question the documents do not answer, one line each>

# Summary
<three to five sentences>
"""

_PAPERS_JUDGE = """\
You are the judge for a multi-model reading sweep over the documents under {target}.

{n} independent readers were given the same question (the wrapped prompt is at {prompt_path};
the question as I wrote it is at {user_prompt_path}). Their findings are in:

{findings_list}
{failed_block}
Sweep manifest (models, timings, exit codes): {manifest_path}

Your job:
1. Read the question and every findings file in full.
2. For each distinct claim, verify it against the source: open the cited file, find the quoted
   passage, and check that the reading is what the passage says. Label each one CONFIRMED,
   MISREAD (the passage does not support the reading), UNVERIFIABLE (source or passage not
   found), or DUPLICATE-of-<id>. Note which readers raised it and whether they agree.
3. Show me a compact table (claim, label, importance, readers, one-line reason) before writing
   anything, and flag anything you could not settle so I can weigh in.
4. Then write {collated_path}: confirmed findings grouped by theme with source attribution and
   the verified quotes, a short list of misreadings with the reason, and gaps — parts of the
   question no reader covered well.

Do not edit anything else under {target}. Start with step 1.
"""

# ---------------------------------------------------------------------------------------- refs

_REFS_SWEEP = """\
You are one of several independent fact-checkers examining the references cited in the documents
under {target}. Other checkers are working the same question in parallel; a reviewer will later
re-check the disputed ones and collate the results, so say exactly what you verified and how.

Rules:
- Read-only on the documents. Do not create, edit, or delete files.
- If you have web tools, use them to resolve references (DOI, URL, publisher or venue page) and
  to compare what the cited work actually says with the claim it is cited for. If you have no
  web tools, check internal consistency only (authors, year, title, venue agree across the
  document; the reference exists in the bibliography; the claim matches the citation's
  summary) and say so.
- Never guess that a reference exists. "Could not resolve" is a valid answer; "looks fine" is not.
- Rate your own confidence honestly.

Question:

{prompt}

When you are done, reply with your findings as markdown in exactly this shape. The reply itself
is your deliverable; there is no file to write.

# References

## R1: <the reference, short form, as it appears>
- **Cited in:** <file, location> for the claim: "<the claim, short>"
- **Resolves:** yes | no | partly — <what you found: DOI/URL, title, authors, venue, year>
- **Supports claim:** yes | no | partly | not checked — <one line why>
- **Verdict:** verified | unresolved | mismatched | suspicious
- **Confidence:** high | medium | low
- **Evidence:** <what you fetched or compared; URLs you opened>

## R2: ...

# Not checked
<references you did not get to, one line each, and why>

# Summary
<three to five sentences: how many checked, how many problems, the worst one>
"""

_REFS_JUDGE = """\
You are the judge for a multi-model reference check over the documents under {target}.

{n} independent checkers were given the same question (the wrapped prompt is at {prompt_path};
the question as I wrote it is at {user_prompt_path}). Their findings are in:

{findings_list}
{failed_block}
Sweep manifest (models, timings, exit codes): {manifest_path}

Your job:
1. Read the question and every findings file in full.
2. Build one list of distinct references. For each, take the checkers' verdicts and re-check
   every one that is unresolved, mismatched, suspicious, or where checkers disagree: fetch the
   DOI/URL or search for the work, open the cited location in the document, and decide. Label
   each VERIFIED, UNRESOLVED, MISMATCHED (exists but does not support the claim), SUSPICIOUS
   (likely fabricated or badly garbled), or DUPLICATE-of-<id>. Note which checkers agreed.
3. Show me a compact table (reference, label, checkers, one-line reason) before writing
   anything, and flag what you could not settle so I can weigh in.
4. Then write {collated_path}: problems first (mismatched and suspicious, with the evidence),
   then unresolved, then the verified list, then references nobody checked.

Do not edit anything else under {target}. Start with step 1.
"""

_RAW_JUDGE = """\
You are the judge for a multi-model sweep run from {target}.

{n} independent seats were given the same prompt, verbatim, with no wrapper (it is at
{user_prompt_path}; {prompt_path} holds the identical text). Their answers are in:

{findings_list}
{failed_block}
Sweep manifest (models, timings, exit codes): {manifest_path}

Your job:
1. Read the prompt and every answer in full.
2. Pull out the distinct claims. Verify each one where you can — against the files here, with
   your tools, or against what you know — and label it CONFIRMED, FALSE, UNVERIFIABLE, or
   DUPLICATE-of-<id>. Note which seats made it and whether they agree.
3. Show me a compact table (claim, label, seats, one-line reason) before writing anything, and
   flag what you could not settle so I can weigh in.
4. Then write {collated_path}: the collated answer with seat attribution, the claims you
   rejected and why, and where the seats disagree.

Do not edit anything else under {target}. Start with step 1.
"""

_READ_ONLY_CLAUDE = ["Read", "Grep", "Glob", "Bash(rg:*)"]
_NO_WRITE_CLAUDE = ["Edit", "Write", "NotebookEdit"]
_READ_ONLY_PI = ["read", "grep", "find", "ls"]


def _code_profile() -> Profile:
    # The code profile is the one the FANOUT_* settings describe, so env overrides still apply.
    return Profile(
        name="code",
        description="Audit a repository: findings cite path:line; judge verifies against the code.",
        target_noun="repository",
        sweep_template=_CODE_SWEEP,
        judge_template=_CODE_JUDGE,
        claude_allowed_tools=list(settings.claude_allowed_tools),
        claude_disallowed_tools=list(settings.claude_disallowed_tools),
        pi_tools=list(settings.pi_tools),
        default_models=list(settings.default_models),
    )


def _papers_profile() -> Profile:
    return Profile(
        name="papers",
        description="Read a folder of papers/notes: findings quote sources; judge checks the text.",
        target_noun="document set",
        sweep_template=_PAPERS_SWEEP,
        judge_template=_PAPERS_JUDGE,
        claude_allowed_tools=list(_READ_ONLY_CLAUDE),
        claude_disallowed_tools=list(_NO_WRITE_CLAUDE),
        pi_tools=list(_READ_ONLY_PI),
        default_models=["claude:fable", "pi:gemma4-26b"],
    )


def _refs_profile() -> Profile:
    return Profile(
        name="refs",
        description=(
            "Check references: resolve citations and whether they support the claim. Claude seats "
            "get WebFetch/WebSearch; pi seats are text-only (internal consistency) unless "
            "--pi-tools adds bash."
        ),
        target_noun="document set",
        sweep_template=_REFS_SWEEP,
        judge_template=_REFS_JUDGE,
        claude_allowed_tools=[*_READ_ONLY_CLAUDE, "WebFetch", "WebSearch"],
        claude_disallowed_tools=list(_NO_WRITE_CLAUDE),
        pi_tools=list(_READ_ONLY_PI),
        default_models=["claude:fable", "pi:gemma4-26b"],
    )


def _raw_profile() -> Profile:
    return Profile(
        name="raw",
        description=(
            "No wrapper: seats get your prompt verbatim (bring your own output format); the judge "
            "collates whatever came back. --system-prompt replaces the tools' own system prompt."
        ),
        target_noun="directory",
        sweep_template="{prompt}",
        judge_template=_RAW_JUDGE,
        claude_allowed_tools=list(_READ_ONLY_CLAUDE),
        claude_disallowed_tools=list(_NO_WRITE_CLAUDE),
        pi_tools=list(_READ_ONLY_PI),
        default_models=["claude:fable", "pi:gemma4-26b"],
    )


BUILTIN: dict[str, object] = {
    "code": _code_profile,
    "papers": _papers_profile,
    "refs": _refs_profile,
    "raw": _raw_profile,
}


def builtin_names() -> list[str]:
    return list(BUILTIN)


def load_profile(ref: str | None) -> Profile:
    """Resolve ``--profile``: a built-in name, or a TOML file layered over its ``base``."""
    if not ref:
        return _code_profile()
    if ref in BUILTIN:
        return BUILTIN[ref]()  # type: ignore[operator]
    path = Path(ref).expanduser()
    if not path.is_file():
        raise SystemExit(
            f"error: unknown profile {ref!r}; built-ins are {', '.join(builtin_names())}, "
            "or pass a path to a TOML profile file"
        )
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"error: {path}: not valid TOML: {e}") from None
    base_name = str(data.pop("base", "code"))
    if base_name not in BUILTIN:
        raise SystemExit(f"error: {path}: base must be one of {', '.join(builtin_names())}")
    base = BUILTIN[base_name]()  # type: ignore[operator]
    merged = {**base.model_dump(), **data}
    merged.setdefault("name", path.stem)
    if "name" not in data:
        merged["name"] = path.stem
    try:
        return Profile.model_validate(merged)
    except ValidationError as e:
        raise SystemExit(f"error: {path}: bad profile: {e}") from None
