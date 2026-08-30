"""`forge book revise`: evidence from a project dir, edit resolution, comment-preserving apply."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from forge.book_researcher import main as main_mod
from forge.book_researcher import revise as revise_mod
from forge.book_researcher.models import (
    BookConfig,
    OutlineEdit,
    ResearchFinding,
    RevisionProposal,
    SprintContract,
    SprintFindings,
    VerificationResult,
    VerificationScores,
)
from forge.book_researcher.revise import (
    apply_edits,
    gather_evidence,
    render_evidence,
    resolve_edits,
    unified_diff,
)

OUTLINE = """\
# Header comment that must survive a revision.
title: "Federal Corruption"
description: >
  Documented conflicts of interest, separating adjudicated from alleged.

chapters:
  - number: 1
    title: "Scope"
    description: "The framing chapter."   # trailing comment
    research_questions:
      - "What statutes govern presidential conflicts of interest, per the U.S. Code?"
      - "What is the strongest argument that the ethics framework is overstated, and who makes it?"

  - number: 2
    title: "People v. Trump"
    description: "The civil fraud case."
    research_questions:
      - "What findings of fact did Justice Engoron make about asset valuation, per the decision?"
      - "Has the New York Court of Appeals acted on People v. Trump? Report ONLY what a source
        states."
"""


def _book() -> BookConfig:
    return BookConfig.model_validate(yaml.safe_load(OUTLINE))


def _scores(overall: int, **dims: int) -> VerificationScores:
    base = dict(
        source_diversity=overall,
        claim_verification=overall,
        counter_narrative=overall,
        depth=overall,
        actionability=overall,
    )
    base.update(dims)
    return VerificationScores(overall=overall, **base)


def _project(tmp_path: Path) -> Path:
    sprints = tmp_path / "sprints"
    knowledge = tmp_path / "knowledge" / "chapter-02"
    sprints.mkdir(parents=True)
    knowledge.mkdir(parents=True)
    c1 = SprintContract(
        sprint_id="001",
        chapter=2,
        questions=["What findings of fact did Engoron make?"],
        success_criteria=[],
        priority="high",
    )
    r1 = VerificationResult(
        sprint_id="001",
        scores=_scores(3, claim_verification=2),
        passed=False,
        feedback="bad",
        follow_up_questions=["The question references a July 2023 judgment; there was none."],
    )
    c2 = SprintContract(
        sprint_id="002",
        chapter=2,
        questions=["What findings of fact did Engoron make, per the decision?"],
        success_criteria=[],
        priority="high",
    )
    r2 = VerificationResult(
        sprint_id="002",
        scores=_scores(8),
        passed=True,
        feedback="ok",
        follow_up_questions=[],
    )
    for c, r in ((c1, r1), (c2, r2)):
        (sprints / f"sprint-{c.sprint_id}.json").write_text(c.model_dump_json())
        (sprints / f"sprint-{c.sprint_id}-review.json").write_text(r.model_dump_json())
    (sprints / "sprint-003.json").write_text("{garbage")  # tolerated
    findings = SprintFindings(
        sprint_id="002",
        chapter=2,
        findings=[
            ResearchFinding(
                question="What findings of fact did Engoron make, per the decision?",
                answer="Findings...",
                sources=["x"],
                confidence="high",
            ),
            ResearchFinding(
                question="Has the Court of Appeals acted?",
                answer="Research failed: empty response",
                sources=[],
                confidence="low",
            ),
        ],
    )
    (knowledge / "sprint-002.json").write_text(findings.model_dump_json())
    return tmp_path


def test_gather_evidence_reads_reviews_coverage_and_failures(tmp_path):
    ev = gather_evidence(_book(), _project(tmp_path))
    assert ev.total_sprints == 2
    ch1, ch2 = ev.chapters[1], ev.chapters[2]
    assert ch1.attempts == 0 and ch1.reviews == []
    assert ch2.attempts == 2 and ch2.passes == 1 and ch2.best_score == 8
    assert ch2.reviews[0].sprint_id == "002"  # most recent first
    assert ch2.reviews[1].weakest == "claim_verification"
    assert ch2.covered == [("What findings of fact did Engoron make, per the decision?", "high")]
    assert ch2.failed_questions == ["Has the Court of Appeals acted?"]
    text = render_evidence(_book(), ev)
    assert "July 2023 judgment" in text and "Research FAILED" in text and "No sprints yet" in text


def test_gather_evidence_on_empty_project(tmp_path):
    ev = gather_evidence(_book(), tmp_path)
    assert ev.total_sprints == 0 and all(c.attempts == 0 for c in ev.chapters.values())


def test_resolve_edits_fuzzy_old_and_rejections():
    book = _book()
    edits = [
        OutlineEdit(
            op="replace_question",
            chapter=2,
            old="Has the New York Court of Appeals acted on People v. Trump?",  # truncated echo
            new="Has the New York Court of Appeals acted on People v. Trump, per CourtListener?",
            reason="r",
            evidence=["001"],
        ),
        OutlineEdit(op="remove_question", chapter=2, old="Totally unrelated text", reason="r"),
        OutlineEdit(op="add_question", chapter=9, new="Q?", reason="no such chapter"),
        OutlineEdit(op="add_guidance", new="Never invent a docket number.", reason="r"),
        OutlineEdit(op="block_host", new="nycourts.gov", reason="403 every sprint"),
        OutlineEdit(op="note", reason="chapter 2 is done"),
        OutlineEdit(
            op="add_question",
            chapter=1,
            new="What statutes govern presidential conflicts of interest, per the U.S. Code and "
            "the OGE regulations, with citations?",
            reason="paraphrase of an existing question",
        ),
        OutlineEdit(op="replace_question", chapter=1, old="What statutes govern", reason="no new"),
    ]
    resolved = resolve_edits(book, edits)
    flags = [r.applicable for r in resolved]
    assert flags == [True, False, False, True, True, False, False, False]
    assert resolved[0].matched.startswith("Has the New York Court of Appeals")
    assert "does not match" in resolved[1].note
    assert "near-duplicate" in resolved[6].note


def test_add_guidance_with_chapter_becomes_chapter_guidance():
    book = _book()
    edit = OutlineEdit(op="add_guidance", chapter=2, new="Use official orders only.", reason="r")
    resolved = resolve_edits(book, [edit])
    assert resolved[0].applicable and resolved[0].edit.op == "add_chapter_guidance"
    after = BookConfig.model_validate(yaml.safe_load(apply_edits(OUTLINE, resolved)))
    assert after.chapters[1].guidance == ["Use official orders only."] and after.guidance == []


def test_apply_edits_preserves_comments_and_layout():
    book = _book()
    edits = [
        OutlineEdit(
            op="replace_question",
            chapter=2,
            old="Has the New York Court of Appeals acted on People v. Trump?",
            new="Has the New York Court of Appeals acted on People v. Trump, per CourtListener?",
            reason="r",
        ),
        OutlineEdit(
            op="remove_question", chapter=1, old="What statutes govern presidential", reason="r"
        ),
        OutlineEdit(
            op="add_question",
            chapter=1,
            new="Which rules exempt the President, per 18 U.S.C. 202?",
            reason="r",
        ),
        OutlineEdit(op="add_guidance", new="Never invent a docket number.", reason="r"),
        OutlineEdit(
            op="add_chapter_guidance",
            chapter=2,
            new="Ask for holdings, not docket ids.",
            reason="r",
        ),
        OutlineEdit(op="add_chapter_source", chapter=2, new="courtlistener.com", reason="r"),
        OutlineEdit(op="block_host", new="nycourts.gov", reason="r"),
    ]
    after = apply_edits(OUTLINE, resolve_edits(book, edits))
    assert after.startswith("# Header comment that must survive a revision.")
    assert "# trailing comment" in after
    new_book = BookConfig.model_validate(yaml.safe_load(after))
    assert new_book.guidance == ["Never invent a docket number."]
    assert new_book.sources.blocked == ["nycourts.gov"]
    ch1, ch2 = new_book.chapters
    assert ch1.research_questions == [
        "What is the strongest argument that the ethics framework is overstated, and who makes it?",
        "Which rules exempt the President, per 18 U.S.C. 202?",
    ]
    assert ch2.research_questions[1].endswith("per CourtListener?")
    assert ch2.guidance == ["Ask for holdings, not docket ids."]
    assert ch2.sources == ["courtlistener.com"]
    diff = unified_diff(OUTLINE, after, "book.yaml")
    assert "+guidance:" in diff and '-      - "What statutes govern' in diff
    # New top-level keys go where a human would put them, not at the end of the file.
    keys = list(yaml.safe_load(after))
    assert keys.index("description") < keys.index("guidance") < keys.index("sources")
    assert keys.index("sources") < keys.index("chapters")
    ch2_keys = list(yaml.safe_load(after)["chapters"][1])
    assert ch2_keys.index("description") < ch2_keys.index("sources")
    assert ch2_keys.index("sources") < ch2_keys.index("guidance")
    assert ch2_keys.index("guidance") < ch2_keys.index("research_questions")
    # Untouched long questions are not re-wrapped, and new strings are quoted like the rest.
    assert "What findings of fact did Justice Engoron make about asset valuation, per the" in after
    assert '- "Which rules exempt the President, per 18 U.S.C. 202?"' in after
    assert '- "Never invent a docket number."' in after


def _mock_propose(monkeypatch, proposal: RevisionProposal | None):
    def fake(book, ev, *, pool, max_tokens, timeout):
        if proposal is None:
            raise RuntimeError("pool exhausted")
        return proposal

    monkeypatch.setattr(revise_mod, "propose_revision", fake)
    monkeypatch.setattr("forge.book_researcher.outline.outline_pool", lambda models=None: object())


@pytest.fixture
def cfg(tmp_path, monkeypatch) -> Path:
    project = _project(tmp_path / "project")
    monkeypatch.setattr(main_mod.settings, "project_dir", project)
    path = tmp_path / "book.yaml"
    path.write_text(OUTLINE)
    return path


def test_revise_report_needs_no_model(cfg, capsys):
    assert main_mod.main(["revise", str(cfg), "--report"]) == 0
    out = capsys.readouterr().out
    assert "Revision evidence" in out and "July 2023" in out


def test_revise_writes_proposal_not_config(cfg, capsys, monkeypatch):
    proposal = RevisionProposal(
        summary="Rewrite the Court of Appeals question.",
        edits=[
            OutlineEdit(
                op="replace_question",
                chapter=2,
                old="Has the New York Court of Appeals acted on People v. Trump?",
                new="Has the New York Court of Appeals acted on People v. Trump, per "
                "CourtListener?",
                reason="Instruction text belongs in guidance.",
                evidence=["001"],
            ),
            OutlineEdit(op="note", reason="Chapter 2 passed in sprint 002."),
        ],
    )
    _mock_propose(monkeypatch, proposal)
    assert main_mod.main(["revise", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "APPLY replace_question" in out and "SKIP  note" in out
    assert cfg.read_text() == OUTLINE  # untouched
    proposed = cfg.with_name("book.proposed.yaml")
    assert "per CourtListener?" in proposed.read_text()
    assert cfg.with_name("book.revision.md").exists()
    saved = list((main_mod.settings.project_dir / "revisions").glob("revision-*.json"))
    assert len(saved) == 1 and json.loads(saved[0].read_text())["summary"].startswith("Rewrite")


def test_revise_apply_writes_config_when_lint_clean(cfg, monkeypatch):
    proposal = RevisionProposal(
        summary="s",
        edits=[OutlineEdit(op="add_guidance", new="Never invent a docket number.", reason="r")],
    )
    _mock_propose(monkeypatch, proposal)
    assert main_mod.main(["revise", str(cfg), "--apply"]) == 0
    assert "Never invent a docket number." in cfg.read_text()
    assert cfg.read_text().startswith("# Header comment")


def test_revise_apply_refuses_lint_errors(cfg, monkeypatch, capsys):
    # Removing every question from chapter 1 leaves it with none — a lint error.
    proposal = RevisionProposal(
        summary="s",
        edits=[
            OutlineEdit(
                op="remove_question",
                chapter=1,
                old="What statutes govern presidential conflicts of interest, per the U.S. Code?",
                reason="r",
            ),
            OutlineEdit(
                op="remove_question",
                chapter=1,
                old="What is the strongest argument that the ethics framework is overstated, "
                "and who makes it?",
                reason="r",
            ),
        ],
    )
    _mock_propose(monkeypatch, proposal)
    assert main_mod.main(["revise", str(cfg), "--apply"]) == 1
    assert cfg.read_text() == OUTLINE
    assert "not applied" in capsys.readouterr().err


def test_revise_without_sprints_explains(cfg, monkeypatch, capsys):
    monkeypatch.setattr(main_mod.settings, "project_dir", cfg.parent / "empty")
    assert main_mod.main(["revise", str(cfg)]) == 1
    assert "No sprints found" in capsys.readouterr().out


def test_revise_model_failure_is_reported(cfg, monkeypatch, capsys):
    _mock_propose(monkeypatch, None)
    assert main_mod.main(["revise", str(cfg)]) == 1
    assert "pool exhausted" in capsys.readouterr().err


def test_propose_revision_raises_with_raw_on_exhaustion(monkeypatch, tmp_path):
    def fake(**kwargs):
        return SimpleNamespace(value=None, error="all seats failed", ok=False, raw="not json")

    monkeypatch.setattr(revise_mod, "structured", fake)
    ev = gather_evidence(_book(), tmp_path)
    with pytest.raises(RuntimeError, match="all seats failed"):
        revise_mod.propose_revision(_book(), ev, pool=object())  # type: ignore[arg-type]
