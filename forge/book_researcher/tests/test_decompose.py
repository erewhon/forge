"""`forge book decompose`: recon digest, the human framing gate, lint-validated decomposition."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from forge.book_researcher import decompose as dec
from forge.book_researcher import main as main_mod
from forge.book_researcher.models import (
    BookConfig,
    ChapterOutline,
    ChapterSketch,
    HostVerdict,
    OutlineFraming,
    Reachability,
    SlicingOption,
)
from forge.general_researcher.models import ResearchFinding as GRFinding
from forge.general_researcher.models import SprintFindings as GRSprint
from forge.general_researcher.models import VerificationResult as GRReview
from forge.general_researcher.models import VerificationScores as GRScores


def _recon_dir(root: Path, slug: str = "corruption-trump") -> Path:
    d = root / slug
    (d / "findings").mkdir(parents=True)
    (d / "sprints").mkdir()
    (d / "topic.yaml").write_text(
        yaml.safe_dump(
            {
                "question": "What are ALL the corrupt dealings?",
                "context": "Month by month.",
                "sub_questions": ["Who profited?"],
                "slug": slug,
            }
        )
    )
    (d / "synthesis.md").write_text("# Synthesis\n\nCivil fraud was established...")
    sprint = GRSprint(
        sprint_id="001",
        findings=[
            GRFinding(
                question="What did the NY AG case find?",
                answer="Fraud.",
                sources=["JURIST (https://www.jurist.org/news/x)", "AP (https://apnews.com/a)"],
                confidence="medium",
            ),
            GRFinding(
                question="Dead", answer="Research failed: empty", sources=[], confidence="low"
            ),
        ],
    )
    (d / "findings" / "sprint-001.json").write_text(sprint.model_dump_json())
    review = GRReview(
        sprint_id="001",
        scores=GRScores(
            source_diversity=4,
            claim_verification=3,
            counter_narrative=5,
            depth=5,
            actionability=4,
            overall=5,
        ),
        passed=False,
        feedback="advocacy summaries",
        follow_up_questions=["Cite the primary filing, not CREW's summary."],
    )
    (d / "sprints" / "sprint-001-review.json").write_text(review.model_dump_json())
    return d


def test_load_recon_digest(tmp_path):
    _recon_dir(tmp_path)
    r = dec.load_recon("corruption-trump", root=tmp_path)
    assert r.question.startswith("What are ALL") and r.sub_questions == ["Who profited?"]
    assert r.best_score == 5 and r.sprint_count == 1
    assert r.findings == [
        ("What did the NY AG case find?", "medium", ["apnews.com", "www.jurist.org"])
    ]
    assert r.follow_ups == ["Cite the primary filing, not CREW's summary."]
    text = dec.render_recon(r)
    assert "Verifier challenges" in text and "Synthesis" in text and "apnews.com" in text


def test_load_recon_missing_lists_available(tmp_path):
    _recon_dir(tmp_path, "other-topic")
    with pytest.raises(FileNotFoundError, match="other-topic"):
        dec.load_recon("nope", root=tmp_path)


def _framing(approved: bool = False, n: int = 4) -> OutlineFraming:
    return OutlineFraming(
        thesis="Corruption, separated into adjudicated and alleged.",
        slicing_options=[SlicingOption(name="mechanism", principle="p", tradeoffs="t")],
        recommended="mechanism",
        chapter_sketch=[
            ChapterSketch(title=f"Ch {i}", one_line="x", primary_sources=["courtlistener.com"])
            for i in range(1, n + 1)
        ],
        guidance=["Distinguish adjudicated from alleged."],
        approved=approved,
    )


def _mock_structured(monkeypatch, value, error=None, raw=""):
    calls: list[dict] = []

    def fake(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(value=value, error=error, ok=value is not None, raw=raw)

    monkeypatch.setattr(dec, "structured", fake)
    return calls


def test_propose_framing_forces_approved_false(monkeypatch, tmp_path):
    _recon_dir(tmp_path)
    recon = dec.load_recon("corruption-trump", root=tmp_path)
    calls = _mock_structured(monkeypatch, _framing(approved=True))
    framing = dec.propose_framing(recon, pool=object())  # type: ignore[arg-type]
    assert framing.approved is False
    assert "Verifier challenges" in calls[0]["user"]
    assert "approved" not in calls[0]["system"].split("Return ONLY JSON")[1].lower().replace(
        'do not include an "approved" field', ""
    )


def test_propose_framing_exhaustion_raises_with_raw(monkeypatch, tmp_path):
    _recon_dir(tmp_path)
    recon = dec.load_recon("corruption-trump", root=tmp_path)
    _mock_structured(monkeypatch, None, error="exhausted", raw="garbage")
    with pytest.raises(dec.DecomposeError, match="exhausted") as ei:
        dec.propose_framing(recon, pool=object())  # type: ignore[arg-type]
    assert ei.value.raw == "garbage"


def test_persist_refuses_overwrite_then_approve_flips_gate(tmp_path):
    path = tmp_path / "outline-framing.json"
    dec.persist_framing(_framing(), path)
    assert path.with_suffix(".md").read_text().startswith("# Outline framing")
    assert "NOT APPROVED" in path.with_suffix(".md").read_text()
    with pytest.raises(dec.FramingExistsError):
        dec.persist_framing(_framing(), path)
    approved = dec.approve_framing(path)
    assert approved.approved and dec.load_framing(path).approved
    assert "\nAPPROVED" in path.with_suffix(".md").read_text()


def test_human_framing_is_approved_by_construction(tmp_path):
    path = tmp_path / "my-framing.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "thesis": "t",
                "recommended": "by mechanism",
                "chapter_sketch": [
                    {"title": "A", "one_line": "a"},
                    {"title": "B", "one_line": "b"},
                ],
            }
        )
    )
    assert dec.load_human_framing(path).approved is True


def _good_book(n: int = 4) -> BookConfig:
    return BookConfig(
        title="Federal Corruption",
        description="A thesis long enough to satisfy the description rule in the linter.",
        guidance=["Model rule."],
        chapters=[
            ChapterOutline(
                number=i,
                title=f"Ch {i}",
                description="What this chapter establishes, in a full sentence.",
                sources=["courtlistener.com"],
                research_questions=[
                    f"What did Justice Engoron find about valuation item {i}, per the decision "
                    "text?",
                    f"What is the strongest argument that valuation item {i} was ordinary "
                    "practice, and who advances it?",
                ],
            )
            for i in range(1, n + 1)
        ],
    )


def test_decompose_refuses_unapproved_framing(monkeypatch, tmp_path):
    _recon_dir(tmp_path)
    recon = dec.load_recon("corruption-trump", root=tmp_path)
    _mock_structured(monkeypatch, _good_book())
    with pytest.raises(dec.FramingNotApprovedError):
        dec.decompose(_framing(approved=False), recon, pool=object())  # type: ignore[arg-type]


def test_decompose_merges_framing_guidance_and_probe(monkeypatch, tmp_path):
    _recon_dir(tmp_path)
    recon = dec.load_recon("corruption-trump", root=tmp_path)
    calls = _mock_structured(monkeypatch, _good_book())
    reach = Reachability(
        probed_at="now",
        hosts={
            "gao.gov": HostVerdict(host="gao.gov", url="https://gao.gov/", state="blocked"),
            "fec.gov": HostVerdict(host="fec.gov", url="https://fec.gov/", state="reachable"),
        },
    )
    book = dec.decompose(
        _framing(approved=True),
        recon,
        pool=object(),
        reachability=reach,
        chapter_count=4,  # type: ignore[arg-type]
    )
    assert book.guidance == ["Distinguish adjudicated from alleged.", "Model rule."]
    assert book.sources.blocked == ["gao.gov"] and book.sources.reachable == ["fec.gov"]
    assert calls[0]["predicate"](_good_book())
    assert "Target chapter count: 4" in calls[0]["user"]
    assert "NOT reachable" in calls[0]["user"] and "gao.gov" in calls[0]["user"]


def test_shape_predicate_rejects_lint_errors_and_missing_counter_narrative():
    from forge.book_researcher.models import SourcePolicy

    ok = _good_book()
    assert dec._shape_ok(ok, SourcePolicy())
    too_few = _good_book(2)
    assert not dec._shape_ok(too_few, SourcePolicy())
    no_counter = _good_book()
    no_counter.chapters[0].research_questions = [
        "What did Justice Engoron find about valuation, per the decision text?",
        "What did the First Department hold in 2024, per the opinion?",
    ]
    assert not dec._shape_ok(no_counter, SourcePolicy())
    dup = _good_book()
    dup.chapters[1].number = 1
    assert not dec._shape_ok(dup, SourcePolicy())


def test_write_decomposed_has_provenance_and_refuses_overwrite(tmp_path):
    recon = SimpleNamespace(slug="s", sprint_count=5, best_score=5)
    out = tmp_path / "book.yaml"
    dec.write_decomposed(_good_book(), out, recon, _framing(True), force=False)  # type: ignore[arg-type]
    text = out.read_text()
    assert text.startswith("# Research outline derived by `forge book decompose s`")
    reloaded = BookConfig.model_validate(yaml.safe_load(text))
    assert reloaded == _good_book()
    with pytest.raises(FileExistsError):
        dec.write_decomposed(_good_book(), out, recon, _framing(True), force=False)  # type: ignore[arg-type]


@pytest.fixture
def env(tmp_path, monkeypatch):
    research_root = tmp_path / "research"
    project = tmp_path / "project"
    project.mkdir()
    _recon_dir(research_root)
    monkeypatch.setenv("GENERAL_RESEARCHER_PROJECT_DIR", str(research_root))
    monkeypatch.setattr(main_mod.settings, "project_dir", project)
    monkeypatch.setattr("forge.book_researcher.outline.outline_pool", lambda models=None: object())
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_decompose_subcommand_two_step_gate(env, monkeypatch, capsys):
    # Step 1: framing is proposed and stored, not approved; nothing decomposed.
    monkeypatch.setattr(dec, "propose_framing", lambda recon, **kw: _framing())
    assert main_mod.main(["decompose", "corruption-trump"]) == 0
    out = capsys.readouterr().out
    assert "--approve" in out
    assert main_mod.settings.framing_file.exists()
    assert not (env / "book.yaml").exists()

    # Step 2 without approval: refused.
    assert main_mod.main(["decompose", "corruption-trump"]) == 1
    assert "not approved" in capsys.readouterr().err

    # --approve flips the gate and decomposes in the same run.
    monkeypatch.setattr(dec, "decompose", lambda framing, recon, **kw: _good_book())
    assert main_mod.main(["decompose", "corruption-trump", "--approve"]) == 0
    text = (env / "book.yaml").read_text()
    assert "derived by `forge book decompose corruption-trump`" in text
    assert "Lint:" in capsys.readouterr().out
    # Refuses to clobber without --force.
    assert main_mod.main(["decompose", "corruption-trump"]) == 1


def test_decompose_subcommand_human_framing_skips_the_model(env, monkeypatch, capsys):
    framing_file = env / "mine.yaml"
    framing_file.write_text(json.dumps(_framing().model_dump(exclude={"approved"})))
    monkeypatch.setattr(dec, "decompose", lambda framing, recon, **kw: _good_book())
    assert (
        main_mod.main(
            ["decompose", "corruption-trump", "--framing", str(framing_file), "--out", "b.yaml"]
        )
        == 0
    )
    assert (env / "b.yaml").exists()
    assert "Using your framing" in capsys.readouterr().out


def test_decompose_subcommand_unknown_slug(env, capsys):
    assert main_mod.main(["decompose", "nope"]) == 1
    assert "corruption-trump" in capsys.readouterr().err  # lists what is available
