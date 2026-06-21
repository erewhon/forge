"""Tests for `meta book init` scaffolding (no network)."""

from __future__ import annotations

import pytest
import yaml

from agents.book_researcher import main as main_mod
from agents.book_researcher.models import BookConfig
from agents.book_researcher.scaffold import (
    BOOK_SKELETON,
    DEFAULT_FILENAME,
    resolve_target,
    write_skeleton,
)


def test_skeleton_is_a_valid_book_config():
    # A fresh init must be dry-runnable immediately, so the skeleton has to validate as-is.
    cfg = BookConfig.model_validate(yaml.safe_load(BOOK_SKELETON))
    assert cfg.title and cfg.description
    assert len(cfg.chapters) >= 1
    assert all(ch.research_questions for ch in cfg.chapters)


def test_write_skeleton_default_name(tmp_path):
    target = write_skeleton(str(tmp_path / DEFAULT_FILENAME))
    assert target.read_text() == BOOK_SKELETON
    assert target.name == DEFAULT_FILENAME


def test_resolve_target_appends_filename_to_directory(tmp_path):
    assert resolve_target(str(tmp_path)).name == DEFAULT_FILENAME
    assert resolve_target(str(tmp_path / "mybook.yaml")).name == "mybook.yaml"


def test_write_skeleton_refuses_overwrite_without_force(tmp_path):
    target = tmp_path / DEFAULT_FILENAME
    target.write_text("title: existing\n")
    with pytest.raises(FileExistsError):
        write_skeleton(str(target))
    assert target.read_text() == "title: existing\n"  # untouched


def test_write_skeleton_overwrites_with_force(tmp_path):
    target = tmp_path / DEFAULT_FILENAME
    target.write_text("old")
    write_skeleton(str(target), force=True)
    assert target.read_text() == BOOK_SKELETON


def test_main_init_subcommand_creates_file(tmp_path, capsys):
    target = tmp_path / "book.yaml"
    rc = main_mod.main(["init", str(target)])
    assert rc == 0
    assert target.exists()
    assert "skeleton" in capsys.readouterr().out.lower()


def test_main_init_existing_file_errors(tmp_path, capsys):
    target = tmp_path / "book.yaml"
    target.write_text("title: x\n")
    rc = main_mod.main(["init", str(target)])
    assert rc == 1
    assert "already exists" in capsys.readouterr().err
    # --force succeeds
    assert main_mod.main(["init", str(target), "--force"]) == 0
