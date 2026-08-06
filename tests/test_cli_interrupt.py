"""Ctrl-C through the CLI: a clean stop, and a truthful resume command.

Two promises are made to an operator who stops a job. That the process
ends cleanly rather than in a traceback — the reporter swallows
`Interrupted` so it can record the run, which leaves every `x = ...`
inside the `with` block unbound. And that the printed command resumes
*this* job: same source, same flags, not a similarly-named one.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from siteloom import cli, cli_library
from siteloom.config import SiteConfig
from siteloom.progress import Interrupted
from siteloom.store import OperationRun, get_session, init_db, make_engine

runner = CliRunner()


@pytest.fixture
def Session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/cli.db")
    init_db(engine)
    return get_session(engine)


class InterruptingIndexer:
    """Stands in for the real indexer: stops the way Ctrl-C does."""

    def process(self, *args, progress=None, **kwargs):
        raise Interrupted


def test_interrupted_index_exits_cleanly(Session, monkeypatch):
    monkeypatch.setattr(
        cli_library,
        "_setup",
        lambda *a, **kw: (SiteConfig(site_id="test"), Session, InterruptingIndexer()),
    )
    result = runner.invoke(
        cli.app, ["library", "index", "--config", "unused.yaml", "--all"]
    )
    assert result.exit_code == cli_library.INTERRUPTED_EXIT
    assert not isinstance(result.exception, UnboundLocalError)
    with Session() as session:
        run = session.query(OperationRun).one()
        assert run.status == "interrupted"
        assert run.resume_command.endswith("--all")


# -- the resume command is the invocation -----------------------------------


def resume_recorded(Session) -> str:
    with Session() as session:
        return session.query(OperationRun).one().resume_command


def test_resume_keeps_every_flag(Session, monkeypatch):
    """Dropped flags meant a resumed index could widen to every source
    or silently switch identification back on."""
    monkeypatch.setattr(
        cli_library,
        "_setup",
        lambda *a, **kw: (SiteConfig(site_id="test"), Session, InterruptingIndexer()),
    )
    result = runner.invoke(
        cli.app,
        ["library", "index", "--config", "sites/kai.yaml", "--source-id", "2",
         "--no-identify", "--limit", "500"],
    )
    assert result.exit_code == cli_library.INTERRUPTED_EXIT
    assert resume_recorded(Session) == (
        "siteloom library index --config sites/kai.yaml "
        "--source-id 2 --limit 500 --no-identify"
    )


def test_resume_omits_untouched_defaults(Session, monkeypatch):
    monkeypatch.setattr(
        cli_library,
        "_setup",
        lambda *a, **kw: (SiteConfig(site_id="test"), Session, InterruptingIndexer()),
    )
    runner.invoke(cli.app, ["library", "index", "--config", "site.yaml"])
    assert resume_recorded(Session) == "siteloom library index --config site.yaml"


def test_takeout_resume_quotes_the_path_and_keeps_no_auto_verify(Session, monkeypatch):
    """The dangerous case: resuming without --no-auto-verify starts
    writing unreviewed annotations that training reads as ground truth."""
    from siteloom.library import takeout as takeout_module

    class InterruptingImporter:
        def __init__(self, *args, **kwargs):
            pass

        def import_tree(self, *args, **kwargs):
            raise Interrupted

    monkeypatch.setattr(
        cli_library,
        "_setup",
        lambda *a, **kw: (SiteConfig(site_id="test"), Session, object()),
    )
    monkeypatch.setattr(takeout_module, "TakeoutImporter", InterruptingImporter)
    result = runner.invoke(
        cli.app,
        ["takeout", "import", "/archives/My Photos", "--config", "site.yaml",
         "--no-auto-verify"],
    )
    assert result.exit_code == cli_library.INTERRUPTED_EXIT
    assert resume_recorded(Session) == (
        "siteloom takeout import '/archives/My Photos' "
        "--config site.yaml --no-auto-verify"
    )
