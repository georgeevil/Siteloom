"""Ctrl-C through the CLI: a clean stop, not a traceback.

The reporter swallows `Interrupted` so it can record the run and print
the resume command, which leaves every `x = ...` inside the `with` block
unbound. Each command has to notice that and exit instead of formatting a
summary of work that never happened.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from siteloom import cli_library
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
        cli_library.library_app, ["index", "--config", "unused.yaml", "--all"]
    )
    assert result.exit_code == cli_library.INTERRUPTED_EXIT
    assert not isinstance(result.exception, UnboundLocalError)
    with Session() as session:
        run = session.query(OperationRun).one()
        assert run.status == "interrupted"
        assert run.resume_command.endswith("--all")
