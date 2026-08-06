"""Interrupt + resume must land exactly where an uninterrupted run lands.

This is the claim the whole resumable-jobs design makes, and the one
nothing else tests: `test_progress.py` checks that a stop is *recorded*,
`test_cli_interrupt.py` that the process *exits* cleanly. Neither says
the work came out the same.

The method is a differential: run a corpus clean in one database, run it
interrupted-then-resumed in another, and compare the results by content
rather than by row id. Interrupts are injected at a chosen point rather
than signalled, because "stop after exactly item k" is deterministic
where a real SIGINT is a race — signal delivery itself is covered in
`test_progress.py` and `test_health.py`.
"""

from __future__ import annotations

import pytest
from test_library import StubDetector
from test_takeout_assignment import StubFace, make_photo, write_sidecar

from siteloom.config import IdentityConfig, LibraryConfig, SiteConfig, StorageConfig
from siteloom.dispatch import LocalBackend
from siteloom.library import LibraryIndexer
from siteloom.library.takeout import TakeoutImporter
from siteloom.progress import Interrupted, ProgressReporter
from siteloom.store import Annotation, ItemTag, LibraryItem, get_session, init_db, make_engine

PEOPLE_PAIRS = [
    ["Ann"], ["Bob"], ["Ann", "Bob"], ["Cara"], ["Ann"],
    ["Bob", "Cara"], ["Cara"], ["Ann", "Cara"], ["Bob"],
]


class InterruptingReporter(ProgressReporter):
    """The real reporter, told exactly when to stop.

    Flipping the same flag a signal handler flips, at a chosen phase and
    position, exercises the real interrupt path (the DB row, the
    counters, the `Interrupted` raise) without racing a signal.
    """

    def __init__(self, *args, at_phase: str, after: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._at_phase = at_phase
        self._after = after

    def advance(self, n: int = 1, **counters: int) -> None:
        super().advance(n, **counters)
        if (
            not self.interrupt_requested
            and self._phase_name == self._at_phase
            and self._current >= self._after
        ):
            self.interrupt_requested = True


def make_config(tmp_path, name: str) -> SiteConfig:
    root = tmp_path / name
    root.mkdir()
    return SiteConfig(
        site_id="test",
        identity=IdentityConfig(enabled=False),
        library=LibraryConfig(batch_size=10),
        storage=StorageConfig(
            db_url=f"sqlite:///{root}/lib.db", media_dir=str(root / "media")
        ),
    )


def make_indexer(config: SiteConfig) -> LibraryIndexer:
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    dispatcher = LocalBackend()
    dispatcher.register("detection", StubDetector())
    return LibraryIndexer(config, get_session(engine), dispatcher, resolver=None)


# -- library index ----------------------------------------------------------


@pytest.fixture
def photos(tmp_path):
    """Nine images, enough to stop in several interesting places."""
    import cv2
    import numpy as np

    d = tmp_path / "corpus"
    d.mkdir()
    for i in range(9):
        image = np.full((120, 160, 3), 20 + i * 20, dtype=np.uint8)
        cv2.imwrite(str(d / f"img_{i:02d}.jpg"), image)
    return d


def index_snapshot(Session) -> dict:
    """Content, keyed by path — row ids differ between databases and are
    not part of what "the same result" means."""
    with Session() as session:
        items = {
            item.path: (item.status, item.attempts, bool(item.thumb_path))
            for item in session.query(LibraryItem).all()
        }
        annotations = sorted(
            (
                annotation.item.path,
                annotation.frame_index,
                annotation.bbox,
                annotation.class_name,
                annotation.source,
                annotation.verified,
            )
            for annotation in session.query(Annotation).all()
        )
    return {"items": items, "annotations": annotations}


def index_all(indexer, corpus, reporter=None) -> None:
    source = indexer.add_source(corpus)
    indexer.scan(source.id)
    try:
        indexer.process(limit=1000, progress=reporter)
    except Interrupted:
        pass


@pytest.mark.parametrize("stop_after", [1, 4, 8])
def test_interrupted_index_plus_resume_equals_clean_run(tmp_path, photos, stop_after):
    clean = make_indexer(make_config(tmp_path, "clean"))
    index_all(clean, photos)

    config = make_config(tmp_path, "stopped")
    stopped = make_indexer(config)
    reporter = InterruptingReporter(
        stopped.Session,
        "library-index",
        at_phase="Indexing media",
        after=stop_after,
        bar=False,
    )
    with reporter:
        index_all(stopped, photos, reporter)

    # Mid-run state: stopped where it was told to, nothing in limbo.
    with stopped.Session() as session:
        assert session.query(LibraryItem).filter_by(status="indexed").count() == stop_after
        assert session.query(LibraryItem).filter_by(status="pending").count() == 9 - stop_after

    resumed = stopped.process(limit=1000)  # what the resume command does
    assert resumed.processed == 9 - stop_after
    assert resumed.remaining == 0

    assert index_snapshot(stopped.Session) == index_snapshot(clean.Session)
    # No item was handled twice: resuming is not redoing.
    with stopped.Session() as session:
        assert {i.attempts for i in session.query(LibraryItem).all()} == {1}


def test_interrupt_is_recorded_with_its_position(tmp_path, photos):
    """The run row is the only trace a killed terminal leaves behind, so
    it has to agree with what the database actually contains."""
    from siteloom.store import OperationRun

    config = make_config(tmp_path, "recorded")
    indexer = make_indexer(config)
    reporter = InterruptingReporter(
        indexer.Session,
        "library-index",
        at_phase="Indexing media",
        after=3,
        resume_command="siteloom library index --config x.yaml --all",
        bar=False,
    )
    with reporter:
        index_all(indexer, photos, reporter)

    with indexer.Session() as session:
        run = session.query(OperationRun).one()
        assert run.status == "interrupted"
        assert run.current == 3
        assert run.total == 9
        assert session.query(LibraryItem).filter_by(status="indexed").count() == run.current


# -- takeout import ---------------------------------------------------------


@pytest.fixture
def takeout(tmp_path):
    d = tmp_path / "Takeout" / "Google Photos" / "Album"
    d.mkdir(parents=True)
    for i, names in enumerate(PEOPLE_PAIRS):
        photo = d / f"photo_{i:02d}.jpg"
        make_photo(photo, names)
        write_sidecar(
            d / f"photo_{i:02d}.jpg.supplemental-metadata.json", photo.name, names
        )
    return d


def make_importer(config: SiteConfig) -> TakeoutImporter:
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    indexer = LibraryIndexer(
        config, get_session(engine), LocalBackend(), resolver=None
    )
    return TakeoutImporter(indexer, face_embedder=StubFace())


def takeout_snapshot(Session) -> dict:
    with Session() as session:
        faces = sorted(
            (
                annotation.item.path,
                annotation.bbox,
                annotation.proposed_name,
                annotation.proposal_basis,
                annotation.verified,
                bool(annotation.crop_path),  # the path embeds a row id
            )
            for annotation in session.query(Annotation).all()
        )
        tags = sorted(
            (tag.item.path, tag.kind, tag.value)
            for tag in session.query(ItemTag).all()
        )
    return {"faces": faces, "tags": tags}


def import_tree(importer, root, reporter=None) -> None:
    importer.progress = reporter or importer.progress
    try:
        importer.import_tree(root, batch_size=1)
    except Interrupted:
        pass


@pytest.mark.parametrize(
    "phase", ["Reading metadata", "Detecting faces", "Matching names"]
)
def test_interrupted_takeout_plus_resume_equals_clean_run(tmp_path, takeout, phase):
    """Takeout resume is not a status column — it is three different
    skip-what's-done rules, one per phase, and each has to hold."""
    clean = make_importer(make_config(tmp_path, "clean"))
    import_tree(clean, takeout)

    stopped = make_importer(make_config(tmp_path, "stopped"))
    reporter = InterruptingReporter(
        stopped.Session, "takeout-import", at_phase=phase, after=2, bar=False
    )
    with reporter:
        import_tree(stopped, takeout, reporter)
    assert reporter.interrupt_requested, f"never reached phase {phase!r}"

    from siteloom.library.takeout import _NullProgress

    stopped.progress = _NullProgress()
    stopped.import_tree(takeout, batch_size=1)  # the resume

    assert takeout_snapshot(stopped.Session) == takeout_snapshot(clean.Session)


def test_resumed_takeout_does_not_duplicate_tags_or_faces(tmp_path, takeout):
    """The failure this guards: re-registering an item on resume adding a
    second copy of every person tag, or a second face annotation."""
    importer = make_importer(make_config(tmp_path, "twice"))
    import_tree(importer, takeout)
    first = takeout_snapshot(importer.Session)

    importer.import_tree(takeout, batch_size=1)  # a full, redundant rerun
    assert takeout_snapshot(importer.Session) == first
