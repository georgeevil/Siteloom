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


# -- ingest restart (event stitching) ---------------------------------------


def _ingest_config(tmp_path, name: str, source: Path, identity: bool = False) -> SiteConfig:
    from siteloom.config import CameraConfig

    root = tmp_path / name
    root.mkdir()
    return SiteConfig(
        site_id="test",
        cameras=[
            CameraConfig(
                id="cam1",
                adapter="file",
                source=str(source),
                sample_fps=5.0,
                modules=["detection", "identity"] if identity else ["detection"],
            )
        ],
        identity=(
            IdentityConfig(vector_db_path=str(root / "vectors"))
            if identity
            else IdentityConfig(enabled=False)
        ),
        storage=StorageConfig(
            db_url=f"sqlite:///{root}/ingest.db", media_dir=str(root / "media")
        ),
    )


def _ingest_snapshot(Session) -> dict:
    """Events and identity links by content — row, track, and identity
    ids differ across runs. confidence_sum is rounded because merge order
    can differ float-associatively between a clean and a split run."""
    from siteloom.store import Event, EventIdentity

    with Session() as session:
        events = sorted(
            (
                e.camera_id,
                e.class_name,
                e.first_seen,
                e.last_seen,
                e.detection_count,
                round(e.confidence_sum, 6),
                e.significant,
            )
            for e in session.query(Event).all()
        )
        links = sorted(
            (
                link.event.first_seen,
                link.identifier_key,
                round(link.similarity, 6),
                link.hit_count,
                link.matched_by,
            )
            for link in session.query(EventIdentity).all()
        )
    return {"events": events, "links": links}


def _make_ingest(config, tracks_per_clip):
    """One IngestService whose stub detector emits the given track id for
    every sampled frame of each clip, in order (10 samples per clip)."""
    from test_ingest import SequenceDetector, _det

    from siteloom.dispatch import LocalBackend
    from siteloom.ingest import IngestService

    frames = [
        [_det(track_id=t)] for t in tracks_per_clip for _ in range(10)
    ]
    dispatcher = LocalBackend()
    dispatcher.register("detection", SequenceDetector(frames))
    return IngestService(config, dispatcher=dispatcher)


def test_restarted_ingest_stitches_like_a_clean_run(tmp_path, sample_video):
    """The stitch fallback reads prior rows, so a run interrupted at a
    clip boundary and resumed by a fresh process (new tracker, new track
    ids) must land on the same events as one uninterrupted run. Frame
    timestamps come from file mtime + frame offset — never wall clock —
    which is what makes this deterministic."""
    import os
    import shutil

    corpus = tmp_path / "clips"
    corpus.mkdir()
    clip1 = corpus / "a_clip1.mp4"
    clip2 = corpus / "b_clip2.mp4"
    shutil.copy(sample_video, clip1)
    shutil.copy(sample_video, clip2)
    base = 1_754_000_000  # clip2 starts 3 s after clip1 (2 s of media)
    os.utime(clip1, (base, base))
    os.utime(clip2, (base + 3, base + 3))

    # Clean: one service processes both clips; the tracker restarts per
    # clip, so clip2 arrives under a different track id and must stitch.
    clean = _make_ingest(
        _ingest_config(tmp_path, "clean", corpus), tracks_per_clip=[1, 2]
    )
    clean.run_camera(clean.config.cameras[0])

    # Interrupted-at-clip-boundary + resumed by a fresh process: same DB,
    # new service (fresh detector state) per half.
    half1 = _make_ingest(
        _ingest_config(tmp_path, "split", clip1), tracks_per_clip=[1]
    )
    half1.run_camera(half1.config.cameras[0])
    resumed_config = _ingest_config(tmp_path, "split-resume", clip2)
    resumed_config.storage = half1.config.storage  # same database
    half2 = _make_ingest(resumed_config, tracks_per_clip=[1])
    half2.run_camera(half2.config.cameras[0])

    clean_snap = _ingest_snapshot(clean.Session)
    split_snap = _ingest_snapshot(half2.Session)
    assert clean_snap == split_snap
    # And the stitch actually happened: one event spanning both clips.
    assert len(clean_snap["events"]) == 1
    assert clean_snap["events"][0][4] == 20  # detections from both clips


def _make_identity_ingest(config, specs):
    """Like _make_ingest, but with the identity path live: `specs` is a
    list of (track_id, bbox) per clip, 10 samples per clip, all resolving
    to one constant embedding."""
    from test_ingest import SequenceDetector, StubIdentity, _det

    from siteloom.dispatch import LocalBackend
    from siteloom.ingest import IngestService

    frames = [
        [_det(track_id=t, bbox=b)] for (t, b) in specs for _ in range(10)
    ]
    dispatcher = LocalBackend()
    dispatcher.register("detection", SequenceDetector(frames))
    dispatcher.register("identity", StubIdentity())
    return IngestService(config, dispatcher=dispatcher)


def test_restarted_ingest_merges_identity_fragments_like_a_clean_run(
    tmp_path, sample_video
):
    """The identity-aware merge (CLD-40) reads prior rows, so it joins
    stitching in this harness: clip2's subject reappears elsewhere in the
    frame (no IoU overlap) under a fresh track id, and only the shared
    identity can fold the fragments together. Interrupted at the clip
    boundary and resumed by a fresh process, the merge must land on the
    same content as one uninterrupted run."""
    import os
    import shutil

    box_a = (10.0, 10.0, 80.0, 120.0)
    box_b = (600.0, 400.0, 700.0, 560.0)

    corpus = tmp_path / "clips"
    corpus.mkdir()
    clip1 = corpus / "a_clip1.mp4"
    clip2 = corpus / "b_clip2.mp4"
    shutil.copy(sample_video, clip1)
    shutil.copy(sample_video, clip2)
    base = 1_754_000_000
    os.utime(clip1, (base, base))
    os.utime(clip2, (base + 3, base + 3))

    clean = _make_identity_ingest(
        _ingest_config(tmp_path, "clean", corpus, identity=True),
        specs=[(1, box_a), (2, box_b)],
    )
    clean.run_camera(clean.config.cameras[0])

    half1 = _make_identity_ingest(
        _ingest_config(tmp_path, "split", clip1, identity=True),
        specs=[(1, box_a)],
    )
    half1.run_camera(half1.config.cameras[0])
    resumed_config = _ingest_config(tmp_path, "split-resume", clip2, identity=True)
    resumed_config.storage = half1.config.storage  # same database...
    resumed_config.identity = half1.config.identity  # ...same vector store
    # Embedded Qdrant is one client per path per machine: release half1's
    # before the resuming process opens it.
    half1.resolver.vectors.close()
    # Track 7: a fresh tracker's id that doesn't collide with half1's
    # (the colliding-restart case is the stitch test above).
    half2 = _make_identity_ingest(resumed_config, specs=[(7, box_b)])
    half2.run_camera(half2.config.cameras[0])

    clean_snap = _ingest_snapshot(clean.Session)
    split_snap = _ingest_snapshot(half2.Session)
    assert clean_snap == split_snap
    # And the merge actually happened: one event, one identity pairing.
    assert len(clean_snap["events"]) == 1
    assert clean_snap["events"][0][4] == 20
    assert len(clean_snap["links"]) == 1
