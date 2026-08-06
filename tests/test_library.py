"""Library indexing tests — stub detection, real DB and file walking."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from siteloom.config import (
    CameraConfig,
    IdentityConfig,
    LibraryConfig,
    SiteConfig,
    StorageConfig,
)
from siteloom.dispatch import LocalBackend
from siteloom.library import LibraryIndexer
from siteloom.store import (
    Annotation,
    LibraryItem,
    get_session,
    init_db,
    make_engine,
)


class StubDetector:
    """One 'person' box per frame, with a crop."""

    def process(self, job):
        crop = np.full((40, 30, 3), 120, dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", crop)
        return {
            "detections": [
                {
                    "class_name": "person",
                    "confidence": 0.9,
                    "bbox": [10.0, 10.0, 50.0, 90.0],
                    "track_id": None,
                    "zones": [],
                    "crop_jpeg": jpeg.tobytes(),
                }
            ]
        }


@pytest.fixture
def photo_dir(tmp_path):
    d = tmp_path / "photos"
    (d / "nested").mkdir(parents=True)
    for i, path in enumerate(
        [d / "a.jpg", d / "b.png", d / "nested" / "c.jpg"]
    ):
        image = np.full((120, 160, 3), 30 + i * 40, dtype=np.uint8)
        cv2.imwrite(str(path), image)
    (d / "notes.txt").write_text("not media")
    return d


@pytest.fixture
def indexer(tmp_path, photo_dir):
    config = SiteConfig(
        site_id="test",
        cameras=[],
        identity=IdentityConfig(enabled=False),
        library=LibraryConfig(batch_size=10),
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/lib.db", media_dir=str(tmp_path / "media")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    dispatcher = LocalBackend()
    dispatcher.register("detection", StubDetector())
    return LibraryIndexer(config, Session, dispatcher, resolver=None)


def test_add_source_and_scan(indexer, photo_dir):
    source = indexer.add_source(photo_dir, name="Photos")
    result = indexer.scan(source.id)
    assert result.added == 3  # notes.txt ignored, nested included
    assert result.total_pending == 3


def test_scan_is_idempotent(indexer, photo_dir):
    source = indexer.add_source(photo_dir)
    indexer.scan(source.id)
    again = indexer.scan(source.id)
    assert again.added == 0
    assert again.skipped == 3


def test_scan_detects_changed_file(indexer, photo_dir):
    source = indexer.add_source(photo_dir)
    indexer.scan(source.id)
    indexer.process(limit=10)
    import os
    import time

    target = photo_dir / "a.jpg"
    os.utime(target, (time.time() + 10, time.time() + 10))
    result = indexer.scan(source.id)
    assert result.updated == 1


def test_partial_indexing_is_resumable(indexer, photo_dir):
    """The whole point of the two-phase design: index in chunks."""
    source = indexer.add_source(photo_dir)
    indexer.scan(source.id)

    first = indexer.process(limit=2)
    assert first.processed == 2
    assert first.remaining == 1

    second = indexer.process(limit=10)
    assert second.processed == 1
    assert second.remaining == 0

    # A third run has nothing left to do.
    assert indexer.process(limit=10).processed == 0


def test_process_creates_annotations_and_thumbs(indexer, photo_dir):
    source = indexer.add_source(photo_dir)
    indexer.scan(source.id)
    result = indexer.process(limit=10)
    assert result.annotations == 3

    with indexer.Session() as session:
        items = session.query(LibraryItem).all()
        annotations = session.query(Annotation).all()
    assert all(i.status == "indexed" for i in items)
    assert all(i.thumb_path for i in items)
    assert all(i.width == 160 and i.height == 120 for i in items)
    assert len(annotations) == 3
    # Boxes are stored normalized so they survive thumbnailing.
    bbox = json.loads(annotations[0].bbox)
    assert all(0.0 <= v <= 1.0 for v in bbox)
    assert annotations[0].source == "auto"
    assert annotations[0].verified is False


def test_failed_item_is_recorded_not_dropped(indexer, photo_dir):
    source = indexer.add_source(photo_dir)
    indexer.scan(source.id)
    (photo_dir / "a.jpg").write_bytes(b"not a real jpeg")
    indexer.process(limit=10)
    with indexer.Session() as session:
        failed = session.query(LibraryItem).filter_by(status="failed").all()
    assert len(failed) == 1
    assert "could not decode" in failed[0].error


def test_reindex_preserves_human_annotations(indexer, photo_dir):
    source = indexer.add_source(photo_dir)
    indexer.scan(source.id)
    indexer.process(limit=10)

    with indexer.Session() as session:
        item = session.query(LibraryItem).first()
        session.add(
            Annotation(
                item_id=item.id,
                bbox=json.dumps([0.1, 0.1, 0.4, 0.4]),
                class_name="face",
                source="human",
                verified=True,
                created_at=item.mtime,
            )
        )
        item.status = "pending"
        session.commit()
        item_id = item.id

    indexer.process(limit=10)
    with indexer.Session() as session:
        rows = session.query(Annotation).filter_by(item_id=item_id).all()
    kinds = {(a.source, a.verified) for a in rows}
    assert ("human", True) in kinds  # human work survived
    assert ("auto", False) in kinds  # machine boxes regenerated
