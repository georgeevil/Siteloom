"""Factory reset (`siteloom reset`).

The interesting assertions are not "the rows are gone" but the three
ways a reset can be wrong: leaving one of the three stores behind so the
survivors contradict each other, deleting something that was never the
site's to delete (the config, the library's original archives), and
running while a live process holds the vector store open.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from siteloom.config import SiteConfig
from siteloom.reset import (
    UnsafeReset,
    check_safe,
    perform_reset,
    plan_reset,
)
from siteloom.store import (
    Annotation,
    Camera,
    Detection,
    Event,
    EventIdentity,
    Identity,
    LibraryItem,
    LibrarySource,
    OperationRun,
    User,
    get_session,
    init_db,
    make_engine,
)

TS = datetime(2026, 8, 14, 12, 0, 0)


@pytest.fixture
def site(tmp_path):
    """A config whose three stores are all populated, plus a library
    archive *outside* them (where `library add` normally points)."""
    media = tmp_path / "media"
    (media / "front-yard" / "2026-08-14").mkdir(parents=True)
    (media / "front-yard" / "2026-08-14" / "crop.jpg").write_bytes(b"crop")
    (media / "library" / "thumbs").mkdir(parents=True)
    (media / "library" / "thumbs" / "1.jpg").write_bytes(b"thumb")

    vectors = tmp_path / "identity_db"
    (vectors / "collection" / "face").mkdir(parents=True)
    (vectors / "meta.json").write_text("{}")

    training = tmp_path / "training"
    training.mkdir()
    (training / "face_ft.onnx").write_bytes(b"weights")

    archive = tmp_path / "Photos"
    archive.mkdir()
    (archive / "original.jpg").write_bytes(b"the operator's own photo")

    config = SiteConfig(site_id="t", site_name="Test Site")
    config.storage.media_dir = str(media)
    config.storage.db_url = f"sqlite:///{tmp_path}/siteloom.db"
    config.identity.vector_db_path = str(vectors)
    config.identity.face_projection_path = str(training / "face_projection.npy")
    config.training.output_dir = str(training)
    return config


@pytest.fixture
def session(site, tmp_path):
    engine = make_engine(site.storage.db_url)
    init_db(engine)
    with get_session(engine)() as s:
        s.add(Camera(id="front", site_id="t", name="Front"))
        identity = Identity(
            identifier_key="face", class_name="person", first_seen=TS, last_seen=TS
        )
        s.add(identity)
        s.flush()
        event = Event(
            camera_id="front",
            class_name="person",
            first_seen=TS,
            last_seen=TS,
            track_id=1,
        )
        s.add(event)
        s.flush()
        s.add(
            Detection(
                event_id=event.id,
                timestamp=TS,
                class_name="person",
                confidence=0.9,
                bbox="[0,0,10,10]",
            )
        )
        s.add(
            EventIdentity(
                event_id=event.id,
                identity_id=identity.id,
                identifier_key="face",
                similarity=0.9,
            )
        )
        source = LibrarySource(
            path=str(tmp_path / "Photos"), name="Photos", kind="directory", added_at=TS
        )
        s.add(source)
        s.flush()
        item = LibraryItem(
            source_id=source.id,
            path=str(tmp_path / "Photos" / "original.jpg"),
            kind="image",
            status="indexed",
            mtime=TS,
        )
        s.add(item)
        s.flush()
        s.add(
            Annotation(
                item_id=item.id,
                bbox="[0,0,1,1]",
                class_name="face",
                identity_id=identity.id,
                created_at=TS,
            )
        )
        s.commit()
        yield s


def _row_counts(session) -> dict[str, int]:
    from siteloom.store.models import Base

    return {
        t.name: len(session.execute(t.select()).fetchall())
        for t in Base.metadata.sorted_tables
    }


def test_the_plan_reads_every_store_without_touching_any(site, session):
    plan = plan_reset(site, session)
    by_name = {t.name: t.rows for t in plan.tables}
    assert by_name["events"] == 1
    assert by_name["identities"] == 1
    assert by_name["annotations"] == 1
    assert plan.files == 4  # crop + thumb + qdrant meta.json + trained model
    assert plan.bytes > 0
    # Still all there: planning is a read.
    assert Path(site.identity.vector_db_path, "meta.json").exists()
    assert _row_counts(session)["events"] == 1


def test_reset_clears_rows_media_and_vectors_together(site, session):
    perform_reset(site, session, plan_reset(site, session))

    assert all(n == 0 for n in _row_counts(session).values())
    # media_dir survives as a directory — config names it and the
    # indexer makes subdirectories under it at construction time.
    media = Path(site.storage.media_dir)
    assert media.is_dir() and not list(media.iterdir())
    # The vector directory goes whole; Qdrant recreates it on next open.
    assert not Path(site.identity.vector_db_path).exists()
    assert not list(Path(site.training.output_dir).iterdir())


def test_library_originals_are_never_touched(site, session, tmp_path):
    perform_reset(site, session, plan_reset(site, session))
    assert (tmp_path / "Photos" / "original.jpg").read_bytes()


def test_a_source_inside_media_dir_refuses_the_whole_reset(site, session):
    """`library add media/photos` is legal, and makes clearing media_dir
    a deletion of the operator's archive. Refuse, remove nothing."""
    inside = Path(site.storage.media_dir) / "photos"
    inside.mkdir()
    (inside / "original.jpg").write_bytes(b"irreplaceable")
    session.add(
        LibrarySource(path=str(inside), name="inside", kind="directory", added_at=TS)
    )
    session.commit()

    plan = plan_reset(site, session)
    with pytest.raises(UnsafeReset, match="library source"):
        check_safe(site, session, plan)
    with pytest.raises(UnsafeReset):
        perform_reset(site, session, plan)

    assert (inside / "original.jpg").exists()
    assert _row_counts(session)["events"] == 1


def test_a_live_operation_refuses_the_reset(site, session):
    """A running serve/run holds the Qdrant client open and would write
    fresh rows into the database we just emptied."""
    import os

    from siteloom.health import hostname, process_identity

    session.add(
        OperationRun(
            kind="serve",
            status="running",
            started_at=TS,
            updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
            pid=os.getpid(),
            host=hostname(),
            process_start=process_identity(os.getpid()),
        )
    )
    session.commit()

    with pytest.raises(UnsafeReset, match="still running"):
        perform_reset(site, session, plan_reset(site, session))
    assert _row_counts(session)["events"] == 1


def test_a_stale_run_does_not_block(site, session):
    session.add(
        OperationRun(
            kind="backfill",
            status="running",
            started_at=TS,
            updated_at=datetime.now(timezone.utc).replace(tzinfo=None)
            - timedelta(hours=2),
            pid=0,
            host="some-other-host",
        )
    )
    session.commit()

    perform_reset(site, session, plan_reset(site, session))
    assert _row_counts(session)["operation_runs"] == 0


def test_keep_users_holds_back_accounts_and_nothing_else(site, session):
    session.add(User(username="ops", password_hash="x", role="admin", created_at=TS))
    session.commit()

    plan = plan_reset(site, session, keep_users=True)
    assert "users" not in {t.name for t in plan.tables}
    perform_reset(site, session, plan)

    counts = _row_counts(session)
    assert counts["users"] == 1
    assert counts["events"] == 0 and counts["identities"] == 0


def test_ids_restart_so_the_first_event_is_event_one(site, session):
    perform_reset(site, session, plan_reset(site, session))
    session.add(
        Event(
            camera_id="front",
            class_name="person",
            first_seen=TS,
            last_seen=TS,
            track_id=1,
        )
    )
    session.commit()
    assert session.query(Event).one().id == 1


def test_an_already_clean_install_plans_nothing(site, session):
    perform_reset(site, session, plan_reset(site, session))
    assert plan_reset(site, session).is_empty


def test_missing_directories_are_not_an_error(site, session, tmp_path):
    site.storage.media_dir = str(tmp_path / "never-created")
    assert perform_reset(site, session, plan_reset(site, session)) == []
