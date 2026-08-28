"""The /backfill console (CLD-93).

Backfill was terminal-only, so the console could show a gap and not close
it — and since audio never reaches the detector on a live stream, NVR
clips are the only camera-derived source of NoiseEvent rows at all.

Two properties carry the screen. The read must show the resumability
model (`the remaining clips are the ones still pending`, scoped per
camera) and distinguish its three empty states, because "nothing here"
means something different each time. The write must be single-flight
*per camera* and admin-only: it downloads from the NVR and writes events,
and two sweeps of one camera in one process would share the detector's
tracker state for it — while different cameras run side by side, one job
row each (CLD-317).

No NVR, no weights, no cameras: the adapter is stubbed and the run never
gets past the fake scan.
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from siteloom.config import CameraConfig, IdentityConfig, SiteConfig, StorageConfig
from siteloom.store import (
    AuditLog,
    BackfillClip,
    Camera,
    User,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web import backfill_routes
from siteloom.web.auth import hash_password, required_role
from siteloom.web.app import create_app
from siteloom.web.backfill_routes import parse_range

T0 = datetime(2026, 8, 6, 12, 0, 0)


def unifi(camera_id="front", name="Front door"):
    return CameraConfig(id=camera_id, name=name, adapter="unifi", source="nvr-1")


@pytest.fixture(autouse=True)
def clean_state():
    """Module state is per-process and tests build many apps."""
    backfill_routes._state["runs"].clear()
    yield
    backfill_routes._state["runs"].clear()


def idle() -> bool:
    return not backfill_routes.running_cameras()


def build(tmp_path, cameras, clips=()):
    config = SiteConfig(
        site_id="t",
        cameras=cameras,
        identity=IdentityConfig(enabled=False),
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/bf.db", media_dir=str(tmp_path / "m")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        for camera_id in {c.camera_id for c in clips} | {c.id for c in cameras}:
            s.add(Camera(id=camera_id, site_id="t", name=camera_id))
        for clip in clips:
            s.add(clip)
        s.commit()
    return TestClient(create_app(config), follow_redirects=False), Session, config


def clip(camera_id, external_id, status="pending", at=T0, frames=0, processed=None):
    return BackfillClip(
        camera_id=camera_id,
        external_id=external_id,
        kind="motion",
        start=at,
        end=at + timedelta(seconds=20),
        status=status,
        frames=frames,
        processed_at=processed,
    )


# -- the read --------------------------------------------------------------


def table_rows(body: str) -> dict[str, list[str]]:
    """The clip table's numeric cells, per camera name.

    Parsed rather than string-searched because the camera also appears in
    the start form's picker: a bare substring test would pass on the
    form's copy of the name while the table said something else.
    """
    rows = {}
    for block in body.split('<div class="brow">')[1:]:
        name = re.search(r'class="bf-camname">\s*([^<\n]+)', block)
        rows[name.group(1).strip()] = re.findall(r'class="num[^"]*">([^<]*)<', block)
    return rows


def test_counts_are_scoped_per_camera(tmp_path):
    """Two cameras' sweeps are independent operations; a pooled count
    would say a camera is finished because a different one is."""
    client, _, _ = build(
        tmp_path,
        [unifi("front", "Front door"), unifi("gate", "Gate")],
        clips=[
            clip("front", "a", "done", frames=120, processed=T0),
            clip("front", "b", "pending", at=T0 + timedelta(hours=1)),
            clip("front", "c", "pending", at=T0 + timedelta(hours=2)),
            clip("gate", "d", "failed"),
        ],
    )
    rows = table_rows(client.get("/backfill").text)
    # pending, done, failed, frames.
    assert rows["Front door"] == ["2", "1", "0", "120"]
    assert rows["Gate"] == ["0", "0", "1", "0"]


def test_the_covered_range_is_shown(tmp_path):
    client, _, _ = build(
        tmp_path,
        [unifi("front")],
        clips=[
            clip("front", "a", "done", at=T0),
            clip("front", "b", "pending", at=T0 + timedelta(hours=3)),
        ],
    )
    body = client.get("/backfill").text
    assert "2026-08-06 12:00" in body
    assert "2026-08-06 15:00" in body


def test_clips_from_a_camera_no_longer_configured_still_appear(tmp_path):
    """Its footage is in the database either way; dropping the row would
    make the page's counts disagree with the table."""
    client, _, _ = build(
        tmp_path, [unifi("front")], clips=[clip("removed", "x", "pending")]
    )
    body = client.get("/backfill").text
    assert "removed" in body
    assert "no camera with this id is configured" in body


# -- empty states ----------------------------------------------------------


def test_empty_state_no_unifi_cameras(tmp_path):
    client, _, _ = build(
        tmp_path, [CameraConfig(id="clip", adapter="file", source="x")]
    )
    body = client.get("/backfill").text
    assert "No UniFi camera is configured" in body
    # Nothing to start, so no form pretending otherwise.
    assert "/backfill/start" not in body
    assert "Nothing scanned yet" not in body


def test_empty_state_nothing_scanned_yet(tmp_path):
    client, _, _ = build(tmp_path, [unifi("front")])
    body = client.get("/backfill").text
    assert "Nothing scanned yet" in body
    assert "No UniFi camera is configured" not in body
    assert "Every scanned clip has been indexed" not in body
    # The distinction the message exists to draw.
    assert "not the same as" in body
    assert "/backfill/start" in body


def test_empty_state_everything_indexed(tmp_path):
    client, _, _ = build(
        tmp_path, [unifi("front")], clips=[clip("front", "a", "done", frames=90)]
    )
    body = client.get("/backfill").text
    assert "Every scanned clip has been indexed" in body
    assert "Nothing scanned yet" not in body


def test_outstanding_work_gets_no_empty_state(tmp_path):
    client, _, _ = build(
        tmp_path, [unifi("front")], clips=[clip("front", "a", "pending")]
    )
    body = client.get("/backfill").text
    for message in (
        "No UniFi camera is configured",
        "Nothing scanned yet",
        "Every scanned clip has been indexed",
    ):
        assert message not in body


def test_a_failure_is_not_counted_as_pending(tmp_path):
    """`failed` is terminal until retried — folding it into pending would
    show a sweep as finished with clips that will never run."""
    client, _, _ = build(
        tmp_path, [unifi("front")], clips=[clip("front", "a", "failed")]
    )
    body = client.get("/backfill").text
    # Not "all done" (a failure is outstanding), the row says nothing will
    # pick it up on its own, and the retry is offered.
    assert "Every scanned clip has been indexed" not in body
    assert "will not be retried unless asked" in body
    assert "Retry earlier failures" in body
    assert table_rows(body)["Front door"] == ["0", "0", "1", "0"]


def test_the_archive_path_is_scoped_out_with_its_reason(tmp_path):
    """Plain `siteloom backfill` has no per-file checkpointing (CLD-12);
    the screen must not imply the guarantee the rest of it rests on."""
    client, _, _ = build(tmp_path, [unifi("front")])
    body = client.get("/backfill").text
    assert "no per-file checkpointing" in body
    assert "CLD-12" in body


# -- starting a run --------------------------------------------------------


class FakeScan:
    added = 0
    skipped = 3
    pending = 0


def stub_run(monkeypatch, seen: dict, block: threading.Event | None = None):
    """Stand in for the whole NVR path: no adapter, no download, no
    pipeline. The scan records what it was asked for."""
    import siteloom.backfill as backfill_mod
    import siteloom.ingest as ingest_mod

    class FakeService:
        def __init__(self, cfg):
            self.config = cfg
            self.Session = get_session(make_engine(cfg.storage.db_url))

    class FakeBackfill:
        def __init__(self, service, cam):
            seen["camera"] = cam.id
            seen.setdefault("cameras", []).append(cam.id)

        def scan(self, start, end, **kwargs):
            seen["range"] = (start, end)
            seen["kwargs"] = kwargs
            return FakeScan()

        def process(self, **kwargs):
            seen["process"] = kwargs
            if block is not None:
                block.wait(5)
            return backfill_mod.BackfillResult(
                processed=0, frames=0, failed=0, remaining=0,
                failed_total=0, retried=0,
            )

    monkeypatch.setattr(ingest_mod, "IngestService", FakeService)
    monkeypatch.setattr(backfill_mod, "UnifiBackfill", FakeBackfill)


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_start_runs_the_scan_in_the_background(tmp_path, monkeypatch):
    seen: dict = {}
    stub_run(monkeypatch, seen)
    client, _, _ = build(tmp_path, [unifi("front")])
    r = client.post(
        "/backfill/start",
        data={"camera": "front", "start": "2026-08-06T09:00", "mode": "events"},
    )
    assert r.status_code == 303
    assert wait_for(idle)
    assert seen["cameras"] == ["front"]
    # Motion windows are the default: no chunking asked for.
    assert seen["kwargs"]["chunk_minutes"] is None
    assert seen["process"]["retry_failed"] is False
    start, end = seen["range"]
    assert start.tzinfo is not None and end > start


def test_full_sweep_passes_its_chunk_size(tmp_path, monkeypatch):
    seen: dict = {}
    stub_run(monkeypatch, seen)
    client, _, _ = build(tmp_path, [unifi("front")])
    client.post(
        "/backfill/start",
        data={
            "camera": "front",
            "start": "2026-08-06T09:00",
            "end": "2026-08-06T10:00",
            "mode": "sweep",
            "chunk_minutes": "30",
        },
    )
    assert wait_for(idle)
    assert seen["kwargs"]["chunk_minutes"] == 30.0


def busy(camera_id: str, name: str | None = None) -> None:
    """A run in flight for this camera, as the coordinator would leave it."""
    backfill_routes._state["runs"][camera_id] = {
        "camera": camera_id,
        "name": name or camera_id,
        "status": "processing",
        "added": 1,
        "skipped": 0,
    }


def test_a_second_start_of_the_same_camera_is_refused_not_queued(tmp_path):
    """The pipeline keeps per-camera tracker state, so the same camera
    twice is a failure — refused with the camera named, nothing started."""
    client, _, _ = build(tmp_path, [unifi("front", "Front door")])
    busy("front", "Front door")
    r = client.post(
        "/backfill/start", data={"camera": "front", "start": "2026-08-06T09:00"}
    )
    assert r.status_code == 409
    assert "Front door is already running" in r.text
    # With every camera running the button is disabled rather than
    # merely failing on click.
    assert "Backfill running" in client.get("/backfill").text


def test_an_idle_camera_may_start_while_another_runs(tmp_path, monkeypatch):
    """Different cameras are the shape live ingest already runs in — one
    thread each over the shared pipeline — so the guard is per camera."""
    seen: dict = {}
    stub_run(monkeypatch, seen)
    client, _, _ = build(tmp_path, [unifi("front", "Front door"), unifi("gate", "Gate")])
    busy("front", "Front door")
    page = client.get("/backfill").text
    assert "Backfill running" not in page  # one idle camera: the button stays live
    r = client.post(
        "/backfill/start", data={"camera": "gate", "start": "2026-08-06T09:00"}
    )
    assert r.status_code == 303
    assert wait_for(lambda: backfill_routes._state["runs"]["gate"]["status"] == "complete")
    assert seen["cameras"] == ["gate"]
    assert backfill_routes._state["runs"]["front"]["status"] == "processing"


def test_a_mixed_pick_with_one_busy_camera_starts_nothing(tmp_path, monkeypatch):
    """All or nothing: a partial start would leave the form lying about
    which cameras it began."""
    seen: dict = {}
    stub_run(monkeypatch, seen)
    client, _, _ = build(tmp_path, [unifi("front", "Front door"), unifi("gate", "Gate")])
    busy("front", "Front door")
    r = client.post(
        "/backfill/start",
        data={"cameras": ["front", "gate"], "start": "2026-08-06T09:00"},
    )
    assert r.status_code == 409
    assert "Front door" in r.text
    assert "gate" not in backfill_routes._state["runs"]
    assert "cameras" not in seen


def test_several_cameras_start_from_one_submit_one_row_each(tmp_path, monkeypatch):
    from siteloom.store import OperationRun

    seen: dict = {}
    stub_run(monkeypatch, seen)
    client, Session, _ = build(
        tmp_path, [unifi("front", "Front door"), unifi("gate", "Gate")]
    )
    r = client.post(
        "/backfill/start",
        # The same camera twice on the wire is one run, not a race.
        data={"cameras": ["front", "gate", "front"], "start": "2026-08-06T09:00"},
    )
    assert r.status_code == 303
    assert wait_for(idle)
    assert sorted(seen["cameras"]) == ["front", "gate"]
    with Session() as s:
        runs = s.scalars(select(OperationRun)).all()
    assert sorted(run.target for run in runs) == ["front", "gate"]
    assert all(run.kind == "backfill-unifi" for run in runs)
    # Each row's resume line continues *that* camera.
    by_target = {run.target: run.resume_command for run in runs}
    assert by_target["gate"].startswith("siteloom backfill-unifi gate --start")
    assert "front" not in by_target["gate"]
    # One banner per camera, each naming its own (the stub scans nothing
    # new, so both read as the no-op they are).
    body = client.get("/backfill").text
    assert "Nothing new to do for Front door" in body
    assert "Nothing new to do for Gate" in body


def test_no_camera_picked_is_a_400(tmp_path):
    client, _, _ = build(tmp_path, [unifi("front")])
    r = client.post("/backfill/start", data={"start": "2026-08-06T09:00"})
    assert r.status_code == 400
    assert "at least one camera" in r.text
    assert idle()


def test_an_unknown_camera_is_rejected(tmp_path):
    client, _, _ = build(tmp_path, [unifi("front")])
    r = client.post(
        "/backfill/start", data={"camera": "nope", "start": "2026-08-06T09:00"}
    )
    assert r.status_code == 400
    assert idle()


def test_a_file_camera_cannot_be_backfilled_from_an_nvr(tmp_path):
    client, _, _ = build(
        tmp_path,
        [unifi("front"), CameraConfig(id="clip", adapter="file", source="x")],
    )
    r = client.post(
        "/backfill/start", data={"camera": "clip", "start": "2026-08-06T09:00"}
    )
    assert r.status_code == 400
    assert idle()


def test_a_malformed_range_is_rejected_with_a_reason(tmp_path):
    client, _, _ = build(tmp_path, [unifi("front")])
    assert client.post("/backfill/start", data={"camera": "front"}).status_code == 400
    backwards = client.post(
        "/backfill/start",
        data={
            "camera": "front",
            "start": "2026-08-06T10:00",
            "end": "2026-08-06T09:00",
        },
    )
    assert backwards.status_code == 400
    assert "must be after" in backwards.text


def test_parse_range_defaults_the_end_to_now():
    start, end = parse_range("2026-08-06T09:00", "")
    assert start.tzinfo is not None
    assert end > datetime.now(timezone.utc) - timedelta(minutes=1)
    with pytest.raises(ValueError):
        parse_range("", "")
    with pytest.raises(ValueError):
        parse_range("yesterday", "")


# -- the re-run is a visible no-op ----------------------------------------


def test_a_covered_range_reports_itself_as_a_no_op(tmp_path, monkeypatch):
    """`UniqueConstraint(camera_id, external_id)` already guarantees the
    behaviour; an operator who cannot see it avoids the button."""
    seen: dict = {}
    stub_run(monkeypatch, seen)
    client, _, _ = build(
        tmp_path, [unifi("front")], clips=[clip("front", "a", "done")]
    )
    client.post(
        "/backfill/start", data={"camera": "front", "start": "2026-08-06T09:00"}
    )
    assert wait_for(idle)
    body = client.get("/backfill").text
    assert "already covered" in body
    assert "3 recording window(s)" in body
    assert "no event was created twice" in body


def test_the_run_heartbeats_an_operation_run(tmp_path, monkeypatch):
    """A backfill started here must be watchable from another terminal,
    like one started in a shell."""
    from siteloom.store import OperationRun

    seen: dict = {}
    stub_run(monkeypatch, seen)
    client, Session, _ = build(tmp_path, [unifi("front")])
    client.post(
        "/backfill/start", data={"camera": "front", "start": "2026-08-06T09:00"}
    )
    assert wait_for(idle)
    with Session() as s:
        run = s.scalars(select(OperationRun)).one()
        assert run.kind == "backfill-unifi"
        assert run.target == "front"
        # Rebuilt from the whole invocation, so a shell can continue it.
        assert run.resume_command.startswith("siteloom backfill-unifi front --start")


def test_the_resume_command_keeps_the_sweep_and_the_retry(tmp_path, monkeypatch):
    """A resume command missing --chunk-minutes resumes a different job:
    motion windows instead of the full sweep that was asked for."""
    from siteloom.store import OperationRun

    seen: dict = {}
    stub_run(monkeypatch, seen)
    client, Session, _ = build(tmp_path, [unifi("front")])
    client.post(
        "/backfill/start",
        data={
            "camera": "front",
            "start": "2026-08-06T09:00",
            "mode": "sweep",
            "chunk_minutes": "15",
            "retry_failed": "1",
        },
    )
    assert wait_for(idle)
    with Session() as s:
        command = s.scalars(select(OperationRun)).one().resume_command
    assert "--chunk-minutes 15" in command
    assert "--retry-failed" in command


# -- gating ----------------------------------------------------------------


def test_starting_a_backfill_needs_admin_but_looking_does_not():
    # Watching an import is not reading personal data, so the read floor
    # is the bottom rung and only the start is admin (CLD-31).
    assert required_role("GET", "/backfill") == "restricted"
    assert required_role("POST", "/backfill/start") == "admin"


def add_user(Session, username, role):
    with Session() as s:
        s.add(
            User(
                username=username,
                password_hash=hash_password("hunter2!"),
                role=role,
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
        s.commit()


def login(client, username):
    r = client.post(
        "/login", data={"username": username, "password": "hunter2!", "next": "/"}
    )
    assert r.status_code == 303
    return client


def test_an_editor_may_watch_a_backfill_but_not_start_one(tmp_path):
    """It downloads from the NVR and writes events — reconfiguration,
    not judgment."""
    client, Session, _ = build(tmp_path, [unifi("front")])
    add_user(Session, "eddy", "edit")
    login(client, "eddy")
    assert client.get("/backfill").status_code == 200
    r = client.post(
        "/backfill/start", data={"camera": "front", "start": "2026-08-06T09:00"}
    )
    assert r.status_code == 403
    assert idle()


def test_an_admin_start_is_audited(tmp_path, monkeypatch):
    seen: dict = {}
    stub_run(monkeypatch, seen)
    client, Session, _ = build(tmp_path, [unifi("front")])
    add_user(Session, "root", "admin")
    login(client, "root")
    r = client.post(
        "/backfill/start", data={"camera": "front", "start": "2026-08-06T09:00"}
    )
    assert r.status_code == 303
    assert wait_for(idle)
    with Session() as s:
        rows = s.scalars(select(AuditLog)).all()
        assert ("root", "/backfill/start") in {(r.username, r.path) for r in rows}


def test_the_screen_is_reachable_from_the_sidebar(tmp_path):
    """As a tab of the Jobs row now — the link is in the strip, and the
    Jobs entry is the one lit while here."""
    client, _, _ = build(tmp_path, [unifi("front")])
    body = client.get("/backfill").text
    assert 'class="on" href="/backfill"' in body
    assert '<span class="sl-label">Jobs</span>' in body
    assert body.count("sl-nav-item active") == 1
