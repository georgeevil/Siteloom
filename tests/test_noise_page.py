"""The /noise screen's empty states (CLD-26).

An empty table under "No noise events recorded" reads as "it has been
quiet here". On a site of UniFi cameras that conclusion is wrong and
unfalsifiable: audio cannot reach AudioModule on a live stream at all, so
the table cannot fill however loud it gets. The page has to distinguish
"nothing loud happened" from "nothing here can ever report".
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import NoiseEvent, get_session, init_db, make_engine
from siteloom.web.app import create_app

TS = datetime(2026, 8, 5, 22, 0)


def build(cameras, seed_noise=False, tmp_path=None, audio=True):
    config = SiteConfig(
        site_id="t",
        cameras=cameras,
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/n.db", media_dir=str(tmp_path / "m")
        ),
    )
    config.identity.enabled = False
    config.audio.enabled = audio
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    if seed_noise:
        with Session() as s:
            s.add(NoiseEvent(
                camera_id=cameras[0].id, start=TS, end=TS + timedelta(minutes=3),
                peak_db=-6.5, mean_db=-14.2,
            ))
            s.commit()
    return TestClient(create_app(config))


def test_a_live_only_site_says_nothing_can_report(tmp_path):
    client = build([CameraConfig(id="front", adapter="unifi", source="x",
                                 modules=["detection", "audio"])],
                   tmp_path=tmp_path)
    body = client.get("/noise").text
    assert "No camera on this site can record a noise event" in body
    assert "backfill-unifi" in body
    # The misleading reading this page exists to avoid.
    assert "No noise events recorded" not in body


def test_a_file_camera_gets_the_ordinary_quiet_message(tmp_path):
    """Here the table really is empty because nothing was loud enough."""
    client = build([CameraConfig(id="clip", adapter="file", source="x",
                                 modules=["detection", "audio"])],
                   tmp_path=tmp_path)
    body = client.get("/noise").text
    assert "Nothing loud enough yet" in body
    assert "can record a noise event" not in body


def test_audio_enabled_on_a_live_camera_is_called_out(tmp_path):
    """The setting is accepted and then silently ignored, which is worth
    saying on the page that is supposed to show its output."""
    cam = CameraConfig(id="front", adapter="unifi", source="x",
                       modules=["detection", "audio"])
    client = build([cam], tmp_path=tmp_path)
    body = client.get("/noise").text
    assert "having no effect" in body
    assert "front" in body


def test_no_warning_when_audio_is_enabled_on_a_camera_that_can_deliver(tmp_path):
    cam = CameraConfig(id="clip", adapter="file", source="x",
                       modules=["detection", "audio"])
    client = build([cam], tmp_path=tmp_path)
    assert "having no effect" not in client.get("/noise").text


def test_events_render_when_there_are_some(tmp_path):
    client = build([CameraConfig(id="clip", adapter="file", source="x",
                                 modules=["detection", "audio"])],
                   seed_noise=True, tmp_path=tmp_path)
    body = client.get("/noise").text
    assert "-6.5 dB" in body
    assert "3.0 min" in body
    # No empty state alongside real rows.
    assert "Nothing loud enough yet" not in body
    assert "can record a noise event" not in body


def test_audio_off_says_so_rather_than_explaining_live_streams(tmp_path):
    """The live-stream limitation is true but irrelevant when the real
    reason is that nobody turned audio on. Explaining the wrong one is
    the same failure this page exists to fix."""
    client = build([CameraConfig(id="clip", adapter="file", source="x")],
                   tmp_path=tmp_path, audio=False)
    body = client.get("/noise").text
    assert "Noise detection is off" in body
    assert "can record a noise event" not in body


def test_audio_on_but_no_camera_lists_it(tmp_path):
    client = build([CameraConfig(id="clip", adapter="file", source="x")],
                   tmp_path=tmp_path, audio=True)
    body = client.get("/noise").text
    assert "no camera is using it" in body
    assert "Noise detection is off" not in body
