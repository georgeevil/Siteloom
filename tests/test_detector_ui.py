"""The tuning lab's web surface (CLD-101/102/106).

Trials themselves are exercised in test_tuning.py with a stub module;
here the contract under test is the screen's: settings parse whole
before anything runs, applies write minimal overrides and snapshot
first, copy is explicit about sample_fps, revert restores, and the
mutations sit on the admin floor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from siteloom.config import (
    CameraConfig,
    DetectionOverride,
    SiteConfig,
    StorageConfig,
    load_config,
    save_config,
)
from siteloom.web.app import create_app
from siteloom.web.auth import required_role


@pytest.fixture
def env(tmp_path):
    config = SiteConfig(
        site_id="t",
        cameras=[
            CameraConfig(id="cam-a", adapter="file", source="x"),
            CameraConfig(id="cam-b", adapter="file", source="y",
                         sample_fps=8.0),
        ],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/t.db", media_dir=str(tmp_path / "m")
        ),
    )
    config.identity.enabled = False
    config.identity.vector_db_path = str(tmp_path / "v")
    # A real file behind the config, so saves and snapshots are real.
    path = tmp_path / "site.yaml"
    save_config(config, path)
    config = load_config(path)
    config.identity.enabled = False
    client = TestClient(create_app(config))
    client.config = config
    client.config_path = path
    return client


def fake_run(client, run_id="20260826-000000-cam-a", **settings):
    run_dir = Path(client.config.storage.media_dir) / "tuning" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(json.dumps({
        "source": "clip.mp4", "sample_fps": 5.0, "frames": 10,
        # Defaults mirror DetectionConfig's own, so a fake run with no
        # kwargs is "identical to the site" by construction.
        "settings": {
            "model": "yolo11n.pt", "confidence": 0.4,
            "class_confidence": {}, "tracker": {"fuse_score": False},
            "track_buffer_s": 4.0, **settings,
        },
        "groups": {}, "scene": {"classes": {}, "ir": False,
                                "saturation_mean": 60.0, "luma_mean": 100.0},
        "moments": [], "video": None,
    }))
    return run_id


def test_the_page_renders_with_effective_settings(env):
    env.config.cameras[0].detection = DetectionOverride(confidence=0.7)
    body = env.get("/detector").text
    assert "Effective settings per camera" in body
    assert "0.7" in body


def test_mutations_sit_on_the_admin_floor():
    assert required_role("POST", "/detector/run") == "admin"
    assert required_role("POST", "/detector/apply") == "admin"
    assert required_role("GET", "/detector") == "restricted"


def test_a_broken_settings_form_runs_nothing(env):
    resp = env.post("/detector/run", data={
        "source_kind": "clip", "clip": "nope.mp4", "confidence": "eleven",
    })
    assert resp.status_code == 400
    assert "confidence" in resp.text


def test_an_unknown_clip_is_refused(env):
    resp = env.post("/detector/run", data={
        "source_kind": "clip", "clip": "../../etc/passwd",
    })
    assert resp.status_code == 400


def test_apply_to_camera_writes_the_minimal_override(env):
    """Only what differs from the site lands in the override — a
    restated site value would stop following later site changes."""
    run_id = fake_run(env, confidence=0.65, tracker={
        "fuse_score": False, "match_thresh": 0.9,
    })
    resp = env.post("/detector/apply",
                    data={"run_id": run_id, "target": "cam-a"},
                    follow_redirects=False)
    assert resp.status_code == 303
    override = env.config.cameras[0].detection
    assert override.confidence == 0.65
    assert override.tracker == {"match_thresh": 0.9}  # fuse_score = site value
    assert override.model is None                     # matched the site
    # ... and it survived to disk.
    on_disk = load_config(env.config_path)
    assert on_disk.cameras[0].detection.confidence == 0.65


def test_apply_matching_the_site_leaves_no_override(env):
    run_id = fake_run(env)  # settings identical to site defaults
    env.post("/detector/apply", data={"run_id": run_id, "target": "cam-a"},
             follow_redirects=False)
    assert env.config.cameras[0].detection is None


def test_apply_snapshots_first_and_revert_restores(env):
    run_id = fake_run(env, confidence=0.9)
    env.post("/detector/apply", data={"run_id": run_id, "target": "site"},
             follow_redirects=False)
    assert env.config.detection.confidence == 0.9
    history = env.config_path.parent / "config-history"
    snapshots = sorted(p.name for p in history.glob("site-*.yaml"))
    assert snapshots
    resp = env.post("/detector/revert", data={"snapshot": snapshots[-1]},
                    follow_redirects=False)
    assert resp.status_code == 303
    assert env.config.detection.confidence == 0.4  # live config restored
    assert yaml.safe_load(env.config_path.read_text())["detection"][
        "confidence"
    ] == 0.4  # and the file


def test_copy_is_explicit_about_sample_fps(env):
    env.config.cameras[0].detection = DetectionOverride(confidence=0.7)
    env.post("/detector/copy", data={
        "from_camera": "cam-a", "to_camera": "cam-b",
    }, follow_redirects=False)
    assert env.config.cameras[1].detection.confidence == 0.7
    assert env.config.cameras[1].sample_fps == 8.0  # scene property kept
    env.post("/detector/copy", data={
        "from_camera": "cam-a", "to_camera": "cam-b",
        "include_sample_fps": "1",
    }, follow_redirects=False)
    assert env.config.cameras[1].sample_fps == env.config.cameras[0].sample_fps


def test_reset_camera_drops_the_override(env):
    env.config.cameras[0].detection = DetectionOverride(confidence=0.7)
    env.post("/detector/reset-camera", data={"camera": "cam-a"},
             follow_redirects=False)
    assert env.config.cameras[0].detection is None


def test_run_detail_refuses_cross_source_comparison(env):
    a = fake_run(env, run_id="20260826-000001-a")
    b_dir = Path(env.config.storage.media_dir) / "tuning" / "20260826-000002-b"
    b_dir.mkdir(parents=True)
    report = json.loads(
        (Path(env.config.storage.media_dir) / "tuning" / a / "report.json")
        .read_text()
    )
    report["source"] = "different.mp4"
    (b_dir / "report.json").write_text(json.dumps(report))
    resp = env.get(f"/detector/runs/{a}?versus=20260826-000002-b")
    assert resp.status_code == 400
    assert "same source" in resp.text
