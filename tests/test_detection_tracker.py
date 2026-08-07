"""Tracker config materialization (no YOLO involved — the file is what
ultralytics consumes; its content is the contract)."""

from __future__ import annotations

import yaml

from siteloom.config import DetectionConfig
from siteloom.modules.detection import TRACKER_DEFAULTS, tracker_config_path


def test_default_disables_fuse_score():
    path = tracker_config_path(DetectionConfig())
    data = yaml.safe_load(path.read_text())
    # The one non-library default: fused cost breaks track confirmation
    # at our few-fps sampling rate (CLD-5).
    assert data["fuse_score"] is False
    assert data["tracker_type"] == "bytetrack"
    for key, value in TRACKER_DEFAULTS.items():
        if key != "fuse_score":
            assert data[key] == value


def test_overrides_merge_and_hash_to_distinct_files():
    default = tracker_config_path(DetectionConfig())
    tuned = tracker_config_path(
        DetectionConfig(tracker={"fuse_score": False, "match_thresh": 0.9})
    )
    assert tuned != default
    assert yaml.safe_load(tuned.read_text())["match_thresh"] == 0.9
    # Same config maps back to the same file.
    assert tracker_config_path(DetectionConfig()) == default
