"""Takeout sidecar parsing and matching.

Face assignment itself is covered by test_takeout_assignment.py with a
stub face detector; here we pin down the messy real-world file naming.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from siteloom.library.takeout import find_sidecar, index_sidecars, parse_sidecar


def write_sidecar(directory: Path, filename: str, title: str, people=None, ts=None):
    payload = {
        "title": title,
        "description": "",
        "photoTakenTime": {"timestamp": str(ts or 1700000000)},
        "geoData": {"latitude": 0.0, "longitude": 0.0},
    }
    if people:
        payload["people"] = [{"name": n} for n in people]
    path = directory / filename
    path.write_text(json.dumps(payload))
    return path


def test_parse_sidecar_extracts_people_and_time(tmp_path):
    path = write_sidecar(
        tmp_path, "IMG_1.jpg.supplemental-metadata.json", "IMG_1.jpg",
        people=["Jane Doe", "John Smith"], ts=1600000000,
    )
    sidecar = parse_sidecar(path)
    assert sidecar is not None
    assert sidecar.title == "IMG_1.jpg"
    assert sidecar.people == ["Jane Doe", "John Smith"]
    assert sidecar.taken_at == datetime(2020, 9, 13, 12, 26, 40)


def test_parse_sidecar_without_people(tmp_path):
    path = write_sidecar(tmp_path, "a.jpg.json", "a.jpg")
    assert parse_sidecar(path).people == []


def test_parse_ignores_non_media_json(tmp_path):
    (tmp_path / "print-subscriptions.json").write_text('{"foo": 1}')
    assert parse_sidecar(tmp_path / "print-subscriptions.json") is None


def test_parse_ignores_malformed_json(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")
    assert parse_sidecar(tmp_path / "broken.json") is None


def test_match_standard_suffix(tmp_path):
    write_sidecar(
        tmp_path, "IMG_1.jpg.supplemental-metadata.json", "IMG_1.jpg", ["Ann"]
    )
    index = index_sidecars(tmp_path)
    sidecar = find_sidecar(Path("IMG_1.jpg"), index)
    assert sidecar is not None and sidecar.people == ["Ann"]


def test_match_legacy_json_suffix(tmp_path):
    write_sidecar(tmp_path, "IMG_2.jpg.json", "IMG_2.jpg", ["Bob"])
    index = index_sidecars(tmp_path)
    assert find_sidecar(Path("IMG_2.jpg"), index).people == ["Bob"]


def test_match_counter_migration(tmp_path):
    """Takeout moves the (1) counter into the sidecar name:
    media 'IMG_3(1).jpg' pairs with 'IMG_3.jpg(1).supplemental-metadata.json'."""
    write_sidecar(
        tmp_path,
        "IMG_3.jpg(1).supplemental-metadata.json",
        "IMG_3.jpg",
        ["Cara"],
    )
    index = index_sidecars(tmp_path)
    assert find_sidecar(Path("IMG_3(1).jpg"), index).people == ["Cara"]


def test_match_truncated_sidecar_name(tmp_path):
    """Long filenames get truncated in the sidecar name; the title field
    is authoritative and prefix matching is the fallback."""
    long_name = "a-very-long-original-filename-from-a-phone-camera-2026.jpg"
    write_sidecar(
        tmp_path,
        "a-very-long-original-filename-from-a-ph.supplemental-metadata.json",
        long_name,
        ["Dan"],
    )
    index = index_sidecars(tmp_path)
    assert find_sidecar(Path(long_name), index).people == ["Dan"]


def test_no_sidecar_returns_none(tmp_path):
    write_sidecar(tmp_path, "other.jpg.json", "other.jpg")
    assert find_sidecar(Path("zzz-unrelated.heic"), index_sidecars(tmp_path)) is None
