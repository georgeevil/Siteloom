"""Identities and Classes console screens (CLD-22, CLD-23)."""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import Identity, get_session, init_db, make_engine
from siteloom.web.app import create_app
from siteloom.web.library_routes import _class_rows

TS = datetime(2026, 8, 5, 12, 0)


@pytest.fixture
def client(tmp_path):
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="c", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/w.db", media_dir=str(tmp_path / "m")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add_all(
            [
                # named and enrolled
                Identity(
                    identifier_key="face",
                    class_name="person",
                    label="Ana",
                    first_seen=TS,
                    last_seen=TS,
                    vector_count=7,
                ),
                # named but with nothing in the vector store
                Identity(
                    identifier_key="face",
                    class_name="person",
                    label="Bo",
                    first_seen=TS,
                    last_seen=TS,
                    vector_count=0,
                ),
                # never labelled
                Identity(
                    identifier_key="vehicle",
                    class_name="car",
                    first_seen=TS,
                    last_seen=TS,
                    vector_count=4,
                ),
            ]
        )
        s.commit()
    return TestClient(create_app(config))


def cards(client, **params) -> int:
    r = client.get("/identities", params=params)
    assert r.status_code == 200, r.text[:400]
    return r.text.count('class="idcard')


def test_identity_chips_filter(client):
    assert cards(client) == 3
    assert cards(client, unlabeled=1) == 1
    assert cards(client, identifier="face") == 2


def test_no_vectors_chip_finds_names_recognition_cannot_match(client):
    """A labelled identity with no embeddings is invisible to recognition;
    an unlabelled one is a different thing and must not be swept in."""
    r = client.get("/identities", params={"unenrolled": 1})
    assert r.text.count('class="idcard') == 1
    assert "Bo" in r.text
    assert "Ana" not in r.text


def test_identity_rail_warns_when_a_name_has_no_vectors(client):
    with_vectors = client.get("/identities", params={"selected": 1}).text
    without = client.get("/identities", params={"selected": 2}).text
    assert "recognition cannot" not in with_vectors
    assert "recognition cannot" in without
    assert "Enrolled samples · 7" in with_vectors


def test_identity_chips_preserve_each_other(client):
    """Ticking one chip must not silently drop the others."""
    r = client.get("/identities", params={"identifier": "face", "unlabeled": 1})
    assert "identifier=face" in r.text and "unlabeled=1" in r.text


def test_class_rows_active_mirrors_configured_classes():
    """Active is not a new flag — it is membership of detection.classes."""
    config = SiteConfig(site_id="t")
    rows = _class_rows(config, seen={"person": 12})
    active = {r["name"] for r in rows if r["active"]}
    assert active == set(config.detection.classes)
    assert next(r for r in rows if r["name"] == "person")["samples"] == 12
    # A catalog class that is not configured is offered but off.
    assert next(r for r in rows if r["name"] == "laptop")["active"] is False


def test_class_rows_attribute_the_identifier_that_applies():
    config = SiteConfig(site_id="t")
    rows = {r["name"]: r for r in _class_rows(config, seen={})}
    assert rows["person"]["identifier"] == "face"
    assert rows["person"]["threshold"] == config.identity.identifiers["face"].threshold
    # No identifier configured, but auto-add means it still gets one later.
    assert rows["laptop"]["identifier"] is None
    assert rows["laptop"]["auto"] is config.identity.auto_add_classes


def test_class_colours_are_stable_per_name():
    """A class must keep its swatch when another is added or removed."""
    a = {r["name"]: r["hue"] for r in _class_rows(SiteConfig(site_id="t"), seen={})}
    config = SiteConfig(site_id="t")
    config.detection.classes = ["zebra"] + list(config.detection.classes)
    b = {r["name"]: r["hue"] for r in _class_rows(config, seen={})}
    assert all(a[name] == b[name] for name in a)


def test_classes_page_never_claims_a_precision_it_cannot_measure(client):
    body = client.get("/classes").text
    assert body.count(">Precision<") == 0
    # And it must not imply sub-classes involve a training run.
    assert "there is no training run" in body


def test_classes_page_shows_event_rules(client):
    r = client.get("/classes")
    assert r.status_code == 200
    assert "Event rules" in r.text
    assert 'id="er-min_detections"' in r.text
    assert 'id="er-identify_only_significant"' in r.text


def test_event_rules_post_updates_live_config_and_yaml(tmp_path):
    from siteloom.config import save_config

    config = SiteConfig(
        site_id="t",
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/er.db", media_dir=str(tmp_path / "m")
        ),
    )
    path = tmp_path / "site.yaml"
    save_config(config, path)  # gives the config a _source_path
    from siteloom.config import load_config

    config = load_config(path)
    client = TestClient(create_app(config))
    r = client.post(
        "/classes/events",
        json={
            "min_detections": 5,
            "min_confidence": 0.6,
            "stitch_gap_s": 8,
            "identify_only_significant": False,
        },
    )
    assert r.status_code == 200
    assert r.json()["written_to"] == str(path)
    # Live object updated...
    assert config.events.min_detections == 5
    assert config.events.identify_only_significant is False
    # ...and the YAML round-trips for the next process start.
    again = load_config(path)
    assert again.events.min_detections == 5
    assert again.events.min_confidence == 0.6
    assert again.events.stitch_gap_s == 8.0
    assert again.events.identify_only_significant is False


def test_class_rows_carry_detection_confidence():
    from siteloom.config import DetectionConfig

    config = SiteConfig(
        site_id="t",
        detection=DetectionConfig(confidence=0.4, class_confidence={"dog": 0.7}),
    )
    rows = {r["name"]: r for r in _class_rows(config, seen={})}
    assert rows["person"]["det_conf"] == 0.4
    assert rows["person"]["det_conf_overridden"] is False
    assert rows["dog"]["det_conf"] == 0.7
    assert rows["dog"]["det_conf_overridden"] is True


def test_class_confidence_post_updates_config_and_yaml(tmp_path):
    from siteloom.config import load_config, save_config

    config = SiteConfig(
        site_id="t",
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/cc.db", media_dir=str(tmp_path / "m")
        ),
    )
    path = tmp_path / "site.yaml"
    save_config(config, path)
    config = load_config(path)
    client = TestClient(create_app(config))
    r = client.post(
        "/classes/detection",
        json={"classes": ["person", "dog"], "class_confidence": {"dog": 0.75}},
    )
    assert r.status_code == 200
    assert config.detection.class_confidence == {"dog": 0.75}
    again = load_config(path)
    assert again.detection.class_confidence == {"dog": 0.75}
    # An empty map clears every override.
    client.post("/classes/detection", json={"class_confidence": {}})
    assert load_config(path).detection.class_confidence == {}


def test_classes_page_renders_editable_confidence(client):
    r = client.get("/classes")
    assert r.status_code == 200
    assert 'class="dconf' in r.text
    assert "Det confidence" in r.text
