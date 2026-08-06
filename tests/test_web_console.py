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
