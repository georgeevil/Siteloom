"""Identities and Classes console screens (CLD-22, CLD-23)."""

from __future__ import annotations

import re
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
    client = TestClient(create_app(config))
    # Tests that need to seed rows after the app is built reach the same
    # database through here rather than rebuilding the config.
    client.sessionmaker = Session
    return client


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


def precision_cells(client) -> list[str]:
    """Just the Precision column. The whole page is full of "100%" in its
    CSS, so an assertion about what the column claims has to read the
    column."""
    body = client.get("/classes").text
    return [
        " ".join(re.sub(r"<[^>]+>", " ", cell).split())
        for cell in re.findall(r'<div class="prec">(.*?)</div>\s*<div>', body, re.S)
    ]


def test_classes_page_never_claims_a_precision_it_cannot_measure(client):
    """The column exists now that /stats supplies it (CLD-87). An
    identifier that has never claimed anything has no precision to show —
    and must not borrow a reassuring number from the empty set."""
    body = client.get("/classes").text
    assert body.count(">Precision<") == 1
    cells = precision_cells(client)
    assert cells and all(c == "—" for c in cells), cells
    # And it must not imply sub-classes involve a training run.
    assert "there is no training run" in body


def _claim(session, ident, verdict):
    from siteloom.store import Event, EventIdentity

    ev = Event(
        camera_id="c", class_name="person",
        first_seen=TS, last_seen=TS, detection_count=1,
    )
    session.add(ev)
    session.flush()
    session.add(EventIdentity(
        event_id=ev.id, identity_id=ident.id, identifier_key="face",
        matched_by="visual", similarity=0.4, verdict=verdict,
    ))


def test_classes_precision_says_so_when_nothing_was_judged(client):
    """Claims exist but no operator judged them. That is unknown, not
    perfect — the exact reading a bare 100% would invite."""
    from siteloom.store import Camera

    with client.sessionmaker() as s:
        s.add(Camera(id="c", site_id="t", name="C"))
        ident = s.query(Identity).filter_by(label="Ana").one()
        _claim(s, ident, None)
        _claim(s, ident, None)
        s.commit()

    cells = precision_cells(client)
    assert "no verdicts" in cells, cells
    assert not any("%" in c for c in cells), cells


def test_classes_precision_carries_its_denominator(client):
    """One wrong of two reviewed is 50% — but 50% alone is the misleading
    number, so the reviewed count rides along with it."""
    from siteloom.store import Camera

    with client.sessionmaker() as s:
        s.add(Camera(id="c", site_id="t", name="C"))
        ident = s.query(Identity).filter_by(label="Ana").one()
        _claim(s, ident, "confirmed")
        _claim(s, ident, "wrong")
        s.commit()

    assert "50% of 2" in precision_cells(client), precision_cells(client)




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


# -- CLD-38: identity thresholds are editable, and previewable ----------


def test_identifier_rows_carry_their_own_algo_scale():
    """Face and generic thresholds are not on the same scale, so the
    control for each is bounded by its own algorithm's distribution."""
    from siteloom.web.library_routes import _identifier_rows

    config = SiteConfig(site_id="t")
    rows = {r["key"]: r for r in _identifier_rows(config)}
    assert rows["face"]["threshold"] == 0.36
    # Different worlds: the face track is centred where face scores live
    # and stops short of where generic ones start being meaningful.
    assert rows["face"]["min"] < rows["person"]["min"]
    assert rows["face"]["max"] < rows["person"]["max"]
    assert rows["face"]["typical"] < rows["face"]["max"] < rows["person"]["typical"]
    assert rows["person"]["typical"] == rows["vehicle"]["typical"]  # same algo
    assert rows["vehicle"]["applies_to"] == ["car", "truck", "bus", "motorcycle"]
    assert rows["vehicle"]["plate_ocr"] is True


def test_identifier_rows_name_the_cameras_that_override_them():
    from siteloom.config import CameraIdentityOverride
    from siteloom.web.library_routes import _identifier_rows

    config = SiteConfig(
        site_id="t",
        cameras=[
            CameraConfig(
                id="doorway",
                source="x",
                identity=CameraIdentityOverride(thresholds={"face": 0.44}),
            ),
            CameraConfig(id="quiet", source="x"),
        ],
    )
    rows = {r["key"]: r for r in _identifier_rows(config)}
    assert rows["face"]["camera_overrides"] == [
        {"camera": "doorway", "threshold": 0.44}
    ]
    assert rows["person"]["camera_overrides"] == []


def test_classes_page_renders_threshold_sliders_not_only_bars(client):
    r = client.get("/classes")
    assert r.status_code == 200
    assert 'class="ithr" type="range" data-key="face"' in r.text
    assert 'data-key="vehicle"' in r.text
    # The page must not imply one scale fits every identifier.
    assert "not comparable with each other" in r.text
    # ...and must stay honest about when a saved value reaches ingest.
    assert "separate\n  processes" in r.text or "separate processes" in r.text


def test_identifier_threshold_post_updates_config_and_yaml(tmp_path):
    from siteloom.config import load_config, save_config

    config = SiteConfig(
        site_id="t",
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/it.db", media_dir=str(tmp_path / "m")
        ),
    )
    path = tmp_path / "site.yaml"
    save_config(config, path)
    config = load_config(path)
    client = TestClient(create_app(config))
    r = client.post(
        "/classes/detection",
        json={"identifiers": {"face": {"threshold": 0.42}}},
    )
    assert r.status_code == 200
    assert config.identity.identifiers["face"].threshold == 0.42
    again = load_config(path)
    assert again.identity.identifiers["face"].threshold == 0.42
    # Editing one identifier must not disturb another's scale.
    assert again.identity.identifiers["person"].threshold == 0.80


def test_identifier_threshold_rejects_a_value_off_the_similarity_scale(client):
    r = client.post(
        "/classes/detection", json={"identifiers": {"face": {"threshold": 42}}}
    )
    assert r.status_code == 400
    assert (
        client.post("/classes/detection", json={"auto_add_threshold": 42}).status_code
        == 400
    )


def test_auto_added_classes_get_a_threshold_control(client):
    """A class with no identifier of its own still gets one on first
    sighting — its threshold must be tunable too, on the generic scale."""
    body = client.get("/classes").text
    assert 'id="auto-add-threshold"' in body
    assert client.post(
        "/classes/detection", json={"auto_add_threshold": 0.88}
    ).status_code == 200


def test_threshold_preview_counts_both_directions(tmp_path):
    """The dry run answers the two questions worth asking, from scores
    already recorded on EventIdentity (CLD-38)."""
    from siteloom.store import Event, EventIdentity

    config = SiteConfig(
        site_id="t",
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/tp.db", media_dir=str(tmp_path / "m")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(
            Identity(
                identifier_key="face", class_name="person", first_seen=TS, last_seen=TS
            )
        )
        # The other identifier's history belongs to its own identity: an
        # Identity row has one identifier_key, and one identity holds at
        # most one standing claim per event (CLD-133).
        s.add(
            Identity(
                identifier_key="vehicle", class_name="car", first_seen=TS, last_seen=TS
            )
        )
        s.flush()
        for i, (similarity, matched_by) in enumerate(
            [
                (0.30, "visual"),  # a match that 0.40 would break
                (0.55, "visual"),  # a match that survives
                (0.45, None),      # a near-miss 0.40 would have caught
                (0.20, None),      # a near-miss still below 0.40
                (0.01, "plate"),   # plates ignore the threshold entirely
            ]
        ):
            event = Event(
                camera_id="c",
                class_name="person",
                first_seen=TS,
                last_seen=TS,
                best_confidence=0.9,
            )
            s.add(event)
            s.flush()
            s.add(
                EventIdentity(
                    event_id=event.id,
                    identity_id=1,
                    identifier_key="face",
                    similarity=similarity,
                    matched_by=matched_by,
                )
            )
            # A different identifier's history must not leak in.
            s.add(
                EventIdentity(
                    event_id=event.id,
                    identity_id=2,
                    identifier_key="vehicle",
                    similarity=0.99,
                    matched_by="visual",
                )
            )
        s.commit()

    client = TestClient(create_app(config))
    body = client.get(
        "/classes/thresholds/preview", params={"identifier": "face", "threshold": 0.40}
    ).json()
    assert body["sampled"] == 5
    assert body["visual_matches"] == 2 and body["would_unmatch"] == 1
    assert body["near_misses"] == 2 and body["would_match"] == 1
    assert body["plate_matches"] == 1
    assert body["current"] == 0.36


def test_threshold_preview_validates_its_inputs(client):
    assert client.get(
        "/classes/thresholds/preview", params={"identifier": "nope", "threshold": 0.4}
    ).status_code == 404
    assert client.get(
        "/classes/thresholds/preview", params={"identifier": "face", "threshold": 7}
    ).status_code == 400
    # No history is a real answer, not an error.
    empty = client.get(
        "/classes/thresholds/preview", params={"identifier": "face", "threshold": 0.4}
    ).json()
    assert empty["sampled"] == 0


# -- CLD-39: per-camera overrides are visible, read-only ---------------


def test_camera_override_rows_report_only_what_is_set():
    from siteloom.config import CameraIdentityOverride, EventRulesOverride
    from siteloom.web.library_routes import _camera_override_rows

    config = SiteConfig(
        site_id="t",
        cameras=[
            CameraConfig(
                id="doorway",
                name="Doorway",
                source="x",
                events=EventRulesOverride(identify_min_crop_px=96),
                identity=CameraIdentityOverride(thresholds={"face": 0.44}),
            ),
            CameraConfig(id="quiet", source="x"),
        ],
    )
    rows = {r["id"]: r for r in _camera_override_rows(config)}
    assert rows["doorway"]["events"] == {"identify_min_crop_px": 96}
    assert rows["doorway"]["thresholds"] == {"face": 0.44}
    assert rows["quiet"]["count"] == 0


def test_classes_page_shows_per_camera_overrides_read_only(tmp_path):
    from siteloom.config import CameraIdentityOverride

    config = SiteConfig(
        site_id="t",
        cameras=[
            CameraConfig(
                id="doorway",
                source="x",
                identity=CameraIdentityOverride(thresholds={"face": 0.44}),
            )
        ],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/cam.db", media_dir=str(tmp_path / "m")
        ),
    )
    body = TestClient(create_app(config)).get("/classes").text
    assert "Per-camera overrides" in body
    assert "face threshold" in body and "0.44" in body
    # Read-only for now, and the page says where the values live.
    assert "cameras[].identity.thresholds" in body
