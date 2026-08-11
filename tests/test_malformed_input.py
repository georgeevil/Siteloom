"""Console endpoints answer bad input instead of crashing (CLD-61/62).

Two failures used to look identical from a browser — a 500 — and neither
told the operator anything they could act on.

The first is malformed input. `/classes/detection` and `/classes/events`
patch the live site config and then write it to YAML, so an uncaught
`float("abc")` was not merely an ugly traceback: the fields before the
bad one had already been applied, in memory and on disk, with nothing to
say which. Every case here therefore asserts both halves — the status and
message the caller gets, *and* that nothing moved.

The second is the vector store. Embedded Qdrant allows one client per
path per machine, so "another process holds it" is the ordinary state
whenever ingest, a backfill or an index run is going — which is exactly
when someone is reviewing proposals. Merge and split already answer that
with an actionable 503; confirming a face proposal answered it with a
traceback.

The lock is simulated by making the shared-store lookup raise what a
real flock collision raises. A second live client would work here too
(merge and split do that) but needs a real Qdrant directory to fight
over, and the assertion is about how the refusal is phrased, not about
Qdrant.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from siteloom.config import (
    CameraConfig,
    IdentityConfig,
    SiteConfig,
    StorageConfig,
    load_config,
    save_config,
)
from siteloom.store import (
    Annotation,
    CustomClass,
    LibraryItem,
    LibrarySource,
    TrainingRun,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web.app import create_app

TS = datetime(2026, 8, 10, 12, 0)


@pytest.fixture
def env(tmp_path):
    """A console over a config with a YAML file behind it.

    The file matters: half-applied edits are only visible as damage once
    they are persisted, so the partial-mutation tests read it back.
    """
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="c", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/w.db", media_dir=str(tmp_path / "m")
        ),
        identity=IdentityConfig(vector_db_path=str(tmp_path / "vectors")),
    )
    path = tmp_path / "site.yaml"
    save_config(config, path)
    config = load_config(path)  # gives the live object a _source_path
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as session:
        session.add(
            LibrarySource(id=1, path=str(tmp_path), name="archive", added_at=TS)
        )
        session.add(
            LibraryItem(
                id=1,
                source_id=1,
                path=str(tmp_path / "a.jpg"),
                kind="image",
                status="indexed",
                mtime=TS,
            )
        )
        session.add(CustomClass(name="delivery-van", parent_class="car", created_at=TS))
        session.add(
            Annotation(
                id=1,
                item_id=1,
                bbox="[0.1,0.1,0.4,0.4]",
                class_name="face",
                confidence=0.9,
                source="import",
                proposed_name="Ana",
                proposal_basis="unambiguous",
                created_at=TS,
            )
        )
        # A crop with a file behind it: the classify path reads the image
        # before it builds anything, so this is what makes it reach the
        # vector store at all.
        crop = tmp_path / "crop.jpg"
        cv2.imwrite(str(crop), np.full((32, 32, 3), 90, dtype=np.uint8))
        session.add(
            Annotation(
                id=2,
                item_id=1,
                bbox="[0.5,0.5,0.9,0.9]",
                class_name="car",
                confidence=0.8,
                source="auto",
                crop_path=str(crop),
                created_at=TS,
            )
        )
        session.commit()
    return SimpleNamespace(
        client=TestClient(create_app(config)),
        Session=Session,
        config=config,
        path=path,
        tmp_path=tmp_path,
    )


def detail(response) -> str:
    return response.json()["detail"]


# -- a body that is not a body ---------------------------------------------


@pytest.mark.parametrize(
    "url",
    ["/classes/detection", "/classes/events", "/api/training/review"],
)
def test_a_body_that_is_not_json_is_named_as_such(env, url):
    r = env.client.post(url, content=b"not json at all")
    assert r.status_code == 400
    assert "must be JSON" in detail(r)


@pytest.mark.parametrize(
    "url",
    ["/classes/detection", "/classes/events", "/api/training/review"],
)
def test_a_json_array_where_an_object_belongs_is_refused(env, url):
    """`body.get(...)` on a list is an AttributeError, which is a 500 —
    and says nothing about the shape the endpoint wanted."""
    r = env.client.post(url, json=["classes"])
    assert r.status_code == 400
    assert "must be a JSON object" in detail(r)


# -- /classes/detection ----------------------------------------------------


def test_a_confidence_that_is_not_a_number_is_refused(env):
    r = env.client.post("/classes/detection", json={"confidence": "abc"})
    assert r.status_code == 400
    assert "confidence must be a confidence in 0..1" in detail(r)


def test_a_confidence_off_the_scale_is_refused_rather_than_stored(env):
    """A detector confidence of 9 admits nothing. Storing it is worse
    than refusing it: the site simply stops seeing anything."""
    r = env.client.post("/classes/detection", json={"confidence": 9})
    assert r.status_code == 400
    assert env.config.detection.confidence != 9
    assert load_config(env.path).detection.confidence != 9


def test_a_true_in_a_numeric_field_is_not_read_as_one(env):
    """`float(True)` is 1.0 — a checkbox posted into a threshold would
    otherwise arrive as a plausible setting."""
    r = env.client.post("/classes/detection", json={"confidence": True})
    assert r.status_code == 400


def test_a_bad_per_class_confidence_names_the_class(env):
    r = env.client.post(
        "/classes/detection", json={"class_confidence": {"dog": "loud"}}
    )
    assert r.status_code == 400
    assert "class_confidence[dog]" in detail(r)


def test_an_empty_class_list_is_refused_not_ignored(env):
    """It used to be dropped silently, so "I turned everything off" was
    answered with 200 and the old list still in force."""
    r = env.client.post("/classes/detection", json={"classes": []})
    assert r.status_code == 400
    assert "at least one class" in detail(r)


def test_a_class_list_that_is_not_names_is_refused(env):
    r = env.client.post("/classes/detection", json={"classes": ["person", 7]})
    assert r.status_code == 400
    assert "every entry of classes" in detail(r)


def test_identifier_settings_that_are_not_an_object_are_refused(env):
    """`"threshold" in "0.4"` is False, so this whole edit used to be
    dropped and reported as saved."""
    r = env.client.post("/classes/detection", json={"identifiers": {"face": "0.4"}})
    assert r.status_code == 400
    assert "identifiers[face] must be a JSON object" in detail(r)


def test_an_unknown_identifier_is_named_not_skipped(env):
    r = env.client.post(
        "/classes/detection", json={"identifiers": {"nose": {"threshold": 0.4}}}
    )
    assert r.status_code == 400
    assert "unknown identifier 'nose'" in detail(r)
    # ...and says which ones exist, so the next attempt can be right.
    assert "face" in detail(r)


def test_applies_to_as_a_bare_string_is_refused(env):
    """Iterating "car" yields three one-character class names, and the
    identifier would then apply to none of them."""
    r = env.client.post(
        "/classes/detection", json={"identifiers": {"vehicle": {"applies_to": "car"}}}
    )
    assert r.status_code == 400
    assert env.config.identity.identifiers["vehicle"].applies_to != ["c", "a", "r"]


def test_a_string_in_a_boolean_field_is_refused(env):
    """`bool("false")` is True."""
    r = env.client.post(
        "/classes/detection", json={"identifiers": {"vehicle": {"plate_ocr": "false"}}}
    )
    assert r.status_code == 400
    assert "plate_ocr must be true or false" in detail(r)
    assert env.config.identity.identifiers["vehicle"].plate_ocr is True


def test_a_setting_this_endpoint_does_not_write_is_named(env):
    r = env.client.post("/classes/detection", json={"tracker": {"fuse_score": True}})
    assert r.status_code == 400
    assert "unknown detection setting: tracker" in detail(r)


def test_a_refused_body_applies_none_of_its_valid_fields(env):
    """The whole point of parsing before mutating: `classes` is valid and
    comes first, `confidence` is nonsense and comes second. One 400 must
    not leave a site tracking a new class list nobody confirmed."""
    before = list(env.config.detection.classes)
    r = env.client.post(
        "/classes/detection",
        json={"classes": ["cat"], "confidence": "abc"},
    )
    assert r.status_code == 400
    assert env.config.detection.classes == before
    assert load_config(env.path).detection.classes == before


def test_a_valid_body_still_applies(env):
    """The gate must not have closed on the ordinary case."""
    r = env.client.post(
        "/classes/detection",
        json={
            "classes": ["person", "car"],
            "confidence": 0.45,
            "class_confidence": {"car": 0.7},
            "identifiers": {"face": {"threshold": 0.4}},
            "auto_add_classes": True,
            "auto_add_threshold": 0.9,
        },
    )
    assert r.status_code == 200
    again = load_config(env.path)
    assert again.detection.classes == ["person", "car"]
    assert again.detection.confidence == 0.45
    assert again.detection.class_confidence == {"car": 0.7}
    assert again.identity.identifiers["face"].threshold == 0.4
    assert again.identity.auto_add_threshold == 0.9


# -- /classes/events -------------------------------------------------------


def test_an_event_rule_that_is_not_a_number_is_refused(env):
    r = env.client.post("/classes/events", json={"min_detections": "three"})
    assert r.status_code == 400
    assert "min_detections must be a whole number" in detail(r)


def test_a_null_from_an_empty_field_is_refused(env):
    """The page sends `Number(input.value)`, and `JSON.stringify(NaN)` is
    `null` — so this is what a typo in the form actually posts."""
    r = env.client.post("/classes/events", json={"min_confidence": None})
    assert r.status_code == 400


def test_event_rules_are_bounded_by_what_they_mean(env):
    """Each of these parses as a number and means nothing: a count of
    detections cannot be negative, and an IoU is a fraction. Zero is
    allowed where it means "gate off", so only the meaningless is here."""
    for field, value in (
        ("min_detections", -2),
        ("stitch_min_iou", 4),
        ("min_confidence", 7),
        ("min_duration_s", -1),
        ("identify_min_crop_px", -8),
    ):
        r = env.client.post("/classes/events", json={field: value})
        assert r.status_code == 400, field
        assert getattr(load_config(env.path).events, field) != value


def test_a_boolean_rule_rejects_a_string(env):
    r = env.client.post("/classes/events", json={"identify_only_significant": "no"})
    assert r.status_code == 400
    assert env.config.events.identify_only_significant is True


def test_an_unknown_event_rule_is_named_with_what_is_editable(env):
    """Accepting it silently returned 200 for a rule that never moved."""
    r = env.client.post("/classes/events", json={"merge_gap_s": 30})
    assert r.status_code == 400
    assert "unknown event rule: merge_gap_s" in detail(r)
    assert "stitch_gap_s" in detail(r)


def test_a_refused_event_body_applies_none_of_its_valid_fields(env):
    before = env.config.events.min_detections
    r = env.client.post(
        "/classes/events", json={"min_detections": 9, "min_confidence": 7}
    )
    assert r.status_code == 400
    assert env.config.events.min_detections == before
    assert load_config(env.path).events.min_detections == before


# -- /classes/custom -------------------------------------------------------


def test_a_custom_class_threshold_is_on_the_similarity_scale(env):
    """A class saved at 8.5 can never match and never says why."""
    r = env.client.post(
        "/classes/custom", data={"name": "forklift", "threshold": "8.5"}
    )
    assert r.status_code == 400
    assert "cosine similarity in 0..1" in detail(r)
    with env.Session() as session:
        assert session.query(CustomClass).filter_by(name="forklift").count() == 0


# -- the box editor --------------------------------------------------------


def test_a_bbox_that_is_not_four_numbers_is_refused(env):
    """Clamping whatever arrived stored a two-element box, which the
    editor could not draw and no export could read."""
    r = env.client.post(
        "/api/items/1/annotations", json={"annotations": [{"bbox": [0.1, 0.2]}]}
    )
    assert r.status_code == 400
    assert "four numbers" in detail(r)


def test_a_bbox_of_text_is_refused(env):
    r = env.client.post(
        "/api/items/1/annotations",
        json={"annotations": [{"bbox": ["a", "b", "c", "d"]}]},
    )
    assert r.status_code == 400
    assert "annotations[0].bbox[0]" in detail(r)


def test_a_refused_save_deletes_nothing(env):
    """This endpoint replaces an item's boxes wholesale — it deletes
    every row it was not sent. A body it cannot read must therefore take
    no rows with it."""
    r = env.client.post(
        "/api/items/1/annotations",
        json={"annotations": [{"bbox": [0, 0, 1, 1]}, {"bbox": "everywhere"}]},
    )
    assert r.status_code == 400
    with env.Session() as session:
        row = session.get(Annotation, 1)
        assert row is not None
        assert json.loads(row.bbox) == [0.1, 0.1, 0.4, 0.4]


def test_a_valid_save_still_lands(env):
    r = env.client.post(
        "/api/items/1/annotations",
        json={
            "annotations": [
                {
                    "id": 1,
                    "bbox": [0.2, 0.2, 0.5, 0.5],
                    "class_name": "face",
                    "verified": True,
                    "rejected": False,
                }
            ]
        },
    )
    assert r.status_code == 200
    with env.Session() as session:
        assert json.loads(session.get(Annotation, 1).bbox) == [0.2, 0.2, 0.5, 0.5]


def test_a_tag_that_is_not_text_is_refused(env):
    r = env.client.post("/api/items/1/tags", json={"tags": [3]})
    assert r.status_code == 400
    assert "every entry of tags" in detail(r)


# -- /api/training/review --------------------------------------------------


def review(env, *decisions):
    return env.client.post("/api/training/review", json={"decisions": list(decisions)})


def test_decisions_must_be_a_list(env):
    r = env.client.post("/api/training/review", json={"decisions": "confirm"})
    assert r.status_code == 400
    assert "decisions must be a JSON array" in detail(r)


def test_a_decision_without_an_id_is_refused(env):
    r = review(env, {"action": "confirm", "name": "Ana"})
    assert r.status_code == 400
    assert "must name the annotation id" in detail(r)


def test_a_decision_id_that_is_not_an_id_is_refused(env):
    r = review(env, {"id": "abc", "action": "reject"})
    assert r.status_code == 400
    assert "decisions[0].id" in detail(r)


def test_an_unknown_action_is_named_rather_than_ignored(env):
    """It used to fall through every branch and report success with all
    counters at zero."""
    r = review(env, {"id": 1, "action": "approve"})
    assert r.status_code == 400
    assert "confirm" in detail(r) and "reject" in detail(r)
    with env.Session() as session:
        assert session.get(Annotation, 1).verified is False


def test_a_bad_decision_stops_the_whole_batch(env):
    """Read the batch, then apply it: the valid first decision must not
    land when the second cannot be read."""
    r = review(env, {"id": 1, "action": "reject"}, {"id": 2, "action": "flag"})
    assert r.status_code == 400
    with env.Session() as session:
        assert session.get(Annotation, 1).rejected is False


def test_a_decision_for_a_vanished_crop_is_reported_not_swallowed(env):
    """Not malformed — a crop deleted since the grid was rendered. The
    caller still has to be able to see it went nowhere."""
    r = review(env, {"id": 999, "action": "reject"})
    assert r.status_code == 200
    assert r.json()["missing"] == 1
    assert r.json()["rejected"] == 0


def test_a_valid_decision_still_applies(env):
    r = review(env, {"id": 1, "action": "reject"})
    assert r.status_code == 200
    assert r.json()["rejected"] == 1
    with env.Session() as session:
        assert session.get(Annotation, 1).rejected is True


# -- CLD-62: the store is held by another process --------------------------


@pytest.fixture
def locked(monkeypatch):
    """What a flock collision looks like from the web layer.

    `VectorStore` raises RuntimeError when the embedded database is
    already open elsewhere; `identity_ops.shared_store` is the one place
    that turns it into an answer.
    """

    def refuse(*args, **kwargs):
        raise RuntimeError(
            "Storage folder vectors is already accessed by another instance"
        )

    monkeypatch.setattr("siteloom.identity.get_shared_store", refuse)


def test_confirming_a_proposal_says_the_store_is_busy(env, locked):
    """Confirming enrolls, enrolment needs the vector store, and ingest
    holds it all night — the case an operator hits most, answered with a
    500 that named nothing."""
    r = review(env, {"id": 1, "action": "confirm", "name": "Ana"})
    assert r.status_code == 503
    assert "locked by another process" in detail(r)
    # The same guidance merge and split give: what to look at, and what
    # to do afterwards.
    assert "/jobs" in detail(r)
    assert "confirmation" in detail(r)


def test_a_busy_store_leaves_the_review_untouched(env, locked):
    """A label written without its vectors is a name the system cannot
    see — worse than a refusal, because it looks done."""
    review(env, {"id": 1, "action": "confirm", "name": "Ana"})
    with env.Session() as session:
        row = session.get(Annotation, 1)
        assert row.verified is False
        assert row.identity_id is None
        assert row.proposed_name == "Ana"  # unchanged, not re-stamped


def test_assigning_a_custom_class_says_the_store_is_busy(env, locked):
    """The other half of the same endpoint: a class example is a vector
    too, in the `class-examples` collection rather than a gallery. It
    reached the store by a different route and crashed the same way."""
    r = review(env, {"id": 2, "action": "classify", "custom_class": "delivery-van"})
    assert r.status_code == 503
    assert "locked by another process" in detail(r)
    with env.Session() as session:
        row = session.get(Annotation, 2)
        assert row.custom_class is None
        assert row.verified is False


def test_decisions_that_need_no_vectors_are_unaffected(env, locked):
    """Rejecting touches no gallery, so a held store must not block it."""
    r = review(env, {"id": 1, "action": "reject"})
    assert r.status_code == 200


def test_the_import_wizard_says_the_store_is_busy_too(env, locked):
    """Same shape, one screen over, and named by neither issue: the
    wizard's indexer needs the same store. It has to be resolved in the
    request for the refusal to reach anyone — resolved in the worker it
    was a 303 to a progress page for a job that had already died."""
    r = env.client.post(
        "/library/import/index", data={"source_id": 1}, follow_redirects=False
    )
    assert r.status_code == 503
    assert "locked by another process" in detail(r)


# -- /train/adopt ----------------------------------------------------------


def _run(env, **kwargs):
    fields = dict(
        kind="face-embed",
        started_at=TS,
        finished_at=TS,
        status="complete",
        sample_count=10,
        identity_count=3,
    )
    fields.update(kwargs)
    with env.Session() as session:
        run = TrainingRun(**fields)
        session.add(run)
        session.commit()
        return run.id


def _metrics(best_threshold=0.478):
    scored = {"valid": True, "auc": 0.9, "margin": 0.2, "accuracy": 0.9}
    return json.dumps(
        {
            "before": scored,
            "after": {**scored, "best_threshold": best_threshold},
            "improved": True,
        }
    )


def test_adopting_refuses_a_threshold_off_the_scale(env, tmp_path):
    """The tuned cutoff is written straight into site.yaml. A damaged
    metrics blob would otherwise install a threshold no face can clear —
    one the /classes page refuses to accept but would happily display."""
    artifact = tmp_path / "projection.npy"
    artifact.write_bytes(b"x")
    run_id = _run(env, artifact_path=str(artifact), metrics=_metrics(42))
    before = env.config.identity.identifiers["face"].threshold
    r = env.client.post(
        f"/train/adopt/{run_id}", data={"apply_threshold": "1"}, follow_redirects=False
    )
    assert r.status_code == 400
    assert "cosine similarity in 0..1" in detail(r)
    assert env.config.identity.identifiers["face"].threshold == before
    # ...and the projection is not adopted either: swapping the embedding
    # space while leaving the cutoff behind is the worst of both.
    assert env.config.identity.face_projection_path != str(artifact)


def test_adopting_refuses_an_apply_threshold_it_cannot_read(env, tmp_path):
    artifact = tmp_path / "projection.npy"
    artifact.write_bytes(b"x")
    run_id = _run(env, artifact_path=str(artifact), metrics=_metrics())
    r = env.client.post(f"/train/adopt/{run_id}", data={"apply_threshold": "yes"})
    assert r.status_code == 400
    assert "apply_threshold" in detail(r)


def test_an_unticked_checkbox_leaves_the_threshold_alone(env, tmp_path):
    """An unchecked checkbox is not submitted at all, so a default of "1"
    meant the box could be ticked but never unticked."""
    artifact = tmp_path / "projection.npy"
    artifact.write_bytes(b"x")
    run_id = _run(env, artifact_path=str(artifact), metrics=_metrics())
    before = env.config.identity.identifiers["face"].threshold
    r = env.client.post(f"/train/adopt/{run_id}", follow_redirects=False)
    assert r.status_code == 303
    assert env.config.identity.face_projection_path == str(artifact)
    assert env.config.identity.identifiers["face"].threshold == before
