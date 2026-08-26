"""The Models screen: enrolment, fine-tune readiness, adoption (CLD-94).

The assertions worth reading are the ones about a score that does not
exist. `EvalMetrics.valid` is False when a split cannot produce both
same-person and different-person pairs, and the dataclass still carries
floats — a screen that prints them adopts a model on nothing. So the
tests here check the *absence* of numbers as carefully as their presence,
and that adoption is refused on anything unscored.

The second half is the enrolment backlog: a label without vectors is a
name the system cannot see, and until this screen there was no way to
find one outside a terminal.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from siteloom.config import (
    CameraConfig,
    IdentityConfig,
    SiteConfig,
    StorageConfig,
)
from siteloom.store import (
    Annotation,
    Identity,
    LibraryItem,
    LibrarySource,
    OperationRun,
    TrainingRun,
    User,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web import train_routes
from siteloom.web.app import create_app
from siteloom.web.auth import hash_password, required_role

TS = datetime(2026, 8, 9, 12, 0)


class StubEmbedder:
    """A face embedder that needs no ONNX weights.

    Embeds any crop as a unit vector chosen by its dominant channel, so
    two people's crops land in different directions and enrolment is
    observable without downloading a model.
    """

    dim = 4

    def embed(self, bgr):
        mean = bgr.reshape(-1, 3).mean(axis=0)
        vector = np.zeros(4, dtype=np.float32)
        vector[int(np.argmax(mean))] = 1.0
        return vector


@pytest.fixture(autouse=True)
def _idle_jobs():
    """No job claimed before or after a test — the state is module-wide."""
    train_routes._train_state.update({"thread": None, "kind": None})
    yield
    train_routes._train_state.update({"thread": None, "kind": None})


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "siteloom.identity.embedders.build_embedder",
        lambda algo, device="mps", projection_path=None: StubEmbedder(),
    )
    config = SiteConfig(
        site_id="t",
        site_name="T",
        cameras=[CameraConfig(id="c", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/w.db", media_dir=str(tmp_path / "m")
        ),
        identity=IdentityConfig(vector_db_path=str(tmp_path / "vectors")),
    )
    config.training.min_samples_per_person = 2
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as session:
        session.add(LibrarySource(id=1, path=str(tmp_path), name="a", added_at=TS))
        session.commit()
    return SimpleNamespace(
        client=TestClient(create_app(config), follow_redirects=False),
        Session=Session,
        config=config,
        tmp_path=tmp_path,
        counter=[0],
    )


def add_face(
    env,
    person: str | None,
    *,
    verified: bool = True,
    rejected: bool = False,
    enrolled: bool = False,
    crop: bool = True,
    colour=(255, 0, 0),
) -> int:
    """One verified face annotation with a real image and crop on disk."""
    env.counter[0] += 1
    n = env.counter[0]
    image_path = env.tmp_path / f"img{n}.jpg"
    cv2.imwrite(str(image_path), np.full((80, 80, 3), 60, dtype=np.uint8))
    crop_path = None
    if crop:
        crop_path = env.tmp_path / f"crop{n}.jpg"
        cv2.imwrite(str(crop_path), np.full((32, 32, 3), colour, dtype=np.uint8))
    with env.Session() as session:
        item = LibraryItem(
            source_id=1, path=str(image_path), kind="image", mtime=TS, status="indexed"
        )
        session.add(item)
        session.flush()
        annotation = Annotation(
            item_id=item.id,
            bbox=json.dumps([0.2, 0.2, 0.7, 0.8]),
            class_name="face",
            proposed_name=person,
            verified=verified,
            rejected=rejected,
            enrolled=enrolled,
            crop_path=str(crop_path) if crop_path else None,
            source="import",
            created_at=TS,
        )
        session.add(annotation)
        session.commit()
        return annotation.id


def add_run(env, **kw) -> int:
    with env.Session() as session:
        run = TrainingRun(
            kind=kw.pop("kind", "face-embed"),
            started_at=TS,
            finished_at=TS,
            status=kw.pop("status", "complete"),
            sample_count=kw.pop("sample_count", 12),
            identity_count=kw.pop("identity_count", 3),
            metrics=json.dumps(kw.pop("metrics", {})),
            artifact_path=kw.pop("artifact_path", None),
            notes=kw.pop("notes", ""),
        )
        session.add(run)
        session.commit()
        return run.id


def metrics(
    *,
    valid_before=True,
    valid_after=True,
    improved=True,
    auc_before=0.7123,
    auc_after=0.9456,
) -> dict:
    return {
        "improved": improved,
        "people": 3,
        "before": {"auc": auc_before, "margin": 0.2111, "valid": valid_before},
        "after": {
            "auc": auc_after,
            "margin": 0.4222,
            "accuracy": 0.8333,
            "threshold": 0.36,
            "best_threshold": 0.4777,
            "best_accuracy": 0.9111,
            "pairs": 66,
            "same_pairs": 12,
            "valid": valid_after,
        },
    }


def join_job(timeout: float = 30.0) -> None:
    thread = train_routes._train_state["thread"]
    if thread is not None:
        thread.join(timeout)
        assert not thread.is_alive(), "background job did not finish"


# -- an unscored evaluation is not a zero score ----------------------------


def test_an_invalid_evaluation_is_never_rendered_as_a_number(env):
    """The failure this screen exists to avoid: a split that could not be
    scored, shown as figures an operator can act on."""
    add_run(env, metrics=metrics(valid_after=False, improved=False))
    body = env.client.get("/train").text

    assert "Evaluation invalid" in body
    assert "same-person and different-person pairs" in body
    # Not one of the run's numbers reaches the page — including the AUC
    # that would otherwise read as a real, and rather good, result.
    for number in ("0.9456", "0.7123", "0.4222", "0.8333", "0.9111"):
        assert number not in body, f"rendered {number} from an unscored run"


def test_an_invalid_before_also_invalidates_the_comparison(env):
    """Comparing a real score against an undefined one is not a
    comparison; the pair is scored or it is nothing."""
    add_run(env, metrics=metrics(valid_before=False))
    body = env.client.get("/train").text
    assert "Evaluation invalid" in body
    assert "0.9456" not in body


def test_a_valid_evaluation_does_show_its_figures(env):
    """The counterpart: the refusal above must be about validity, not
    about the screen being unable to render numbers at all."""
    add_run(env, artifact_path=str(env.tmp_path / "p.npy"), metrics=metrics())
    body = env.client.get("/train").text
    assert "0.9456" in body and "0.7123" in body
    assert "Evaluation invalid" not in body


def test_run_row_carries_no_figures_when_the_evaluation_is_invalid(env):
    """Asserted on the payload too: a later template must not be able to
    reach a number the route decided was meaningless."""
    run_id = add_run(env, metrics=metrics(valid_after=False))
    with env.Session() as session:
        row = train_routes.run_row(session.get(TrainingRun, run_id), env.config)
    assert row["evaluation"]["valid"] is False
    numeric = [
        k
        for k, v in row["evaluation"].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    assert not numeric, f"numeric fields survived an invalid evaluation: {numeric}"


# -- adoption --------------------------------------------------------------


def test_adoption_is_refused_without_a_held_out_improvement(env, tmp_path):
    artifact = tmp_path / "projection.npy"
    artifact.write_bytes(b"x")
    run_id = add_run(
        env,
        status="no-improvement",
        artifact_path=str(artifact),
        metrics=metrics(improved=False),
    )
    r = env.client.post(f"/train/adopt/{run_id}")
    assert r.status_code == 400
    assert "no improvement" in r.text
    assert env.config.identity.face_projection_path != str(artifact)


def test_adoption_is_refused_when_the_evaluation_could_not_be_scored(env, tmp_path):
    """`improved` on an unscored run is not evidence — refused before the
    flag is even considered."""
    artifact = tmp_path / "projection.npy"
    artifact.write_bytes(b"x")
    run_id = add_run(
        env,
        artifact_path=str(artifact),
        metrics=metrics(valid_after=False, improved=True),
    )
    r = env.client.post(f"/train/adopt/{run_id}")
    assert r.status_code == 400
    assert "not a zero score" in r.text
    assert env.config.identity.face_projection_path != str(artifact)


def test_adoption_is_refused_when_the_projection_is_gone(env, tmp_path):
    run_id = add_run(
        env, artifact_path=str(tmp_path / "missing.npy"), metrics=metrics()
    )
    r = env.client.post(f"/train/adopt/{run_id}")
    assert r.status_code == 400
    assert "no longer on disk" in r.text


def test_adoption_switches_the_live_embedding_and_its_threshold(env, tmp_path):
    artifact = tmp_path / "projection.npy"
    artifact.write_bytes(b"x")
    run_id = add_run(env, artifact_path=str(artifact), metrics=metrics())
    r = env.client.post(f"/train/adopt/{run_id}", data={"apply_threshold": "1"})
    assert r.status_code == 303
    assert env.config.identity.face_projection_path == str(artifact)
    # A projection reshapes the similarity distribution, so the old cutoff
    # is stale the moment it is adopted.
    assert env.config.identity.identifiers["face"].threshold == 0.478
    assert "Adopted" in env.client.get(r.headers["location"]).text


def test_adoption_can_keep_the_existing_threshold(env, tmp_path):
    artifact = tmp_path / "projection.npy"
    artifact.write_bytes(b"x")
    before = env.config.identity.identifiers["face"].threshold
    run_id = add_run(env, artifact_path=str(artifact), metrics=metrics())
    env.client.post(f"/train/adopt/{run_id}", data={"apply_threshold": "0"})
    assert env.config.identity.identifiers["face"].threshold == before


def test_an_unadoptable_run_offers_no_adopt_button(env):
    add_run(env, metrics=metrics(valid_after=False))
    body = env.client.get("/train").text
    assert "/train/adopt/" not in body
    assert "Not adoptable" in body


def test_a_detector_run_is_never_offered_for_adoption(env):
    """Detector training improves detection; there is no embedding to
    adopt from it."""
    run_id = add_run(env, kind="face-detect", metrics={"mAP50": 0.71})
    with env.Session() as session:
        ok, why = train_routes.adoption_state(
            session.get(TrainingRun, run_id), env.config
        )
    assert ok is False and "face-embedding run" in why


def test_the_screen_says_the_detector_trains_detection_only(env):
    body = env.client.get("/train").text
    assert "detection only" in body
    assert "does not improve identification" in body


# -- enrolment backlog -----------------------------------------------------


def test_the_backlog_counts_only_what_a_sweep_would_enrol(env):
    add_face(env, "Ana")  # waiting
    add_face(env, "Ana")  # waiting
    add_face(env, "Bo")  # waiting
    add_face(env, "Cy", enrolled=True)  # done
    add_face(env, "Di", verified=False)  # not ground truth
    add_face(env, "Ed", rejected=True)  # a negative
    add_face(env, "Fi", crop=False)  # nothing to embed
    add_face(env, None)  # nothing to enrol under

    with env.Session() as session:
        backlog = train_routes.enrolment_backlog(session)
    assert backlog.pending == 3
    assert dict(backlog.people) == {"Ana": 2, "Bo": 1}
    assert backlog.enrolled == 1
    # Kept apart from the queue: no sweep turns these into vectors.
    assert (backlog.no_crop, backlog.no_name) == (1, 1)
    body = env.client.get("/train").text
    assert "Enrol 3 verified faces" in body


def test_the_backlog_follows_a_relabelled_identity(env):
    """The name a crop enrols under is the identity's label when it has
    one — the same resolution enroll.py performs."""
    annotation_id = add_face(env, "Old Guess")
    with env.Session() as session:
        identity = Identity(
            identifier_key="face",
            class_name="person",
            label="Renamed",
            first_seen=TS,
            last_seen=TS,
        )
        session.add(identity)
        session.flush()
        session.get(Annotation, annotation_id).identity_id = identity.id
        session.commit()
    with env.Session() as session:
        backlog = train_routes.enrolment_backlog(session)
    assert dict(backlog.people) == {"Renamed": 1}


def test_a_label_with_no_vectors_is_called_out(env):
    """The recognition-returns-nothing case: a confirmed person whose
    gallery is empty."""
    with env.Session() as session:
        session.add(
            Identity(
                identifier_key="face",
                class_name="person",
                label="Ghost",
                vector_count=0,
                first_seen=TS,
                last_seen=TS,
            )
        )
        session.add(
            Identity(
                identifier_key="face",
                class_name="person",
                label="Seen",
                vector_count=3,
                first_seen=TS,
                last_seen=TS,
            )
        )
        session.commit()
    with env.Session() as session:
        backlog = train_routes.enrolment_backlog(session)
    assert backlog.blind == ["Ghost"]
    body = env.client.get("/train").text
    assert "empty gallery" in body and "Ghost" in body


def test_enrolment_sweep_writes_vectors_and_shows_up_in_jobs(env):
    from siteloom.identity import get_shared_store

    add_face(env, "Ana", colour=(255, 0, 0))
    add_face(env, "Bo", colour=(0, 255, 0))
    r = env.client.post("/train/enroll")
    assert r.status_code == 303
    join_job()

    with env.Session() as session:
        rows = session.scalars(select(Annotation)).all()
        assert all(a.enrolled for a in rows)
        labels = {
            i.label: i.vector_count
            for i in session.scalars(select(Identity)).all()
        }
        assert labels == {"Ana": 1, "Bo": 1}
        # A long operation nobody can see is a long operation nobody can
        # trust: the sweep heartbeats an OperationRun like the CLI's.
        run = session.scalars(
            select(OperationRun).filter_by(kind="face-enroll")
        ).one()
        assert run.status == "complete"
        assert run.resume_command == "siteloom train enroll"

    store = get_shared_store(env.config.identity.vector_db_path)
    assert store.count_identity("face", 1) == 1
    # And the page now reports a clear backlog rather than the old count.
    assert "Nothing to enrol" in env.client.get("/train").text


# -- fine-tune readiness ---------------------------------------------------


def test_readiness_reports_the_floor_and_the_gap(env):
    for _ in range(3):
        add_face(env, "Ana")
    add_face(env, "Bo")  # one short of the 2-sample floor
    with env.Session() as session:
        readiness = train_routes.face_readiness(session, 2, 0.25)
    assert (readiness.samples, readiness.people) == (4, 2)
    assert readiness.ready_people == 1
    assert readiness.can_train is False
    assert "at least 2 people" in " ".join(readiness.blockers)
    assert "Bo (+1)" in " ".join(readiness.gap)


def test_readiness_refuses_to_promise_a_score_it_cannot_produce(env):
    """Three samples each splits one held-out sample per person: no
    same-person pairs, so the run would come back unscored."""
    for person in ("Ana", "Bo"):
        for _ in range(3):
            add_face(env, person)
    with env.Session() as session:
        readiness = train_routes.face_readiness(session, 2, 0.25)
    assert readiness.can_train is True
    assert readiness.can_evaluate is False
    assert "no same-person pairs" in readiness.evaluation_blocker
    body = env.client.get("/train").text
    assert "could not be scored" in body


def test_enough_samples_per_person_makes_the_split_scorable(env):
    for person in ("Ana", "Bo"):
        for _ in range(4):
            add_face(env, person)
    with env.Session() as session:
        readiness = train_routes.face_readiness(session, 2, 0.25)
    assert readiness.can_train and readiness.can_evaluate
    assert readiness.val_samples == 4
    assert "held-out split can be scored" in env.client.get("/train").text


def test_only_verified_annotations_count_as_training_data(env):
    for _ in range(4):
        add_face(env, "Ana")
    for _ in range(4):
        add_face(env, "Guessed", verified=False)
    with env.Session() as session:
        readiness = train_routes.face_readiness(session, 2, 0.25)
    assert readiness.samples == 4
    assert [p for p, _n, _r, _h in readiness.coverage] == ["Ana"]


def test_a_fine_tune_is_refused_when_the_data_cannot_support_one(env):
    add_face(env, "Ana")
    r = env.client.post("/train/face")
    assert r.status_code == 400
    assert "at least 2 people" in r.text
    with env.Session() as session:
        assert session.scalars(select(TrainingRun)).all() == []


def test_a_fine_tune_records_its_run_and_its_metrics(env, monkeypatch):
    from siteloom.training.face import EvalMetrics, TrainResult

    captured: dict = {}

    def fake_train(train, val, embedder, **kw):
        captured["train"] = len(train)
        captured["val"] = len(val)
        return TrainResult(
            projection_path=None,
            people=2,
            train_samples=len(train),
            val_samples=len(val),
            before=EvalMetrics(auc=0.6, valid=True),
            after=EvalMetrics(auc=0.5, valid=True),
            improved=False,
            message="no improvement on held-out data; base embeddings kept",
        )

    monkeypatch.setattr(
        "siteloom.training.face.train_face_projection", fake_train
    )
    for person in ("Ana", "Bo"):
        for _ in range(4):
            add_face(env, person)

    assert env.client.post("/train/face").status_code == 303
    join_job()

    assert (captured["train"], captured["val"]) == (4, 4)
    with env.Session() as session:
        run = session.scalars(select(TrainingRun)).one()
        assert run.status == "no-improvement"
        assert json.loads(run.metrics)["improved"] is False
        assert run.artifact_path is None
        operation = session.scalars(
            select(OperationRun).filter_by(kind="face-embed")
        ).one()
        assert operation.status == "complete"
    # A run that did not improve is reported, and not offered for adoption.
    body = env.client.get("/train").text
    assert "no-improvement" in body and "/train/adopt/" not in body


def test_a_heartbeat_keeps_a_silent_step_from_reading_as_dead():
    """`train_face_projection` reports nothing for minutes; a phase that
    goes cold for 120 s reports `is_stale`. The tick advances by zero —
    it claims the process is alive, never that work moved."""
    ticks = SimpleNamespace(n=0, moved=0)

    class FakeProgress:
        def advance(self, n=1, **counters):
            ticks.n += 1
            ticks.moved += n

    started = threading.Event()

    def slow():
        started.set()
        # Long enough to need several ticks at the test's interval.
        threading.Event().wait(0.25)
        return "done"

    result = train_routes.run_with_heartbeat(FakeProgress(), slow, interval=0.02)
    assert result == "done"
    assert started.is_set()
    assert ticks.n >= 2
    assert ticks.moved == 0


def test_an_exception_inside_the_step_reaches_the_caller():
    """Swallowed on the worker thread, the run would be recorded complete."""

    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        train_routes.run_with_heartbeat(
            SimpleNamespace(advance=lambda *a, **k: None), boom, interval=0.01
        )


# -- one at a time, and who may press the buttons --------------------------


def test_only_one_model_job_runs_at_a_time(env):
    """Both jobs write the one embedded vector store; a second concurrent
    run is not slower, it is wrong."""
    release = threading.Event()
    busy = threading.Thread(target=release.wait, daemon=True)
    busy.start()
    train_routes._train_state.update({"thread": busy, "kind": "enrolment"})
    try:
        for path in ("/train/enroll", "/train/face"):
            r = env.client.post(path)
            assert r.status_code == 409, path
            assert "already running" in r.text
    finally:
        release.set()
        busy.join(5)


def test_the_page_survives_an_empty_install(env):
    body = env.client.get("/train").text
    assert "No training runs recorded yet" in body
    assert "base SFace" in body


def test_models_shares_the_training_sidebar_entry(env):
    """One sidebar entry, two screens (the CLD-94 split lives in the
    routes, not in a second nav row). The entry stays lit on both pages,
    and each page carries the in-page tab to the other."""
    body = env.client.get("/train").text
    assert '<span class="sl-label">Training</span>' in body
    # No separate Models entry any more.
    assert '<span class="sl-label">Models</span>' not in body
    # The Training entry is the active one while on /train.
    assert 'class="sl-nav-item active"' in body
    # Cross-links both ways.
    assert 'href="/training"' in body
    assert 'href="/train"' in env.client.get("/training").text


# -- roles -----------------------------------------------------------------


def test_model_changes_are_admin_but_the_readout_is_not():
    for path in ("/train/enroll", "/train/face", "/train/adopt/1"):
        assert required_role("POST", path) == "admin"
    assert required_role("GET", "/train") == "view"
    # Labelling is judgement, not reconfiguration — the trailing slash in
    # the prefix is what keeps /training on `edit`.
    assert required_role("POST", "/training") == "edit"
    assert required_role("POST", "/api/training/review") == "edit"


def add_user(env, username, role):
    with env.Session() as session:
        session.add(
            User(
                username=username,
                password_hash=hash_password("hunter2!"),
                role=role,
                created_at=TS,
            )
        )
        session.commit()


def login(env, username):
    r = env.client.post(
        "/login", data={"username": username, "password": "hunter2!", "next": "/train"}
    )
    assert r.status_code in (200, 303)


def test_an_editor_may_look_but_not_retrain(env):
    add_face(env, "Ana")
    add_user(env, "eddy", "edit")
    login(env, "eddy")
    assert env.client.get("/train").status_code == 200
    assert env.client.post("/train/enroll").status_code == 403
    assert env.client.post("/train/face").status_code == 403
    assert env.client.post("/train/adopt/1").status_code == 403
    assert train_routes._train_state["thread"] is None


def test_a_viewer_can_read_the_screen(env):
    add_user(env, "vera", "view")
    login(env, "vera")
    assert env.client.get("/train").status_code == 200
