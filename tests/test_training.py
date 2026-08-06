"""Training-data collection, splitting, evaluation, and export."""

from __future__ import annotations

import json
from datetime import datetime

import cv2
import numpy as np
import pytest

from siteloom.store import (
    Annotation,
    Identity,
    LibraryItem,
    LibrarySource,
    get_session,
    init_db,
    make_engine,
)
from siteloom.training.dataset import FaceSample, collect_face_samples, split_by_person
from siteloom.training.detector import export_yolo_dataset
from siteloom.training.face import evaluate_embeddings

TS = datetime(2026, 8, 5, 12, 0)


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/train.db")
    init_db(engine)
    with get_session(engine)() as s:
        s.add(LibrarySource(id=1, path=str(tmp_path), name="t", added_at=TS))
        s.commit()
        yield s


def add_item(session, tmp_path, name="a.jpg"):
    path = tmp_path / name
    cv2.imwrite(str(path), np.full((100, 100, 3), 128, dtype=np.uint8))
    item = LibraryItem(
        source_id=1, path=str(path), kind="image", mtime=TS, status="indexed"
    )
    session.add(item)
    session.flush()
    return item


def add_face(session, item, name, verified=True, rejected=False, identity_id=None):
    annotation = Annotation(
        item_id=item.id,
        bbox=json.dumps([0.2, 0.2, 0.6, 0.7]),
        class_name="face",
        proposed_name=name,
        identity_id=identity_id,
        verified=verified,
        rejected=rejected,
        source="import",
        created_at=TS,
    )
    session.add(annotation)
    session.flush()
    return annotation


def test_collect_only_verified(session, tmp_path):
    item = add_item(session, tmp_path)
    add_face(session, item, "Ann", verified=True)
    add_face(session, item, "Bob", verified=False)
    add_face(session, item, "Cara", verified=True, rejected=True)
    session.commit()

    samples = collect_face_samples(session)
    assert [s.person for s in samples] == ["Ann"]


def test_identity_label_overrides_proposed_name(session, tmp_path):
    identity = Identity(
        identifier_key="face",
        class_name="person",
        label="Renamed Person",
        first_seen=TS,
        last_seen=TS,
    )
    session.add(identity)
    session.flush()
    item = add_item(session, tmp_path)
    add_face(session, item, "Old Guess", identity_id=identity.id)
    session.commit()

    assert collect_face_samples(session)[0].person == "Renamed Person"


def test_min_per_person_filter(session, tmp_path):
    for i in range(4):
        item = add_item(session, tmp_path, f"ann{i}.jpg")
        add_face(session, item, "Ann")
    item = add_item(session, tmp_path, "bob.jpg")
    add_face(session, item, "Bob")
    session.commit()

    samples = collect_face_samples(session, min_per_person=3)
    assert {s.person for s in samples} == {"Ann"}


def make_samples(person: str, n: int, start: int = 0) -> list[FaceSample]:
    return [
        FaceSample(person=person, image_path="x", bbox=[0, 0, 1, 1], crop_path=None,
                   annotation_id=start + i)
        for i in range(n)
    ]


def test_split_keeps_every_person_in_train():
    samples = make_samples("Ann", 8) + make_samples("Bob", 4, 100)
    train, val = split_by_person(samples, val_fraction=0.25)
    assert {s.person for s in train} == {"Ann", "Bob"}
    # Ann: 25% of 8 = 2. Bob: 25% of 4 = 1, raised to the 2-sample floor.
    assert len(val) == 4


def test_split_guarantees_same_person_pairs_in_val():
    """Without at least two val samples for some person, validation has
    no same-person pairs and AUC is undefined — the bug this floor
    prevents."""
    from collections import Counter

    samples = make_samples("Ann", 6) + make_samples("Bob", 6, 100)
    _train, val = split_by_person(samples, val_fraction=0.1)
    assert max(Counter(s.person for s in val).values()) >= 2


def test_split_never_strands_a_single_sample_person():
    samples = make_samples("Ann", 1) + make_samples("Bob", 6, 50)
    train, val = split_by_person(samples, val_fraction=0.9)
    assert "Ann" in {s.person for s in train}
    assert all(s.person != "Ann" for s in val)


def test_evaluate_embeddings_separates_perfectly():
    """Two tight clusters -> AUC 1.0."""
    a = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (4, 1))
    b = np.tile(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), (4, 1))
    vectors = np.vstack([a, b])
    people = ["Ann"] * 4 + ["Bob"] * 4
    metrics = evaluate_embeddings(vectors, people, threshold=0.5)
    assert metrics.auc == 1.0
    assert metrics.accuracy == 1.0
    assert metrics.same_mean > metrics.diff_mean
    assert metrics.pairs == 28


def test_evaluate_embeddings_random_is_chance():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(20, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    people = ["Ann"] * 10 + ["Bob"] * 10
    metrics = evaluate_embeddings(vectors, people, threshold=0.5)
    assert 0.3 < metrics.auc < 0.7  # no real signal


def test_evaluate_single_class_is_degenerate():
    vectors = np.eye(3, dtype=np.float32)
    assert evaluate_embeddings(vectors, ["Ann"] * 3).auc == 0.0


def test_export_yolo_dataset(session, tmp_path):
    for i in range(5):
        item = add_item(session, tmp_path, f"p{i}.jpg")
        add_face(session, item, "Ann")
    session.commit()

    out = tmp_path / "dataset"
    result = export_yolo_dataset(session, out, val_fraction=0.2)
    assert result.boxes == 5
    assert result.train_images + result.val_images == 5
    assert (out / "dataset.yaml").exists()

    labels = sorted((out / "labels" / "train").glob("*.txt"))
    assert labels
    parts = labels[0].read_text().strip().split()
    assert parts[0] == "0"  # class index for "face"
    cx, cy, w, h = (float(v) for v in parts[1:])
    # bbox [0.2, 0.2, 0.6, 0.7] -> center (0.4, 0.45), size (0.4, 0.5)
    assert abs(cx - 0.4) < 1e-6 and abs(cy - 0.45) < 1e-6
    assert abs(w - 0.4) < 1e-6 and abs(h - 0.5) < 1e-6


def test_export_excludes_unverified(session, tmp_path):
    item = add_item(session, tmp_path)
    add_face(session, item, "Ann", verified=False)
    session.commit()
    result = export_yolo_dataset(session, tmp_path / "ds")
    assert result.boxes == 0
    assert "verify training data" in result.message
