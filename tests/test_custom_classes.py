"""Custom sub-class k-NN classification."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from siteloom.identity.classes import CustomClassifier
from siteloom.identity.vectors import VectorStore
from siteloom.store import CustomClass, get_session, init_db, make_engine

TS = datetime(2026, 8, 5, 12, 0)


def unit(v) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    return arr / np.linalg.norm(arr)


@pytest.fixture
def vectors(tmp_path):
    store = VectorStore(tmp_path / "vec")
    yield store
    store.close()


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/cc.db")
    init_db(engine)
    with get_session(engine)() as s:
        s.add(
            CustomClass(
                name="delivery-van", parent_class="car", threshold=0.8, created_at=TS
            )
        )
        s.add(
            CustomClass(
                name="my-truck", parent_class="truck", threshold=0.95, created_at=TS
            )
        )
        s.commit()
        yield s


def test_classify_matching_examples(vectors, session):
    clf = CustomClassifier(vectors, k=5)
    van = unit([1.0, 0.1, 0.0, 0.0])
    for _ in range(3):
        clf.add_example(van, "delivery-van")

    vote = clf.classify(session, van)
    assert vote is not None
    assert vote.name == "delivery-van"
    assert vote.score > 0.95
    assert vote.votes == 3


def test_below_threshold_returns_none(vectors, session):
    clf = CustomClassifier(vectors, k=5)
    clf.add_example(unit([1.0, 0.0, 0.0, 0.0]), "delivery-van")
    # Orthogonal query — similarity ~0, well under the 0.8 threshold.
    assert clf.classify(session, unit([0.0, 1.0, 0.0, 0.0])) is None


def test_per_class_thresholds_are_independent(vectors, session):
    """my-truck demands 0.95; a 0.9-similar crop must not match it."""
    clf = CustomClassifier(vectors, k=5)
    base = unit([1.0, 0.0, 0.0, 0.0])
    clf.add_example(base, "my-truck")
    near = unit([1.0, 0.45, 0.0, 0.0])  # cos ~0.912
    assert clf.classify(session, near) is None
    # The same crop clears delivery-van's looser 0.8 bar.
    clf.add_example(base, "delivery-van")
    vote = clf.classify(session, near)
    assert vote is not None and vote.name == "delivery-van"


def test_parent_class_scoping(vectors, session):
    clf = CustomClassifier(vectors, k=5)
    v = unit([1.0, 0.0, 0.0, 0.0])
    clf.add_example(v, "delivery-van")  # refines "car"
    assert clf.classify(session, v, parent_class="car") is not None
    # Asking within "truck" must not return a car sub-class.
    assert clf.classify(session, v, parent_class="truck") is None


def test_deleted_class_examples_ignored(vectors, session):
    clf = CustomClassifier(vectors, k=5)
    clf.add_example(unit([1.0, 0.0, 0.0, 0.0]), "removed-class")
    assert clf.classify(session, unit([1.0, 0.0, 0.0, 0.0])) is None


def test_majority_vote_wins_over_single_close_hit(vectors, session):
    clf = CustomClassifier(vectors, k=5)
    session.add(
        CustomClass(name="other-van", parent_class="car", threshold=0.8, created_at=TS)
    )
    session.commit()
    query = unit([1.0, 0.0, 0.0, 0.0])
    for _ in range(3):
        clf.add_example(unit([1.0, 0.12, 0.0, 0.0]), "delivery-van")
    clf.add_example(query, "other-van")  # one perfect but lone hit

    vote = clf.classify(session, query)
    assert vote.name == "delivery-van"
    assert vote.votes == 3
