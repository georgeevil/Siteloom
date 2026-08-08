"""Assigning a custom class from the training inspector (CLD-29).

The interesting assertions are all about what assignment must *not* do.
A custom class is k-NN over the shared appearance embedding, living in
its own `class-examples` collection; enrolment is a different operation
into an identity's gallery. Conflating them would put a face vector
somewhere it was never meant to go, and the two paths sit two lines apart
in the same endpoint.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import (
    Annotation,
    CustomClass,
    Identity,
    LibraryItem,
    LibrarySource,
    get_session,
    init_db,
    make_engine,
)
from siteloom.web.app import create_app

TS = datetime(2026, 8, 5, 12, 0)


@pytest.fixture
def client(tmp_path):
    config = SiteConfig(
        site_id="t",
        cameras=[CameraConfig(id="c", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/t.db", media_dir=str(tmp_path / "m")
        ),
    )
    config.identity.enabled = False
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        source = LibrarySource(name="s", path=str(tmp_path / "src"), added_at=TS)
        s.add(source)
        s.add(CustomClass(name="delivery-van", parent_class="car", created_at=TS))
        ident = Identity(
            identifier_key="face", class_name="person", label="Ana",
            first_seen=TS, last_seen=TS,
        )
        s.add(ident)
        s.flush()
        item = LibraryItem(
            source_id=source.id, path=str(tmp_path / "src" / "a.jpg"),
            kind="image", status="indexed", mtime=TS,
        )
        s.add(item)
        s.flush()
        # Two with a crop path on record, one without. The last is the
        # case that must label but not count.
        s.add(Annotation(
            item_id=item.id, bbox="[0,0,1,1]", class_name="car",
            confidence=0.9, source="auto", created_at=TS,
            crop_path=str(tmp_path / "m" / "crop1.jpg"),
        ))
        s.add(Annotation(
            item_id=item.id, bbox="[0,0,1,1]", class_name="face",
            confidence=0.9, source="auto", identity_id=ident.id,
            created_at=TS, crop_path=str(tmp_path / "m" / "crop2.jpg"),
        ))
        s.add(Annotation(
            item_id=item.id, bbox="[0,0,1,1]", class_name="car",
            confidence=0.9, source="auto", created_at=TS,
        ))
        s.commit()
    client = TestClient(create_app(config))
    client.sessionmaker = Session
    return client


def review(client, **decision):
    return client.post("/api/training/review", json={"decisions": [decision]})


def test_assigning_a_class_labels_and_verifies_the_crop(client):
    r = review(client, id=1, action="classify", custom_class="delivery-van")
    assert r.status_code == 200
    assert r.json()["classified"] == 1
    with client.sessionmaker() as s:
        a = s.get(Annotation, 1)
        assert a.custom_class == "delivery-van"
        assert a.verified is True
        assert a.rejected is False


def test_an_unknown_class_name_is_refused_not_created(client):
    """CustomClass rows carry the threshold and parent class that make a
    name mean anything. Typing a new one must not conjure those."""
    r = review(client, id=1, action="classify", custom_class="not-a-class")
    assert r.json()["classified"] == 0
    assert r.json()["skipped"] == 1
    with client.sessionmaker() as s:
        assert s.get(Annotation, 1).custom_class is None
        assert s.query(CustomClass).count() == 1


def test_a_blank_class_is_refused(client):
    assert review(client, id=1, action="classify", custom_class="")\
        .json()["classified"] == 0
    assert review(client, id=1, action="classify").json()["skipped"] == 1


def test_assignment_never_touches_the_identity(client):
    """A crop can be both "Ana" and "delivery-van". Assigning one must
    not restate — or clear — the other."""
    review(client, id=2, action="classify", custom_class="delivery-van")
    with client.sessionmaker() as s:
        a = s.get(Annotation, 2)
        assert a.custom_class == "delivery-van"
        assert a.identity_id == 1        # untouched
        assert a.enrolled is False       # not an enrolment


def test_a_crop_with_no_file_is_labelled_but_not_counted_as_an_example(client):
    """Annotation 3 has no crop_path. The label lands; the vote cannot.
    Reporting them as one number would claim an example that has no way
    to influence anything — and would disagree with what
    CustomClassifier.rebuild() recomputes, which skips exactly these."""
    body = review(client, id=3, action="classify",
                  custom_class="delivery-van").json()
    assert body["classified"] == 1
    assert body["examples"] == 0
    with client.sessionmaker() as s:
        assert s.get(Annotation, 3).custom_class == "delivery-van"
        assert s.query(CustomClass).one().example_count == 0


def test_example_count_is_derived_not_incremented(client):
    """Re-assigning an already-classified crop must not inflate the chip."""
    for _ in range(3):
        review(client, id=1, action="classify", custom_class="delivery-van")
    with client.sessionmaker() as s:
        assert s.query(CustomClass).one().example_count == 1


def test_rejecting_a_classified_crop_drops_it_from_the_count(client):
    review(client, id=1, action="classify", custom_class="delivery-van")
    review(client, id=2, action="classify", custom_class="delivery-van")
    with client.sessionmaker() as s:
        assert s.query(CustomClass).one().example_count == 2

    review(client, id=2, action="reject")
    # The count is recomputed on the next assignment touching the class.
    review(client, id=1, action="classify", custom_class="delivery-van")
    with client.sessionmaker() as s:
        assert s.query(CustomClass).one().example_count == 1


def test_other_decisions_still_work(client):
    """The new branch sits between confirm and reject in one endpoint."""
    assert review(client, id=1, action="reject").json()["rejected"] == 1
    with client.sessionmaker() as s:
        assert s.get(Annotation, 1).rejected is True
    assert review(client, id=1, action="unset").status_code == 200
    with client.sessionmaker() as s:
        assert s.get(Annotation, 1).rejected is False
