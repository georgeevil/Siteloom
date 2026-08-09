"""Two-pass face-name assignment, the core of the Takeout importer.

A stub face embedder stands in for YuNet+SFace: it reports one face per
distinctly-coloured square in the image and embeds each as a fixed
vector, so the assignment logic is tested deterministically without
model downloads.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from siteloom.config import IdentityConfig, LibraryConfig, SiteConfig, StorageConfig
from siteloom.dispatch import LocalBackend
from siteloom.library import LibraryIndexer
from siteloom.library.takeout import TakeoutImporter
from siteloom.store import Annotation, get_session, init_db, make_engine

# Each "person" is a unique colour; the stub turns colour into identity.
PEOPLE = {
    "Ann": (255, 0, 0),
    "Bob": (0, 255, 0),
    "Cara": (0, 0, 255),
}
VECTORS = {
    "Ann": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "Bob": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "Cara": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
}


class StubFace:
    """Finds one 'face' per coloured square; embeds by colour."""

    dim = 4

    def detect(self, bgr):
        faces = []
        for name, colour in PEOPLE.items():
            # Tolerant match: the fixtures are JPEGs, so exact equality
            # would fail on compression artefacts.
            mask = (
                np.abs(bgr.astype(np.int16) - np.array(colour[::-1], dtype=np.int16))
                .max(axis=-1)
                < 40
            )
            if mask.sum() < 100:
                continue
            ys, xs = np.where(mask)
            x, y = float(xs.min()), float(ys.min())
            w = float(xs.max() - xs.min() + 1)
            h = float(ys.max() - ys.min() + 1)
            row = np.zeros(15, dtype=np.float32)
            row[:4] = [x, y, w, h]
            row[-1] = 0.99
            faces.append(row)
        return faces

    def embed_face(self, bgr, face):
        x, y, w, h = (int(v) for v in face[:4])
        patch = bgr[y : y + h, x : x + w]
        mean = patch.reshape(-1, 3).mean(axis=0)[::-1]  # BGR -> RGB
        best, best_dist = None, 1e9
        for name, colour in PEOPLE.items():
            dist = float(np.abs(mean - np.array(colour, dtype=np.float32)).sum())
            if dist < best_dist:
                best, best_dist = name, dist
        return VECTORS[best] if best_dist < 150 else None


def make_photo(path, names, size=240):
    """One coloured square per named person, side by side."""
    image = np.full((size, size, 3), 250, dtype=np.uint8)
    for i, name in enumerate(names):
        x = 10 + i * 70
        image[40:110, x : x + 50] = PEOPLE[name][::-1]
    cv2.imwrite(str(path), image)


def write_sidecar(path, title, people):
    path.write_text(
        json.dumps(
            {
                "title": title,
                "photoTakenTime": {"timestamp": "1700000000"},
                "people": [{"name": n} for n in people],
            }
        )
    )


@pytest.fixture
def takeout_dir(tmp_path):
    d = tmp_path / "Takeout" / "Google Photos" / "Album"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def importer(tmp_path, takeout_dir):
    config = SiteConfig(
        site_id="test",
        cameras=[],
        identity=IdentityConfig(enabled=False),
        library=LibraryConfig(),
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/tk.db", media_dir=str(tmp_path / "media")
        ),
    )
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    indexer = LibraryIndexer(config, Session, LocalBackend(), resolver=None)
    return TakeoutImporter(indexer, face_embedder=StubFace())


def test_unambiguous_assignment(importer, takeout_dir):
    """One face, one tag — certain, and auto-verified."""
    make_photo(takeout_dir / "solo.jpg", ["Ann"])
    write_sidecar(
        takeout_dir / "solo.jpg.supplemental-metadata.json", "solo.jpg", ["Ann"]
    )
    stats = importer.import_tree(takeout_dir)
    assert stats.unambiguous == 1

    with importer.Session() as session:
        rows = session.query(Annotation).all()
    assert len(rows) == 1
    assert rows[0].proposed_name == "Ann"
    assert rows[0].proposal_basis == "unambiguous"
    assert rows[0].verified is True
    assert rows[0].class_name == "face"
    # Auto-verified means nobody looked, and the row says so. `source`
    # cannot: it stays "import" after a human confirms (CLD-95).
    assert rows[0].source == "import"
    assert rows[0].verified_by == "import"
    assert rows[0].verified_at is not None


def test_auto_verify_can_be_disabled(tmp_path, takeout_dir, importer):
    importer.auto_verify_unambiguous = False
    make_photo(takeout_dir / "solo.jpg", ["Ann"])
    write_sidecar(
        takeout_dir / "solo.jpg.supplemental-metadata.json", "solo.jpg", ["Ann"]
    )
    importer.import_tree(takeout_dir)
    with importer.Session() as session:
        row = session.query(Annotation).one()
    assert row.verified is False
    # No sign-off, so no verifier — the pair never drifts apart.
    assert row.verified_by is None and row.verified_at is None


def test_group_photo_matched_against_gallery(importer, takeout_dir):
    """Pass 1 seeds Ann and Bob from solo shots; pass 2 uses that gallery
    to assign both faces in a group photo tagged with both names."""
    for name in ("Ann", "Bob"):
        make_photo(takeout_dir / f"{name}.jpg", [name])
        write_sidecar(
            takeout_dir / f"{name}.jpg.supplemental-metadata.json",
            f"{name}.jpg",
            [name],
        )
    make_photo(takeout_dir / "group.jpg", ["Ann", "Bob"])
    write_sidecar(
        takeout_dir / "group.jpg.supplemental-metadata.json",
        "group.jpg",
        ["Ann", "Bob"],
    )

    stats = importer.import_tree(takeout_dir)
    assert stats.unambiguous == 2
    assert stats.matched == 2

    with importer.Session() as session:
        group = [
            a
            for a in session.query(Annotation).all()
            if a.proposal_basis == "matched"
        ]
    assert {a.proposed_name for a in group} == {"Ann", "Bob"}
    # Gallery matches always need review — they are inferences.
    assert all(a.verified is False for a in group)


def test_each_name_used_once_in_a_photo(importer, takeout_dir):
    """Two faces, two names: greedy assignment must not give both faces
    the same name even if one matches better."""
    for name in ("Ann", "Bob"):
        make_photo(takeout_dir / f"{name}.jpg", [name])
        write_sidecar(
            takeout_dir / f"{name}.jpg.supplemental-metadata.json",
            f"{name}.jpg",
            [name],
        )
    make_photo(takeout_dir / "pair.jpg", ["Ann", "Bob"])
    write_sidecar(
        takeout_dir / "pair.jpg.supplemental-metadata.json", "pair.jpg", ["Ann", "Bob"]
    )
    importer.import_tree(takeout_dir)
    with importer.Session() as session:
        names = [
            a.proposed_name
            for a in session.query(Annotation).all()
            if a.proposal_basis == "matched"
        ]
    assert sorted(names) == ["Ann", "Bob"]


def test_single_face_single_remaining_name(importer, takeout_dir):
    """One face, one tag, but a multi-face photo elsewhere means it goes
    through pass 2 — still safely assignable with no gallery."""
    make_photo(takeout_dir / "x.jpg", ["Cara", "Ann"])
    write_sidecar(
        takeout_dir / "x.jpg.supplemental-metadata.json", "x.jpg", ["Cara", "Ann"]
    )
    stats = importer.import_tree(takeout_dir)
    # No gallery exists (no unambiguous photos), two faces two names:
    # nothing can be safely assigned.
    assert stats.unambiguous == 0
    assert stats.unresolved == 2


def test_untagged_faces_left_unassigned(importer, takeout_dir):
    make_photo(takeout_dir / "notag.jpg", ["Ann"])
    write_sidecar(
        takeout_dir / "notag.jpg.supplemental-metadata.json", "notag.jpg", []
    )
    stats = importer.import_tree(takeout_dir)
    # No person tags at all -> the item is not even a candidate.
    assert stats.unambiguous == 0
    with importer.Session() as session:
        assert session.query(Annotation).count() == 0


def test_edited_derivatives_skipped_by_default(importer, takeout_dir):
    """Google exports '-edited' copies without sidecars. They depict the
    same moment as the original, so importing both would seed the gallery
    with near-duplicates and inflate per-person coverage."""
    make_photo(takeout_dir / "solo.jpg", ["Ann"])
    write_sidecar(
        takeout_dir / "solo.jpg.supplemental-metadata.json", "solo.jpg", ["Ann"]
    )
    make_photo(takeout_dir / "solo-edited.jpg", ["Ann"])

    stats = importer.import_tree(takeout_dir)
    assert stats.skipped_derivative == 1
    assert stats.unambiguous == 1
    with importer.Session() as session:
        assert session.query(Annotation).count() == 1


def test_derivatives_can_be_included(importer, takeout_dir):
    make_photo(takeout_dir / "solo.jpg", ["Ann"])
    write_sidecar(
        takeout_dir / "solo.jpg.supplemental-metadata.json", "solo.jpg", ["Ann"]
    )
    make_photo(takeout_dir / "solo-edited.jpg", ["Ann"])

    stats = importer.import_tree(takeout_dir, include_derivatives=True)
    assert stats.skipped_derivative == 0


def test_import_is_idempotent(importer, takeout_dir):
    make_photo(takeout_dir / "solo.jpg", ["Ann"])
    write_sidecar(
        takeout_dir / "solo.jpg.supplemental-metadata.json", "solo.jpg", ["Ann"]
    )
    importer.import_tree(takeout_dir)
    importer.import_tree(takeout_dir)
    with importer.Session() as session:
        assert session.query(Annotation).count() == 1
