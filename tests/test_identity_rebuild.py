"""The embedding-space stamp and the vector rebuild (CLD-106).

A real embedded qdrant in a temp dir, a fake 8-dimensional embedder —
no model weights. The properties under test are the ticket's own rules:
labels survive, the honest unrecoverable count, the commit point that
leaves collections empty rather than mixed-space, and resumability.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pytest

from siteloom.config import SiteConfig, StorageConfig
from siteloom.identity.rebuild import plan_rebuild, run_rebuild
from siteloom.identity.space import (
    STAMP_FILE,
    compute_stamp,
    read_stamp,
    stamp_diff,
    write_stamp,
)
from siteloom.identity.vectors import VectorStore
from siteloom.store import Identity, get_session, init_db, make_engine

TS = __import__("datetime").datetime(2026, 8, 20, 12, 0, 0)


class FakeEmbedder:
    """Deterministic 8-d embedding from pixel content — same crop, same
    vector, which is all a space needs."""

    dim = 8

    def embed(self, bgr):
        seed = float(bgr.mean())
        vec = np.array([seed + i for i in range(8)], dtype=np.float32)
        return vec / np.linalg.norm(vec)


class Progress:
    """The reporter surface run_rebuild uses, with an optional trip
    wire — the InterruptingReporter idea, sized to this module."""

    def __init__(self, interrupt_phase=None, interrupt_after=None):
        self.interrupt_phase = interrupt_phase
        self.interrupt_after = interrupt_after
        self.advances = 0
        self.phases: list[str] = []

    @contextmanager
    def phase(self, name, total=0):
        self.phases.append(name)
        if self.interrupt_phase and name.startswith(self.interrupt_phase):
            raise KeyboardInterrupt(name)
        yield

    def advance(self, n=1, **counters):
        self.advances += 1

    def check_interrupt(self):
        if (
            self.interrupt_after is not None
            and self.advances >= self.interrupt_after
        ):
            raise KeyboardInterrupt("tripped")


def crop_file(tmp_path, name, shade):
    path = tmp_path / "media" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((40, 40, 3), shade, dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return str(path)


@pytest.fixture
def env(tmp_path):
    config = SiteConfig(
        site_id="t", cameras=[],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/t.db",
            media_dir=str(tmp_path / "media"),
        ),
    )
    config.identity.vector_db_path = str(tmp_path / "vectors")
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    vectors = VectorStore(config.identity.vector_db_path)

    with Session() as s:
        alice = Identity(identifier_key="face", class_name="person",
                         label="Alice", first_seen=TS, last_seen=TS)
        car = Identity(identifier_key="vehicle", class_name="car",
                       first_seen=TS, last_seen=TS)
        s.add_all([alice, car])
        s.flush()
        crops = {
            "alice1": crop_file(tmp_path, "a1.jpg", 40),
            "alice2": crop_file(tmp_path, "a2.jpg", 90),
            "car1": crop_file(tmp_path, "c1.jpg", 160),
        }
        # The OLD space: 4-dimensional vectors, some with provenance.
        old = np.ones(4, dtype=np.float32)
        vectors.add("face", old, alice.id, crop_path=crops["alice1"])
        vectors.add("face", old, alice.id, crop_path=crops["alice2"])
        vectors.add("vehicle", old, car.id, crop_path=crops["car1"])
        vectors.add("vehicle", old, car.id)  # no crop: unrecoverable
        vectors.add_labeled("face-pending", old, {"ts": 1.0})
        alice.vector_count, car.vector_count = 2, 2
        s.commit()
        ids = (alice.id, car.id)

    yield {
        "config": config, "Session": Session, "vectors": vectors,
        "ids": ids, "tmp": tmp_path,
    }
    vectors.close()


def rebuild(env, progress=None, resume=False):
    return run_rebuild(
        env["Session"], env["vectors"], env["config"],
        progress=progress or Progress(),
        embedder_for=lambda key: FakeEmbedder(),
        resume=resume,
    )


def test_plan_counts_the_recoverable_and_the_lost(env):
    with env["Session"]() as s:
        plan = plan_rebuild(s, env["vectors"], env["config"])
    assert plan.identities == 2
    assert plan.recoverable_points == 3
    assert plan.unrecoverable_points == 1  # the crop-less vehicle vector


def test_rebuild_keeps_labels_and_re_embeds_in_the_new_space(env):
    report = rebuild(env)
    assert report.vectors_written == 3
    assert report.unrecoverable_points == 1
    with env["Session"]() as s:
        alice = s.get(Identity, env["ids"][0])
        assert alice.label == "Alice"              # labels survive
        assert alice.vector_count == 2
        car = s.get(Identity, env["ids"][1])
        assert car.vector_count == 1               # the lost one is gone
    # The store agrees, in the new dimension.
    assert env["vectors"].count_identity("face", env["ids"][0]) == 2
    # Old-space pending evidence is dropped, not carried over.
    assert "face-pending" not in env["vectors"].collection_names()
    # And the stamp records the space that was just built.
    assert read_stamp(env["config"].identity.vector_db_path) is not None


def test_an_interrupt_after_the_commit_point_leaves_empty_not_mixed(env):
    """The ticket's rule: degraded-and-honest. Once collections drop,
    an interruption may leave them empty or partially refilled in the
    NEW space — never holding two spaces at once."""
    with pytest.raises(KeyboardInterrupt):
        rebuild(env, Progress(interrupt_phase="Re-embedding"))
    for name in ("face", "vehicle"):
        assert env["vectors"].count_identity(name, env["ids"][0]) == 0
        assert env["vectors"].count_identity(name, env["ids"][1]) == 0
    with env["Session"]() as s:
        assert s.get(Identity, env["ids"][0]).vector_count == 0  # unenrolled
    assert read_stamp(env["config"].identity.vector_db_path) is not None


def test_an_interrupted_re_embed_resumes_instead_of_restarting(env):
    with pytest.raises(KeyboardInterrupt):
        rebuild(env, Progress(interrupt_after=1))
    report = rebuild(env, resume=True)
    assert report.resumed_past == 1                # the done-log held
    with env["Session"]() as s:
        assert s.get(Identity, env["ids"][0]).vector_count == 2
        assert s.get(Identity, env["ids"][1]).vector_count == 1


def test_the_stamp_lives_with_the_vectors_and_qdrant_ignores_it(env):
    """The stamp file sits inside the embedded store's directory — wiped
    with the vectors by reset, carried with a copied store — and the
    store must keep working around it."""
    write_stamp(env["config"].identity.vector_db_path,
                compute_stamp(env["config"]))
    assert (Path(env["config"].identity.vector_db_path) / STAMP_FILE).is_file()
    probe = np.ones(4, dtype=np.float32)
    env["vectors"].add("face", probe, env["ids"][0])  # store still writable
    assert env["vectors"].count_identity("face", env["ids"][0]) == 3


def test_stamp_diff_names_what_moved(env):
    recorded = compute_stamp(env["config"])
    env["config"].detection.crop_margin = 0.3
    drifted = stamp_diff(recorded, compute_stamp(env["config"]))
    assert any("crop_margin" in line for line in drifted)
    assert stamp_diff(recorded, recorded) == []


def test_health_check_reads_the_three_states(env):
    from siteloom.health import Report, check_embedding_space

    config = env["config"]
    report = Report()
    check_embedding_space(report, config)
    assert report.warned and "unknown" in report.warned[0].detail  # unstamped

    write_stamp(config.identity.vector_db_path, compute_stamp(config))
    report = Report()
    check_embedding_space(report, config)
    assert not report.warned                                        # in sync

    config.detection.crop_margin = 0.5
    report = Report()
    check_embedding_space(report, config)
    assert report.warned and "crop_margin" in report.warned[0].detail
