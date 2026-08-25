"""The replay lab: corpus building, embed-once caching, sandbox seeding,
variant sweeps and verdict scoring (siteloom/lab.py).

Stub embedders throughout — a solid-colour crop embeds as its colour's
one-hot unit vector, so same-colour crops score 1.0 and different
colours 0.0, and every resolver outcome is arranged by choosing colours.
The vector store is real embedded Qdrant in tmp_path; no model weights.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

from siteloom import lab
from siteloom.config import (
    CameraConfig,
    IdentifierConfig,
    IdentityConfig,
    SiteConfig,
    StorageConfig,
)
from siteloom.identity.vectors import VectorStore
from siteloom.store import (
    Camera,
    Detection,
    Event,
    EventIdentity,
    Identity,
    PlateRead,
    get_session,
    init_db,
    make_engine,
)

T0 = datetime(2026, 8, 23, 17, 0, 0)

#: Colour name -> BGR fill. Each embeds as its own one-hot vector.
PALETTE = {
    "ana": (255, 0, 0),
    "bob": (0, 255, 0),
    "carol": (0, 0, 255),
    "dcar": (255, 255, 0),
    "ecar": (0, 255, 255),
    "freya": (255, 0, 255),
    "gus": (128, 128, 128),
    "hana": (255, 255, 255),
}
_ORDER = list(PALETTE)


def _vector_for(colour: str) -> np.ndarray:
    vec = np.zeros(len(_ORDER), dtype=np.float32)
    vec[_ORDER.index(colour)] = 1.0
    return vec


class StubEmbedder:
    """Nearest palette colour -> that colour's one-hot vector."""

    dim = len(_ORDER)

    def __init__(self):
        self.calls = 0

    def _colour(self, bgr) -> str:
        mean = bgr.reshape(-1, 3).mean(axis=0)
        return min(
            PALETTE,
            key=lambda name: float(
                np.abs(mean - np.array(PALETTE[name], dtype=np.float32)).sum()
            ),
        )

    def embed(self, bgr):
        self.calls += 1
        return _vector_for(self._colour(bgr))


class StubFaceEmbedder(StubEmbedder):
    """A face-shaped stub: reports a fixed face score via embed_best."""

    def __init__(self, score: float = 0.4):
        super().__init__()
        self.score = score

    def embed_best(self, bgr):
        self.calls += 1
        return _vector_for(self._colour(bgr)), self.score


def _write_crops(tmp_path):
    crops = tmp_path / "crops"
    crops.mkdir()
    paths = {}
    for name, colour in PALETTE.items():
        square = np.full((64, 64, 3), colour, dtype=np.uint8)
        path = crops / f"{name}.jpg"
        cv2.imwrite(str(path), square)
        paths[name] = str(path)
    return paths


def _identity_cfg(**person_overrides) -> IdentityConfig:
    """Two identifiers, everything explicit so tests read as intent."""
    person = dict(
        algo="generic",
        applies_to=["person"],
        threshold=0.8,
        min_margin=0.0,
        min_sightings=2,
        immediate_quality=0.95,
        learn_min_quality=0.0,
        learn_max_per_event=3,
        mint_max_per_event=0,
        max_vectors_per_identity=20,
    )
    person.update(person_overrides)
    return IdentityConfig(
        identifiers={
            "person": IdentifierConfig(**person),
            "vehicle": IdentifierConfig(
                algo="generic",
                applies_to=["car"],
                threshold=0.8,
                min_sightings=1,
                mint_max_per_event=0,
            ),
        }
    )


@pytest.fixture
def env(tmp_path):
    """Two target events plus per-identity gallery events.

    Event `people` (person): frames ana, ana, bob, carol, carol, ana —
    plus one low-confidence and one tiny-bbox frame the identify gates
    refuse. Its live claims: Ana confirmed, Bob wrong. Event `car`
    (car): three dcar frames, an accepted plate on the middle one, and
    a live claim on Tacoma (which owns that plate but a *different*
    colour gallery, so only the plate can match it).
    """
    crops = _write_crops(tmp_path)
    config = SiteConfig(
        site_id="lab-test",
        cameras=[CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/lab.db",
            media_dir=str(tmp_path / "media"),
        ),
        identity=_identity_cfg(),
    )
    config.identity.vector_db_path = str(tmp_path / "live_vectors")
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)

    def add_event(session, cls, colours, start, *, confidences=None, bboxes=None):
        event = Event(
            camera_id="cam1",
            track_id=1,
            class_name=cls,
            first_seen=start,
            last_seen=start + timedelta(seconds=len(colours)),
            detection_count=len(colours),
            best_confidence=0.9,
        )
        session.add(event)
        session.flush()
        dets = []
        for i, colour in enumerate(colours):
            det = Detection(
                event_id=event.id,
                timestamp=start + timedelta(seconds=i),
                class_name=cls,
                confidence=(confidences or {}).get(i, 0.9),
                bbox=(bboxes or {}).get(i, "[0, 0, 200, 200]"),
                zones="[]",
                crop_path=crops[colour],
            )
            session.add(det)
            dets.append(det)
        session.flush()
        return event, dets

    with Session() as session:
        session.add(Camera(id="cam1", site_id="lab-test", name="Cam One"))

        def gallery_identity(key, cls, label, colour, start, *, plate=None):
            event, _ = add_event(session, cls, [colour, colour], start)
            identity = Identity(
                identifier_key=key,
                class_name=cls,
                label=label,
                plate=plate,
                first_seen=start,
                last_seen=start,
            )
            session.add(identity)
            session.flush()
            session.add(
                EventIdentity(
                    event_id=event.id,
                    identity_id=identity.id,
                    identifier_key=key,
                    similarity=0.9,
                    hit_count=2,
                )
            )
            return identity

        ana = gallery_identity(
            "person", "person", "Ana", "ana", T0 - timedelta(hours=1)
        )
        bob = gallery_identity(
            "person", "person", "Bob", "bob", T0 - timedelta(minutes=50)
        )
        tacoma = gallery_identity(
            "vehicle", "car", "Tacoma", "ecar", T0 - timedelta(minutes=40),
            plate="TEST123",
        )

        people, _ = add_event(
            session,
            "person",
            ["ana", "ana", "bob", "carol", "carol", "ana", "ana", "ana"],
            T0,
            confidences={2: 0.7, 6: 0.3},
            bboxes={7: "[0, 0, 20, 20]"},
        )
        for identity, verdict in ((ana, "confirmed"), (bob, "wrong")):
            session.add(
                EventIdentity(
                    event_id=people.id,
                    identity_id=identity.id,
                    identifier_key="person",
                    similarity=0.9,
                    hit_count=3,
                    verdict=verdict,
                    verdict_at=T0,
                )
            )

        car, car_dets = add_event(
            session, "car", ["dcar", "dcar", "dcar"], T0 + timedelta(minutes=5)
        )
        session.add(
            PlateRead(
                event_id=car.id,
                detection_id=car_dets[1].id,
                camera_id="cam1",
                class_name="car",
                identifier_key="vehicle",
                at=car_dets[1].timestamp,
                text="TEST123",
                accepted=True,
            )
        )
        session.add(
            EventIdentity(
                event_id=car.id,
                identity_id=tacoma.id,
                identifier_key="vehicle",
                similarity=1.0,
                hit_count=1,
                matched_by="plate",
            )
        )
        session.commit()
        ids = {
            "people": people.id,
            "car": car.id,
            "ana": ana.id,
            "bob": bob.id,
            "tacoma": tacoma.id,
        }

    return config, Session, ids, tmp_path


def _stub_factory(made: dict | None = None):
    def factory(algo: str):
        embedder = (
            StubFaceEmbedder() if algo == "face" else StubEmbedder()
        )
        if made is not None:
            made.setdefault(algo, []).append(embedder)
        return embedder

    return factory


def _prepared(env, *, seed_max=20):
    """Corpus + bank + plan + seeder, the way the CLI wires them."""
    config, Session, ids, tmp_path = env
    replayed = [ids["people"], ids["car"]]
    with Session() as session:
        corpus = lab.build_corpus(session, config, replayed)
        algo_for = lab.algo_map(config, corpus, [])
        plan = lab.seed_plan(
            session,
            list(algo_for),
            scope="live",
            max_vectors=seed_max,
            exclude_event_ids=tuple(replayed),
        )
        targets = lab.embedding_targets(corpus, plan, config, algo_for)
        bank, stats = lab.embed_corpus(
            targets, config, tmp_path / "cache", embedder_factory=_stub_factory()
        )

    def seeder(sandbox, store):
        return lab.seed_reembed(
            sandbox, store, plan, bank, algo_for, max_vectors=seed_max
        )

    return config, corpus, algo_for, plan, bank, stats, seeder


def _run(env, name="baseline", overrides=(), *, face_quality="detector", **kw):
    config, corpus, algo_for, plan, bank, stats, seeder = _prepared(env)
    cfg = (
        lab.apply_overrides(config.identity, list(overrides))
        if overrides
        else config.identity.model_copy(deep=True)
    )
    result = lab.run_variant(
        name, cfg, corpus, bank, seeder,
        config=config, algo_for=algo_for, face_quality=face_quality, **kw,
    )
    return lab.score_variant(result, corpus), result, corpus


# -- corpus -----------------------------------------------------------------


def test_corpus_orders_gates_plates_and_verdicts(env):
    config, Session, ids, _ = env
    with Session() as session:
        corpus = lab.build_corpus(session, config, [ids["people"], ids["car"]])

    stamps = [f.ts for f in corpus.frames]
    assert stamps == sorted(stamps)
    assert corpus.classes == ["person", "car"]

    people = [f for f in corpus.frames if f.event_id == ids["people"]]
    assert [f.gated for f in people] == [
        None, None, None, None, None, None, "confidence", "crop_px",
    ]
    plated = [f for f in corpus.frames if f.plates]
    assert len(plated) == 1
    assert plated[0].plates == {"vehicle": "TEST123"}

    info = corpus.events[ids["people"]]
    assert info.judged == {ids["ana"]: "confirmed", ids["bob"]: "wrong"}
    assert info.live_links == {ids["ana"], ids["bob"]}


def test_corpus_refuses_unknown_events(env):
    config, Session, ids, _ = env
    with Session() as session:
        with pytest.raises(lab.LabError, match="99999"):
            lab.build_corpus(session, config, [ids["people"], 99999])


# -- embeddings -------------------------------------------------------------


def test_embed_cache_makes_resweeps_free(env, tmp_path):
    config, Session, ids, root = env
    with Session() as session:
        corpus = lab.build_corpus(session, config, [ids["people"], ids["car"]])
        algo_for = lab.algo_map(config, corpus, [])
        plan = lab.seed_plan(session, list(algo_for))
    targets = lab.embedding_targets(corpus, plan, config, algo_for)

    made: dict = {}
    _, first = lab.embed_corpus(
        targets, config, root / "cache", embedder_factory=_stub_factory(made)
    )
    assert first["embedded"] == len({p for p, _ in targets})
    calls_first = sum(e.calls for lst in made.values() for e in lst)
    assert calls_first == first["embedded"]

    made_again: dict = {}
    bank, second = lab.embed_corpus(
        targets, config, root / "cache", embedder_factory=_stub_factory(made_again)
    )
    assert second["embedded"] == 0
    assert made_again == {}  # cache hit: no embedder was even built
    assert all(entry[0] is not None for entry in bank.values())


def test_missing_crop_files_are_counted_not_fatal(env):
    config, _, _, root = env
    targets = {(str(root / "gone.jpg"), "generic")}
    bank, stats = lab.embed_corpus(
        targets, config, root / "cache2", embedder_factory=_stub_factory()
    )
    assert stats["missing_files"] == 1
    assert bank == {}


# -- seeding ----------------------------------------------------------------


def test_seed_plan_honors_an_over_cap_gallery(env):
    config, Session, ids, _ = env
    with Session() as session:
        # Ana's gallery event has 2 crops; ask for more than the live
        # cap ever kept and the plan simply carries what exists, capped
        # by the request — the "more vectors" experiment needs no
        # special path.
        plan = lab.seed_plan(session, ["person", "vehicle"], max_vectors=25)
    by_label = {p.label: p for p in plan}
    assert set(by_label) == {"Ana", "Bob", "Tacoma"}
    assert 1 <= len(by_label["Ana"].crop_paths) <= 25
    assert by_label["Tacoma"].plate == "TEST123"


def test_seed_plan_keeps_a_plate_only_identity(env):
    config, Session, ids, _ = env
    with Session() as session:
        session.add(
            Identity(
                identifier_key="vehicle",
                class_name="car",
                label="Ghost",
                plate="GHOST99",
                first_seen=T0,
                last_seen=T0,
            )
        )
        session.commit()
        plan = lab.seed_plan(session, ["vehicle"])
    ghost = next(p for p in plan if p.label == "Ghost")
    assert ghost.crop_paths == []  # no crops, but the plate can still match


def test_copy_seed_refuses_a_held_store(env, tmp_path):
    config, Session, ids, _ = env
    held = VectorStore(config.identity.vector_db_path)
    try:
        engine = make_engine("sqlite://")
        init_db(engine)
        store = VectorStore(tmp_path / "sandbox_vectors")
        try:
            with get_session(engine)() as sandbox:
                with pytest.raises(lab.LabError, match="--seed reembed"):
                    lab.seed_copy(
                        config.identity.vector_db_path, sandbox, store, []
                    )
        finally:
            store.close()
    finally:
        held.close()


# -- variants ---------------------------------------------------------------


def test_baseline_replay_matches_the_operator_story(env):
    scored, result, corpus = _run(env)
    config, Session, ids, _ = env
    people = scored["events"][ids["people"]]
    car = scored["events"][ids["car"]]

    # Ana and Bob are re-found; carol's two sightings promote one mint.
    assert people["confirmed_links"] == [ids["ana"]]
    assert people["wrong_links"] == [ids["bob"]]
    assert people["mints"] == 1
    assert people["outcomes"].get("gated") == 2

    # The car matches Tacoma through its plate; the unmatched colour
    # mints once (vehicle min_sightings=1) and re-matches itself after.
    assert car["mints"] == 1
    assert ids["tacoma"] in car["linked_live"]
    plate_frames = [
        d for d in result.decisions
        if d.event_id == ids["car"] and d.gate == "plate"
    ]
    assert len(plate_frames) == 1


def test_variant_sweep_diverges_and_leaves_the_base_config_alone(env):
    """An unreachable threshold changes every number — and, notably,
    *reduces* mints: the pending pool clusters at the matching threshold
    itself, so nothing can ever accumulate the sightings to promote.
    The lab reproducing that coupling on four lines of assertions is
    exactly what it exists for."""
    config = env[0]
    before = config.identity.identifiers["person"].threshold
    base, _, _ = _run(env)
    strict, _, _ = _run(env, "strict", ["person.threshold=1.5"])
    assert config.identity.identifiers["person"].threshold == before
    assert strict["totals"]["mints"] < base["totals"]["mints"]
    assert strict["totals"]["confirmed_links"] == 0  # Ana is unreachable too
    verdict = lab.compare(base, strict)
    assert verdict["verdict"] == "MIXED"  # fewer wrong links, but Ana lost
    assert verdict["deltas"]["confirmed_links"] == (1, 0)


def test_decision_traces_attribute_the_mint_gate(env):
    base, result, _ = _run(env)
    config, Session, ids, _ = env
    gates = {
        d.gate for d in result.decisions
        if d.outcome == "minted" and d.event_id == ids["people"]
    }
    assert gates == {"promotion(2)"}  # carol minted from two pooled sightings
    vehicle_gates = {
        d.gate for d in result.decisions
        if d.outcome == "minted" and d.event_id == ids["car"]
    }
    assert vehicle_gates == {"unconditional"}  # min_sightings=1

    # Lowering immediate_quality below the frames' 0.9 confidence flips
    # carol's mint to the immediate path — and the trace says so.
    eager, result_eager, _ = _run(env, "eager", ["person.immediate_quality=0.5"])
    gates = {
        d.gate for d in result_eager.decisions
        if d.outcome == "minted" and d.event_id == ids["people"]
    }
    assert gates == {"immediate_quality"}


def test_mint_budget_shows_up_as_its_own_outcome(env):
    # Threshold 1.5: nothing visual ever matches, so every frame wants
    # to mint; a budget of 1 lets one through and refuses the rest,
    # visibly (not as generic "pending").
    scored, result, _ = _run(
        env, "budget", ["person.threshold=1.5", "person.mint_max_per_event=1",
                        "person.immediate_quality=0.5"],
    )
    config, Session, ids, _ = env
    people = scored["events"][ids["people"]]
    assert people["mints"] == 1
    assert people["outcomes"].get("mint-budget", 0) > 0


def test_face_quality_source_changes_the_mint_path(env):
    # A face-algo identifier whose stub reports a 0.4 face score: under
    # detector quality (0.9) an unmatched frame immediate-mints; under
    # yunet quality (0.4 < immediate_quality) it parks in the pool.
    config, Session, ids, tmp_path = env
    face_cfg = IdentityConfig(
        identifiers={
            "face": IdentifierConfig(
                algo="face",
                applies_to=["person"],
                threshold=0.8,
                min_sightings=2,
                immediate_quality=0.85,
                mint_max_per_event=0,
            )
        }
    )
    with Session() as session:
        corpus = lab.build_corpus(session, config, [ids["people"]])
    site = config.model_copy(deep=True)
    site.identity = face_cfg
    algo_for = {"face": "face"}
    targets = lab.embedding_targets(corpus, [], site, algo_for)
    bank, _ = lab.embed_corpus(
        targets, site, tmp_path / "cache-face", embedder_factory=_stub_factory()
    )

    def mint_gates(face_quality):
        result = lab.run_variant(
            "q", face_cfg, corpus, bank, None,
            config=site, algo_for=algo_for, face_quality=face_quality,
        )
        return {d.gate for d in result.decisions if d.outcome == "minted"}

    # Same crops, different quality source, different mint path: the
    # 0.9 person-box confidence sails over immediate_quality while the
    # 0.4 face score parks every frame until sightings corroborate.
    assert mint_gates("detector") == {"immediate_quality"}
    assert all(g.startswith("promotion(") for g in mint_gates("yunet"))


def test_replay_is_deterministic(env):
    first, _, _ = _run(env)
    second, _, _ = _run(env)
    assert first == second


def test_replay_never_touches_the_live_stores(env):
    config, Session, ids, _ = env
    from pathlib import Path

    with Session() as session:
        identities_before = session.query(Identity).count()
        claims_before = session.query(EventIdentity).count()
    _run(env, "strict", ["person.threshold=1.5"])
    with Session() as session:
        assert session.query(Identity).count() == identities_before
        assert session.query(EventIdentity).count() == claims_before
    # reembed seeding never even opens the live vector store.
    assert not Path(config.identity.vector_db_path).exists()


# -- overrides parser -------------------------------------------------------


def test_apply_overrides_parses_and_validates():
    cfg = _identity_cfg()
    out = lab.apply_overrides(
        cfg,
        [
            "person.threshold=0.85",
            "person.min_sightings=3",
            "person.score_aggregation=mean_top_k",
            "recency_window_s=60",
        ],
    )
    person = out.identifiers["person"]
    assert person.threshold == 0.85
    assert person.min_sightings == 3
    assert person.score_aggregation == "mean_top_k"
    assert out.recency_window_s == 60.0
    # The input was deep-copied, not edited.
    assert cfg.identifiers["person"].threshold == 0.8


@pytest.mark.parametrize(
    "spec, match",
    [
        ("person.threshold", "key=value"),
        ("ghost.threshold=1", "unknown identifier"),
        ("person.no_such_field=1", "unknown field"),
        ("person.algo=face", "cannot be swept"),
        ("no_such_top=1", "unknown identity field"),
        ("person.score_aggregation=median", "invalid config"),
    ],
)
def test_apply_overrides_refuses_bad_specs(spec, match):
    with pytest.raises(lab.LabError, match=match):
        lab.apply_overrides(_identity_cfg(), [spec])


# -- B7: the module reports the face pipeline's own quality -----------------


def test_identity_module_carries_embedder_quality(monkeypatch):
    from siteloom.dispatch.base import Job
    from siteloom.modules.identity import IdentityModule

    class Registryish:
        def __init__(self, cfg, device="mps"):
            self.cfg = cfg

        def identifiers_for(self, class_name):
            return [
                ("face", IdentifierConfig(algo="face", applies_to=["person"])),
                ("person", IdentifierConfig(algo="generic", applies_to=["person"])),
            ]

        def embedder_for(self, key):
            return StubFaceEmbedder(0.42) if key == "face" else StubEmbedder()

    monkeypatch.setattr(
        "siteloom.modules.identity.IdentifierRegistry", Registryish
    )
    module = IdentityModule(IdentityConfig())
    square = np.full((64, 64, 3), PALETTE["ana"], dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", square)
    assert ok
    result = module.process(
        Job(module="identity", payload={"crop_jpeg": buf.tobytes(),
                                        "class_name": "person"})
    )
    by_key = {e["identifier"]: e for e in result["embeddings"]}
    assert by_key["face"]["quality"] == 0.42
    assert by_key["person"]["quality"] is None


# -- review fixes: seed hygiene, fallback semantics, interrupt batches ------


def test_seed_copy_excludes_the_replayed_events_crops(env, tmp_path):
    """An exact copy must still not import vectors the live resolver
    learned from the incident under test — where provenance exists,
    those points are filtered like the reembed path filters crops."""
    config, Session, ids, root = env
    live_path = root / "live_copy_vectors"
    live = VectorStore(live_path)
    live.add(
        "vehicle", _vector_for("ecar"), ids["tacoma"], crop_path="/gallery/e.jpg"
    )
    live.add(
        "vehicle", _vector_for("dcar"), ids["tacoma"], crop_path="/incident/d.jpg"
    )
    live.close()

    seed = lab.SeedIdentity(
        live_id=ids["tacoma"], identifier_key="vehicle", class_name="car",
        label="Tacoma", plate="TEST123", plate_source=None,
        first_seen=T0, last_seen=T0,
    )
    engine = make_engine("sqlite://")
    init_db(engine)
    store = VectorStore(tmp_path / "copy_sandbox")
    try:
        with get_session(engine)() as sandbox:
            id_map = lab.seed_copy(
                str(live_path), sandbox, store, [seed],
                exclude_crop_paths={"/incident/d.jpg"},
            )
            (sandbox_id,) = id_map
            assert store.count_identity("vehicle", sandbox_id) == 1
            hit = store.best_match("vehicle", _vector_for("ecar"))
            assert hit is not None and hit.score > 0.99
    finally:
        store.close()


class FallbackFaceStub:
    """A face embedder that never finds a face but supports the
    enrolment tight-crop fallback path."""

    class _Rec:
        def feature(self, resized):
            mean = resized.reshape(-1, 3).mean(axis=0).astype(np.float32)
            return np.tile(mean, 3)[:8]

    _recognizer = _Rec()

    def embed_best(self, bgr):
        return None, None

    def embed(self, bgr):
        return None

    def _finish(self, feature):
        norm = np.linalg.norm(feature)
        return (feature / norm).astype(np.float32) if norm > 0 else None


def test_fallback_face_vectors_seed_but_never_replay(env, tmp_path):
    """Live matching produced no vector for a faceless crop and ran no
    resolve — so the enrolment-fallback vector may build a seed gallery
    (enrolment does exactly that live) but a corpus frame must replay
    as no-embedding, not as a decision live never made."""
    config, Session, ids, root = env
    face_cfg = IdentityConfig(
        identifiers={
            "face": IdentifierConfig(algo="face", applies_to=["person"])
        }
    )
    with Session() as session:
        corpus = lab.build_corpus(session, config, [ids["people"]])
    site = config.model_copy(deep=True)
    site.identity = face_cfg
    algo_for = {"face": "face"}
    targets = lab.embedding_targets(corpus, [], site, algo_for)
    bank, stats = lab.embed_corpus(
        targets, site, root / "cache-fallback",
        embedder_factory=lambda algo: FallbackFaceStub(),
    )
    assert all(entry.fallback and entry.vector is not None
               for entry in bank.values())
    assert stats["no_embedding"] == 0  # the vectors exist — for seeding

    result = lab.run_variant(
        "fb", face_cfg, corpus, bank, None, config=site, algo_for=algo_for
    )
    face_outcomes = {
        d.outcome for d in result.decisions if d.identifier == "face"
    }
    assert face_outcomes == {"no-embedding"}

    # The same bank still seeds a gallery, the way enrolment would.
    seed = lab.SeedIdentity(
        live_id=1, identifier_key="face", class_name="person",
        label="Ana", plate=None, plate_source=None,
        first_seen=T0, last_seen=T0,
        crop_paths=[corpus.frames[0].crop_path],
    )
    engine = make_engine("sqlite://")
    init_db(engine)
    store = VectorStore(tmp_path / "fb_sandbox")
    try:
        with get_session(engine)() as sandbox:
            id_map = lab.seed_reembed(sandbox, store, [seed], bank, algo_for)
            (sandbox_id,) = id_map
            assert store.count_identity("face", sandbox_id) == 1
    finally:
        store.close()


def test_embed_corpus_commits_batches_and_polls_check(env, monkeypatch):
    """The ProgressReporter contract: an interrupt mid-pass keeps what
    was computed, because the cache is saved before every poll."""
    config, Session, ids, root = env
    monkeypatch.setattr(lab, "_EMBED_BATCH", 2)
    crops = sorted(
        str(p) for p in (root / "crops").iterdir() if p.suffix == ".jpg"
    )
    targets = {(p, "generic") for p in crops[:5]}
    cache = root / "cache-interrupt"

    class Stop(RuntimeError):
        pass

    def check():
        raise Stop()

    with pytest.raises(Stop):
        lab.embed_corpus(
            targets, config, cache,
            embedder_factory=_stub_factory(), check=check,
        )
    saved = np.load(next(cache.glob("lab-emb-generic*.npz")), allow_pickle=True)
    assert len(saved["keys"]) >= 2  # the batch landed before the poll

    # A rerun picks the partial cache up instead of recomputing it.
    bank, stats = lab.embed_corpus(
        targets, config, cache, embedder_factory=_stub_factory()
    )
    assert stats["embedded"] == 5 - len(saved["keys"])
    assert len(bank) == 5


def test_a_camera_without_the_identity_module_is_gated(env):
    config, Session, ids, _ = env
    site = config.model_copy(deep=True)
    site.cameras[0].modules = ["detection"]
    with Session() as session:
        corpus = lab.build_corpus(session, site, [ids["people"]])
    assert {f.gated for f in corpus.frames} == {"no-identity-module"}
