"""Replay lab: re-resolve recorded events under different identity settings.

The question this module answers is the one every gate-tuning argument
ends at: *what would this event have done under those settings?* Live
ingest can only answer it with a config edit, a restart and another week
of traffic; the lab answers it offline, from what ingest already stored.

The shape is **embed once, re-resolve many**. Stored detection crops are
the same images live matching embedded ("one crop, two jobs"), so
re-embedding them lands in the same vector space; embeddings are cached
on disk, and after the first pass a whole sweep of config variants runs
in seconds, with no model in the loop. Each variant gets a fresh sandbox
— a temp-dir Qdrant store and an in-memory database — seeded with the
live identities' galleries, and the corpus is driven through the real
`IdentityResolver`, the same call shape as `ingest._identify`, thresholds
resolved by the same `threshold_for`. The lab reuses the live pipeline's
components and never forks them (PRD §6.6); what it must never do is
write to the live stores — the only live-DB write its CLI makes is the
`OperationRun` heartbeat every job writes.

Scoring leans on ground truth the operator already produced: an event's
`EventIdentity.verdict` rows say which live identities were confirmed or
repudiated on it, so a variant's links map back (through the seed) to
"kept the confirmed link" / "repeated the wrong one" / "minted fresh".

Sandbox fidelity limits, stated rather than hidden: `--seed reembed`
approximates which crops became gallery vectors (the live gates chose a
subset, and pre-CLD-84 vectors have no crop provenance); `--seed copy`
is exact but needs the live vector store unheld. The replay runs one
resolver in global timestamp order, which matches live behaviour for
single-camera incidents and approximates it across cameras.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from siteloom.config import CameraConfig, IdentityConfig, SiteConfig
from siteloom.identity.registry import IdentifierRegistry
from siteloom.identity.resolver import IdentityResolver
from siteloom.identity.vectors import VectorStore
from siteloom.store import get_session, init_db, make_engine
from siteloom.store.models import Detection, Event, EventIdentity, Identity, PlateRead


class LabError(RuntimeError):
    """A replay cannot proceed and the message says what to do about it."""


# -- corpus -----------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    event_id: int
    detection_id: int
    ts: datetime
    camera_id: str
    class_name: str
    crop_path: str | None
    confidence: float
    plates: dict[str, str]  # identifier key -> accepted plate text
    # None, or the identify-stage gate live ingest applied to this frame
    # ("insignificant" | "confidence" | "crop_px" | "missing-crop").
    # Gated frames stay in the corpus, marked — a variant may choose to
    # include them, and silently dropping rows would misreport coverage.
    gated: str | None


@dataclass
class EventInfo:
    camera_id: str
    class_name: str
    first_seen: datetime
    last_seen: datetime
    frames: int
    # Live identity id -> verdict, for every claim carrying one (active
    # or unlinked — an unlinked wrong claim is the strongest ground
    # truth there is).
    judged: dict[int, str]
    # Live identity ids with an active claim today, verdict or not.
    live_links: set[int]


@dataclass
class Corpus:
    frames: list[Frame]  # global timestamp order — live arrival order
    events: dict[int, EventInfo]
    classes: list[str]  # distinct detection classes, first-seen order


def build_corpus(
    session: Session, config: SiteConfig, event_ids: list[int]
) -> Corpus:
    """Everything the identify stage saw for these events, from rows.

    The corpus replays the *identify* stage: detection and tracking are
    upstream and already happened, and their output — the crop, its
    confidence and bbox — is on the `Detection` row. The identify gates
    are re-derived per camera from the same rules `ingest._identify`
    reads, so a frame the live pass never resolved is marked, not lost
    — with one stated approximation: significance is judged from the
    final `Event.significant` column, while live it grows per frame
    (`_update_significance`), so the first `min_detections - 1` frames
    of an eventually-significant event replay ungated where live gated
    them, and an `events retag` can move the column after the fact.
    """
    events = {
        e.id: e
        for e in session.scalars(select(Event).where(Event.id.in_(event_ids)))
    }
    missing = sorted(set(event_ids) - set(events))
    if missing:
        raise LabError(f"no such event(s): {', '.join(map(str, missing))}")

    cams = {c.id: c for c in config.cameras}
    rules_of = {
        eid: (
            config.events.for_camera(cams[e.camera_id])
            if e.camera_id in cams
            else config.events
        )
        for eid, e in events.items()
    }

    plates: dict[int, dict[str, str]] = defaultdict(dict)
    for read in session.scalars(
        select(PlateRead).where(
            PlateRead.event_id.in_(event_ids),
            PlateRead.accepted.is_(True),
            PlateRead.detection_id.is_not(None),
        )
    ):
        if read.text:
            plates[read.detection_id][read.identifier_key or "vehicle"] = read.text

    frames: list[Frame] = []
    for det in session.scalars(
        select(Detection)
        .where(Detection.event_id.in_(event_ids))
        .order_by(Detection.timestamp, Detection.id)
    ):
        event = events[det.event_id]
        rules = rules_of[det.event_id]
        cam = cams.get(event.camera_id)
        x1, y1, x2, y2 = json.loads(det.bbox)
        gated = None
        if cam is not None and "identity" not in cam.modules:
            gated = "no-identity-module"  # per-camera selection (NFR3)
        elif rules.identify_only_significant and not event.significant:
            gated = "insignificant"
        elif det.confidence < rules.identify_min_confidence:
            gated = "confidence"
        elif min(x2 - x1, y2 - y1) < rules.identify_min_crop_px:
            gated = "crop_px"
        elif not det.crop_path:
            gated = "missing-crop"
        frames.append(
            Frame(
                event_id=det.event_id,
                detection_id=det.id,
                ts=det.timestamp,
                camera_id=event.camera_id,
                class_name=det.class_name,
                crop_path=det.crop_path,
                confidence=float(det.confidence),
                plates=dict(plates.get(det.id, {})),
                gated=gated,
            )
        )
    frames.sort(key=lambda f: (f.ts, f.detection_id))

    judged: dict[int, dict[int, str]] = defaultdict(dict)
    live_links: dict[int, set[int]] = defaultdict(set)
    for claim in session.scalars(
        select(EventIdentity).where(EventIdentity.event_id.in_(event_ids))
    ):
        if claim.identity_id is None:
            continue  # a miss record, not a link
        if claim.verdict:
            judged[claim.event_id][claim.identity_id] = claim.verdict
        if claim.unlinked_at is None:
            live_links[claim.event_id].add(claim.identity_id)

    infos = {
        eid: EventInfo(
            camera_id=e.camera_id,
            class_name=e.class_name,
            first_seen=e.first_seen,
            last_seen=e.last_seen,
            frames=sum(1 for f in frames if f.event_id == eid),
            judged=dict(judged.get(eid, {})),
            live_links=set(live_links.get(eid, set())),
        )
        for eid, e in events.items()
    }
    classes = list(dict.fromkeys(f.class_name for f in frames))
    return Corpus(frames=frames, events=infos, classes=classes)


# -- embeddings, once -------------------------------------------------------


class Embedded(NamedTuple):
    """One crop's embedding under one algo.

    A None vector is a real measurement (no face in the crop) and is
    cached so a re-sweep does not re-run the detector. `fallback` marks
    a vector produced by the tight-crop enrolment fallback — usable for
    *seeding* (enrolment uses it live) but never for a corpus frame:
    live matching produced no vector for that frame and ran no resolve,
    and a replay must not invent decisions live never made.
    """

    vector: np.ndarray | None
    quality: float | None
    fallback: bool = False


#: (crop_path, algo) -> Embedded
EmbeddingBank = dict[tuple[str, str], Embedded]


def algo_map(config: SiteConfig, corpus: Corpus, keys: list[str]) -> dict[str, str]:
    """identifier key -> embedder algo, from the *base* config's registry.

    Variants sweep thresholds and gates, never algorithms: the embedding
    bank is keyed by algo, and a variant that changed an identifier's
    algo would silently read vectors from the wrong space. `apply_overrides`
    refuses `algo` for the same reason.
    """
    registry = IdentifierRegistry(config.identity)
    out: dict[str, str] = {}
    for cls in corpus.classes:
        for key, ident in registry.identifiers_for(cls):
            out[key] = ident.algo
    for key in keys:
        if key not in out:
            ident = config.identity.identifiers.get(key)
            out[key] = ident.algo if ident else "generic"
    return out


def embedding_targets(
    corpus: Corpus,
    plan: list["SeedIdentity"],
    config: SiteConfig,
    algo_for: dict[str, str],
) -> set[tuple[str, str]]:
    """Every (crop_path, algo) pair a replay will look up.

    Frames need one embedding per algo their class's identifiers use;
    seed identities need one per their identifier's algo. Computed up
    front so the embed pass is one bounded, progress-reported job.
    """
    targets: set[tuple[str, str]] = set()
    registry = IdentifierRegistry(config.identity)
    for frame in corpus.frames:
        if not frame.crop_path:
            continue
        for _key, ident in registry.identifiers_for(frame.class_name):
            targets.add((frame.crop_path, ident.algo))
    for seed in plan:
        algo = algo_for.get(seed.identifier_key, "generic")
        for crop in seed.crop_paths:
            targets.add((crop, algo))
    return targets


def _projection_tag(config: SiteConfig) -> str:
    """Cache-busting tag for the face projection: retraining it moves the
    whole face vector space, and stale cached vectors would silently
    score against fresh ones."""
    path = Path(config.identity.face_projection_path or "")
    if not path.is_file():
        return "raw"
    return hashlib.sha1(path.read_bytes()).hexdigest()[:8]


def _default_embedder_factory(config: SiteConfig):
    from siteloom.identity.embedders import build_embedder

    def factory(algo: str):
        return build_embedder(
            algo,
            device=config.detection.device,
            projection_path=config.identity.face_projection_path or None,
        )

    return factory


def _embed_one(embedder, image) -> Embedded:
    """One crop through one embedder, with its quality where it has one.

    Face embedders report the detector's score for the face they chose
    (`embed_best`); a crop with no detectable face falls back to the
    tight-crop embed enrolment uses (`enroll.tight_face_fallback`), so
    seed galleries land in the space the enrolled vectors live in — but
    the result is *marked*: live matching never produced a vector for
    such a crop, so a corpus frame must treat it as no-embedding.
    """
    if hasattr(embedder, "embed_best"):
        vector, quality = embedder.embed_best(image)
        if vector is not None:
            return Embedded(vector, quality)
        from siteloom.identity.enroll import tight_face_fallback

        return Embedded(tight_face_fallback(embedder, image), None, fallback=True)
    return Embedded(embedder.embed(image), None)


#: Save the embedding cache and poll for interruption this often during
#: an embed pass — the "commit batch" of a GPU job, so a Ctrl-C keeps
#: everything computed so far (the ProgressReporter contract).
_EMBED_BATCH = 100


def _save_cache(cache_file: Path, cached: dict[str, Embedded]) -> None:
    keys = list(cached)
    np.savez(
        cache_file,
        keys=np.array(keys, dtype=object),
        vecs=np.array([cached[k].vector for k in keys], dtype=object),
        quals=np.array(
            [
                float("nan") if cached[k].quality is None else cached[k].quality
                for k in keys
            ]
        ),
        fbs=np.array([cached[k].fallback for k in keys], dtype=bool),
    )


def embed_corpus(
    targets: set[tuple[str, str]],
    config: SiteConfig,
    cache_dir: Path,
    *,
    embedder_factory=None,
    tick=None,
    check=None,
) -> tuple[EmbeddingBank, dict[str, int]]:
    """Embed every (crop_path, algo) target once, cached on disk.

    The cache is what turns a sweep from a GPU job into arithmetic: the
    embedders are deterministic, so the vectors are computed on the
    first pass and every later variant — today or next week — reuses
    them byte-identically. One `.npz` per algo; the face file carries a
    hash of the projection matrix so retraining it busts the cache.

    `tick` fires once per target (cache hits included — a warm-cache
    run must not read as stalled on /jobs); `check` is polled every
    `_EMBED_BATCH` embeds, right after an incremental cache save, so an
    interrupt keeps what was computed.
    """
    factory = embedder_factory or _default_embedder_factory(config)
    bank: EmbeddingBank = {}
    stats = {"embedded": 0, "cached": 0, "missing_files": 0, "no_embedding": 0}
    cache_dir.mkdir(parents=True, exist_ok=True)

    by_algo: dict[str, list[str]] = defaultdict(list)
    for path, algo in targets:
        by_algo[algo].append(path)

    for algo, paths in sorted(by_algo.items()):
        tag = f"-{_projection_tag(config)}" if algo == "face" else ""
        cache_file = cache_dir / f"lab-emb-{algo}{tag}.npz"
        cached: dict[str, Embedded] = {}
        if cache_file.exists():
            data = np.load(cache_file, allow_pickle=True)
            fbs = data["fbs"] if "fbs" in data else [False] * len(data["keys"])
            cached = {
                str(k): Embedded(
                    v if v is not None else None,
                    None if q != q else float(q),
                    bool(fb),
                )
                for k, v, q, fb in zip(
                    data["keys"], data["vecs"], data["quals"], fbs
                )
            }
        needed = sorted({p for p in paths if p not in cached})
        embedder = None
        since_save = 0
        for path in needed:
            import cv2

            image = cv2.imread(path)
            if image is None:
                stats["missing_files"] += 1
                if tick:
                    tick()
                continue
            if embedder is None:
                embedder = factory(algo)
            entry = _embed_one(embedder, image)
            if entry.vector is not None:
                entry = entry._replace(
                    vector=np.asarray(entry.vector, dtype=np.float32)
                )
            cached[path] = entry
            stats["embedded"] += 1
            since_save += 1
            if tick:
                tick()
            if since_save >= _EMBED_BATCH:
                _save_cache(cache_file, cached)
                since_save = 0
                if check:
                    check()
        if since_save:
            _save_cache(cache_file, cached)
        if check and needed:
            check()
        needed_set = set(needed)
        for path in dict.fromkeys(paths):
            entry = cached.get(path)
            if entry is None:
                continue  # missing file, already counted
            if entry.vector is None:
                stats["no_embedding"] += 1
            stats["cached"] += 1
            bank[(path, algo)] = entry
            if tick and path not in needed_set:
                tick()  # cache hit — progress must still move
    return bank, stats


# -- seeding ----------------------------------------------------------------


@dataclass
class SeedIdentity:
    live_id: int
    identifier_key: str
    class_name: str
    label: str | None
    plate: str | None
    plate_source: str | None
    first_seen: datetime
    last_seen: datetime
    crop_paths: list[str] = field(default_factory=list)


def seed_plan(
    session: Session,
    identifier_keys: list[str],
    *,
    scope: str = "live",
    event_ids: list[int] | None = None,
    max_vectors: int = 20,
    exclude_event_ids: tuple[int, ...] = (),
) -> list[SeedIdentity]:
    """Which live identities the sandbox starts with, and from which crops.

    `scope="live"` seeds every identity of the replayed identifiers —
    the incident events matched against the *whole* neighbourhood, and a
    magnet identity that was never linked still stole score. `"linked"`
    restricts to identities claimed on the replayed events, for a
    smaller, faster sandbox. Crops come from `cover_candidates` (active
    claims' detections by confidence, then verified annotations), which
    is the closest DB-derivable approximation of the live gallery — and
    `max_vectors` above the live cap is a real experiment: seed the
    gallery the cap never allowed and see what it would have matched.

    `exclude_event_ids` — normally the replayed events themselves —
    keeps their crops out of the seed: the replay re-drives those
    frames, and a gallery pre-taught by the incident under test would
    both double-count them and smuggle in whatever the incident
    mis-learned. The sandbox starts from what each identity knew from
    *other* events.

    An identity with no usable crops still seeds when it carries a plate
    (plate matching needs no vectors); with neither, it is skipped — a
    zero-vector, plate-less identity is invisible to matching live too.
    """
    from siteloom.web.identity_ops import cover_candidates

    query = select(Identity).where(Identity.identifier_key.in_(identifier_keys))
    if scope == "linked":
        if not event_ids:
            raise LabError('seed scope "linked" needs event ids')
        query = (
            query.join(EventIdentity, EventIdentity.identity_id == Identity.id)
            .where(EventIdentity.event_id.in_(event_ids))
            .distinct()
        )
    plan: list[SeedIdentity] = []
    for identity in session.scalars(query):
        crops = cover_candidates(
            session,
            identity,
            limit=max_vectors,
            exclude_event_ids=exclude_event_ids,
        )
        if not crops and not identity.plate:
            continue
        plan.append(
            SeedIdentity(
                live_id=identity.id,
                identifier_key=identity.identifier_key,
                class_name=identity.class_name,
                label=identity.label,
                plate=identity.plate,
                plate_source=identity.plate_source,
                first_seen=identity.first_seen,
                last_seen=identity.last_seen,
                crop_paths=crops,
            )
        )
    return plan


def _sandbox_identity(sandbox: Session, seed: SeedIdentity) -> Identity:
    row = Identity(
        identifier_key=seed.identifier_key,
        class_name=seed.class_name,
        label=seed.label,
        plate=seed.plate,
        plate_source=seed.plate_source,
        first_seen=seed.first_seen,
        last_seen=seed.last_seen,
    )
    sandbox.add(row)
    sandbox.flush()
    return row


def seed_reembed(
    sandbox: Session,
    store: VectorStore,
    plan: list[SeedIdentity],
    bank: EmbeddingBank,
    algo_for: dict[str, str],
    *,
    max_vectors: int = 20,
) -> dict[int, int]:
    """Build the sandbox galleries by re-embedding the plan's crops.

    Returns sandbox identity id -> live identity id — the map that lets
    a variant's links be scored against the live verdicts.
    """
    id_map: dict[int, int] = {}
    for seed in plan:
        row = _sandbox_identity(sandbox, seed)
        algo = algo_for.get(seed.identifier_key, "generic")
        for crop in seed.crop_paths[:max_vectors]:
            entry = bank.get((crop, algo))
            if entry is None or entry.vector is None:
                continue
            # Fallback vectors are fine HERE: enrolment builds live
            # galleries with the same tight-crop fallback, so a seed
            # using it stays faithful to what live matching ran against.
            store.add(seed.identifier_key, entry.vector, row.id, crop_path=crop)
            row.vector_count += 1
        id_map[row.id] = seed.live_id
    sandbox.flush()
    return id_map


def seed_copy(
    live_vector_path: str,
    sandbox: Session,
    store: VectorStore,
    plan: list[SeedIdentity],
    *,
    max_vectors: int = 20,
    exclude_crop_paths: frozenset[str] | set[str] = frozenset(),
) -> dict[int, int]:
    """Exact-copy the live galleries into the sandbox.

    Opens the live embedded Qdrant directly, which takes its flock —
    possible only while no serve/ingest/backfill process holds it, and
    holding it briefly is also what guarantees no concurrent writer.
    Refuses with the alternative rather than waiting: the lab's default
    (`reembed`) works alongside a running site.

    `exclude_crop_paths` — the replayed events' own crops — keeps
    vectors the live resolver learned *from the incident under test*
    out of the seed, the same hygiene `seed_plan` applies to reembed;
    without it a copy pre-teaches the sandbox exactly what the incident
    mis-learned. Only filterable where provenance exists: a pre-CLD-84
    vector carries no crop_path and rides along, which is the "exact"
    in exact copy — the count of such vectors is worth reporting.
    """
    try:
        live = VectorStore(live_vector_path)
    except Exception as exc:  # embedded Qdrant raises on a held flock
        raise LabError(
            "--seed copy needs exclusive access to the live vector store "
            f"({live_vector_path}), which another process holds — stop "
            "serve/ingest (see /jobs) and retry, or use --seed reembed, "
            f"which runs alongside them. ({exc})"
        )
    try:
        id_map: dict[int, int] = {}
        for seed in plan:
            row = _sandbox_identity(sandbox, seed)
            kept = 0
            for vector, payload in live.identity_points(
                seed.identifier_key, seed.live_id
            ):
                if kept >= max_vectors:
                    break
                crop = payload.get("crop_path")
                if crop and crop in exclude_crop_paths:
                    continue
                store.add(seed.identifier_key, vector, row.id, crop_path=crop)
                row.vector_count += 1
                kept += 1
            id_map[row.id] = seed.live_id
        sandbox.flush()
        return id_map
    finally:
        live.close()


# -- config variants --------------------------------------------------------

#: Fields a --set may not touch: `algo` decides the vector space the
#: embedding bank was built in, and `applies_to` re-wires which frames
#: an identifier even sees — both silently invalidate the corpus.
_FROZEN_FIELDS = {"algo", "applies_to"}


def _parse_value(raw: str):
    text = raw.strip()
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def apply_overrides(identity_cfg: IdentityConfig, sets: list[str]) -> IdentityConfig:
    """A deep-copied IdentityConfig with `key.field=value` overrides applied.

    `person.threshold=0.85` targets an identifier field;
    `recency_window_s=60` a top-level one. The result is re-validated
    through pydantic, so a typo'd value fails loudly instead of running
    a sweep against a config that ignored it.
    """
    cfg = identity_cfg.model_copy(deep=True)
    for spec in sets:
        key, eq, raw = spec.partition("=")
        if not eq:
            raise LabError(f"--set wants key=value, got {spec!r}")
        value = _parse_value(raw)
        key = key.strip()
        if "." in key:
            ident_key, field_name = key.split(".", 1)
            ident = cfg.identifiers.get(ident_key)
            if ident is None:
                raise LabError(
                    f"unknown identifier {ident_key!r} — this config has: "
                    + ", ".join(sorted(cfg.identifiers))
                )
            if field_name in _FROZEN_FIELDS:
                raise LabError(
                    f"{field_name!r} cannot be swept: it changes the vector "
                    "space / frame wiring the corpus was embedded for"
                )
            if field_name not in type(ident).model_fields:
                raise LabError(
                    f"unknown field {field_name!r} on identifier "
                    f"{ident_key!r} — fields: "
                    + ", ".join(sorted(type(ident).model_fields))
                )
            setattr(ident, field_name, value)
        else:
            if key not in type(cfg).model_fields:
                raise LabError(f"unknown identity field {key!r}")
            setattr(cfg, key, value)
    try:
        return IdentityConfig.model_validate(cfg.model_dump())
    except Exception as exc:
        raise LabError(f"overrides produce an invalid config: {exc}")


# -- the replay itself ------------------------------------------------------


@dataclass
class Decision:
    """What one (frame, identifier) pass decided, and why.

    `top`/`runner` are recorded from a read-only candidate query *before*
    the resolver ran, so every frame carries its near-miss scores — the
    view CLD-251 wants — including frames the resolver then refused.
    """

    event_id: int
    detection_id: int
    ts: str
    identifier: str
    outcome: str  # matched|minted|pending|mint-budget|ambiguous|gated|no-embedding
    gate: str | None
    top: tuple[int, float] | None  # (sandbox identity id, score)
    runner: tuple[int, float] | None
    quality: float | None


@dataclass
class VariantResult:
    name: str
    seed_mode: str
    decisions: list[Decision]
    # sandbox identity id -> live identity id (seeded ones only)
    live_of: dict[int, int]
    # sandbox identity id -> label, for seeded identities that carry one
    labels: dict[int, str]
    # event -> {(identifier, sandbox identity id)} — what live ingest
    # would have linked (mints included: a mint creates a claim live).
    claims: dict[int, set[tuple[str, int]]]
    hits: Counter  # (event, sandbox identity) -> matched frames
    minted: dict[int, tuple[str, int]]  # sandbox id -> (identifier, event minted on)
    mints: Counter  # identifier -> count


def run_variant(
    name: str,
    identity_cfg: IdentityConfig,
    corpus: Corpus,
    bank: EmbeddingBank,
    seeder,
    *,
    config: SiteConfig,
    algo_for: dict[str, str],
    seed_mode: str = "reembed",
    face_quality: str = "detector",
    include_gated: bool = False,
    tick=None,
    check=None,
) -> VariantResult:
    """Drive the corpus through one resolver under one config.

    A fresh sandbox per variant: temp-dir Qdrant (never the shared
    store), in-memory SQLite, one real IdentityResolver. `seeder` is
    called with (sandbox session, store) and returns the sandbox->live
    identity map; pass None for an empty sandbox. `check` (interrupt
    poll) runs at event boundaries, right after each commit.

    Stated fidelity limit beyond seeding: the replay drives frames
    under their *post-merge* event ids, while live `resolve()` ran
    before the de-fragmentation merges — so on an event assembled from
    several fragments, live granted each fragment its own per-event
    learn/mint budget and verdict cutoff where the replay pools one.
    """
    cams: dict[str, CameraConfig] = {c.id: c for c in config.cameras}
    tmp = tempfile.mkdtemp(prefix="siteloom-lab-")
    store = VectorStore(tmp)
    try:
        engine = make_engine("sqlite://")
        init_db(engine)
        with get_session(engine)() as sandbox:
            live_of: dict[int, int] = seeder(sandbox, store) if seeder else {}
            labels = {
                sid: row.label
                for sid, row in (
                    (sid, sandbox.get(Identity, sid)) for sid in live_of
                )
                if row is not None and row.label
            }
            registry = IdentifierRegistry(identity_cfg)
            resolver = IdentityResolver(identity_cfg, store)

            decisions: list[Decision] = []
            claims: dict[int, set] = defaultdict(set)
            hits: Counter = Counter()
            minted: dict[int, tuple[str, int]] = {}
            mints: Counter = Counter()
            mint_spent: Counter = Counter()

            current_event = None
            for frame in corpus.frames:
                if frame.event_id != current_event:
                    sandbox.commit()  # live ingest commits per frame batch
                    if check:
                        check()
                    current_event = frame.event_id
                if tick:
                    tick()
                ts_text = frame.ts.isoformat(timespec="seconds")
                if frame.gated and not include_gated:
                    decisions.append(
                        Decision(
                            frame.event_id, frame.detection_id, ts_text, "*",
                            "gated", frame.gated, None, None, None,
                        )
                    )
                    continue
                cam = cams.get(frame.camera_id)
                for key, ident in registry.identifiers_for(frame.class_name):
                    entry = (
                        bank.get((frame.crop_path, algo_for.get(key, ident.algo)))
                        if frame.crop_path
                        else None
                    )
                    if entry is not None and entry.fallback:
                        # Enrolment's tight-crop fallback vector: live
                        # matching produced NO vector for this crop and
                        # ran no resolve, so for a corpus frame it is a
                        # no-embedding, not a decision to invent.
                        entry = None
                    vector = entry.vector if entry else None
                    face_score = entry.quality if entry else None
                    plate = frame.plates.get(key)
                    if vector is None and plate is None:
                        decisions.append(
                            Decision(
                                frame.event_id, frame.detection_id, ts_text,
                                key, "no-embedding", None, None, None, None,
                            )
                        )
                        continue

                    top = runner = None
                    if vector is not None:
                        ranked = store.search_identities(
                            key,
                            vector,
                            aggregation=ident.score_aggregation,
                            top_k=ident.score_top_k,
                        )
                        if ranked:
                            top = (ranked[0].identity_id, round(ranked[0].score, 4))
                        if len(ranked) > 1:
                            runner = (ranked[1].identity_id, round(ranked[1].score, 4))

                    quality = frame.confidence
                    if face_quality == "yunet" and ident.algo == "face":
                        quality = face_score

                    resolution = resolver.resolve(
                        sandbox,
                        identifier_key=key,
                        class_name=frame.class_name,
                        vector=None if vector is None else list(vector),
                        plate=plate,
                        timestamp=frame.ts,
                        crop_path=frame.crop_path,
                        threshold=identity_cfg.threshold_for(key, cam),
                        max_vectors=ident.max_vectors_per_identity,
                        camera_id=frame.camera_id,
                        quality=quality,
                        event_id=frame.event_id,
                    )

                    gate: str | None = None
                    if resolution.ambiguous:
                        outcome = "ambiguous"
                    elif resolution.pending:
                        budget_spent = (
                            ident.mint_max_per_event > 0
                            and plate is None
                            and mint_spent[(key, frame.event_id)]
                            >= ident.mint_max_per_event
                        )
                        outcome = "mint-budget" if budget_spent else "pending"
                    elif resolution.identity is None:
                        outcome = "unresolved"
                    elif resolution.is_new:
                        outcome = "minted"
                        mints[key] += 1
                        mint_spent[(key, frame.event_id)] += 1
                        minted[resolution.identity.id] = (key, frame.event_id)
                        claims[frame.event_id].add((key, resolution.identity.id))
                        if plate:
                            gate = "plate"
                        elif ident.min_sightings <= 1:
                            gate = "unconditional"
                        elif (
                            quality is not None
                            and quality >= ident.immediate_quality
                        ):
                            gate = "immediate_quality"
                        else:
                            gate = f"promotion({resolution.identity.vector_count})"
                    else:
                        outcome = "matched"
                        gate = resolution.matched_by
                        claims[frame.event_id].add((key, resolution.identity.id))
                        hits[(frame.event_id, resolution.identity.id)] += 1
                    decisions.append(
                        Decision(
                            frame.event_id, frame.detection_id, ts_text, key,
                            outcome, gate, top, runner,
                            None if quality is None else round(quality, 3),
                        )
                    )
            sandbox.commit()
            return VariantResult(
                name=name,
                seed_mode=seed_mode,
                decisions=decisions,
                live_of=live_of,
                labels=labels,
                claims={k: set(v) for k, v in claims.items()},
                hits=hits,
                minted=minted,
                mints=mints,
            )
    finally:
        store.close()
        shutil.rmtree(tmp, ignore_errors=True)


# -- scoring & comparison ---------------------------------------------------


def score_variant(result: VariantResult, corpus: Corpus) -> dict:
    """One variant's outcomes, per event and in total, JSON-shaped."""
    per_event: dict[int, dict] = {}
    outcome_totals: Counter = Counter()
    for decision in result.decisions:
        outcome_totals[decision.outcome] += 1
    for eid, info in corpus.events.items():
        linked = result.claims.get(eid, set())
        linked_live = {
            result.live_of[sid] for _, sid in linked if sid in result.live_of
        }
        minted_here = [
            sid for sid, (_, mint_event) in result.minted.items()
            if mint_event == eid
        ]
        wrong = sorted(
            live for live in linked_live if info.judged.get(live) == "wrong"
        )
        confirmed = sorted(
            live for live in linked_live if info.judged.get(live) == "confirmed"
        )
        missed_confirmed = sorted(
            live
            for live, verdict in info.judged.items()
            if verdict == "confirmed" and live not in linked_live
        )
        counts = Counter(
            d.outcome for d in result.decisions if d.event_id == eid
        )
        top_links = sorted(
            (
                (
                    result.labels.get(sid)
                    or (
                        f"live:{result.live_of[sid]}"
                        if sid in result.live_of
                        else f"mint:{sid}"
                    ),
                    hits,
                )
                for (hit_eid, sid), hits in result.hits.items()
                if hit_eid == eid
            ),
            key=lambda pair: -pair[1],
        )[:5]
        per_event[eid] = {
            "class": info.class_name,
            "camera": info.camera_id,
            "frames": info.frames,
            "claims": len(linked),
            "mints": len(minted_here),
            "linked_live": sorted(linked_live),
            "wrong_links": wrong,
            "confirmed_links": confirmed,
            "missed_confirmed": missed_confirmed,
            "outcomes": dict(counts),
            "top_links": top_links,
        }
    totals = {
        "mints": int(sum(result.mints.values())),
        "mints_by_identifier": dict(result.mints),
        "claims": int(sum(len(c) for c in result.claims.values())),
        "wrong_links": int(
            sum(len(e["wrong_links"]) for e in per_event.values())
        ),
        "confirmed_links": int(
            sum(len(e["confirmed_links"]) for e in per_event.values())
        ),
        "missed_confirmed": int(
            sum(len(e["missed_confirmed"]) for e in per_event.values())
        ),
        "outcomes": dict(outcome_totals),
    }
    return {
        "variant": result.name,
        "seed_mode": result.seed_mode,
        "totals": totals,
        "events": per_event,
    }


def compare(baseline: dict, candidate: dict) -> dict:
    """A verdict, not just a diff — the failure mode a sweep invites is
    reading one improved number and shipping (`track_eval.compare`'s
    lesson). Better means: no more mints, no more wrong links, no fewer
    confirmed links, and at least one strict improvement."""
    b, c = baseline["totals"], candidate["totals"]
    deltas = {
        key: (b[key], c[key])
        for key in ("mints", "claims", "wrong_links", "confirmed_links")
    }
    no_worse = (
        c["mints"] <= b["mints"]
        and c["wrong_links"] <= b["wrong_links"]
        and c["confirmed_links"] >= b["confirmed_links"]
    )
    strictly_better = (
        c["mints"] < b["mints"]
        or c["wrong_links"] < b["wrong_links"]
        or c["confirmed_links"] > b["confirmed_links"]
    )
    no_better = (
        c["mints"] >= b["mints"]
        and c["wrong_links"] >= b["wrong_links"]
        and c["confirmed_links"] <= b["confirmed_links"]
    )
    if no_worse and strictly_better:
        verdict = "BETTER"
    elif no_worse:
        verdict = "SAME"
    elif no_better:
        verdict = "WORSE"
    else:
        verdict = "MIXED"
    return {"verdict": verdict, "deltas": deltas}


def config_summary(identity_cfg: IdentityConfig, keys: list[str]) -> dict:
    """The knob values a report should pin next to its numbers."""
    out: dict[str, dict] = {}
    for key in keys:
        ident = identity_cfg.identifiers.get(key)
        if ident is None:
            continue
        out[key] = {
            "threshold": ident.threshold,
            "min_margin": ident.min_margin,
            "min_sightings": ident.min_sightings,
            "immediate_quality": ident.immediate_quality,
            "learn_min_quality": ident.learn_min_quality,
            "learn_max_per_event": ident.learn_max_per_event,
            "mint_max_per_event": ident.mint_max_per_event,
            "max_vectors_per_identity": ident.max_vectors_per_identity,
            "score_aggregation": ident.score_aggregation,
            "score_top_k": ident.score_top_k,
        }
    return out
