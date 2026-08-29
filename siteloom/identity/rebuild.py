"""Reset and re-embed the vector store after a poisoning change (CLD-106).

Some settings are migrations in disguise: after `crop_margin` or the
face projection moves, stored vectors and new ones are incomparable and
matching degrades with no error anywhere. The recorded decision this
module implements: **reset vectors, keep labels** — Identity rows, their
names and the stored crops survive; the vectors are derived data,
rebuilt from those crops.

The order of operations is the safety property:

1. **Catalogue** — scroll every collection for provenance (payload
   `crop_path`, CLD-84) and supplement from the database's own
   detection/annotation crops. This is also where the honest numbers
   come from: points with no crop anywhere (API-enrolled, pre-CLD-84)
   are *unrecoverable* and are counted, never silently dropped.
2. **Clear** — drop every collection, zero every `vector_count`, write
   the new stamp. From this commit the store is degraded-and-honest:
   identities read as unenrolled. An interruption after this point can
   only leave collections empty or partially refilled in the NEW space
   — never a mix of two spaces, which is the disease being cured
   (a half-mixed collection matches worse than either space alone and
   reports nothing).
3. **Re-embed** — additively, per identity, batch-committed, with a
   done-log beside the manifest so an interrupted run resumes instead
   of restarting (the `Annotation.enrolled` pattern, file-shaped).
   Embedders resolve per *algorithm* (the `identifier_embedder` rule)
   so every identifier on an algorithm lands in one space.
4. **Reconcile** — `vector_count` refreshed from the store, the drift
   authority (`identity_ops.refresh_vector_count`'s rule).

Pending pools are dropped and not rebuilt — they are transient evidence
with a TTL, and old-space evidence must not found new-space mints. The
`class-examples` collection is dropped and rebuilt inline via the
existing `CustomClassifier.rebuild`, which already re-embeds from
annotation crops.

Everything is injectable (sessions, store, embedder factory) so the
tests run a full rebuild with a fake 8-dimensional embedder against a
real embedded qdrant in a temp dir — no model weights.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select

from siteloom.identity.enroll import embed_crop_file
from siteloom.identity.space import compute_stamp, write_stamp
from siteloom.store import Identity

log = logging.getLogger(__name__)

#: Collections the re-embed skips: pending pools are TTL'd transient
#: evidence, and class-examples has its own rebuild path.
_PENDING_SUFFIX = "-pending"
CLASS_EXAMPLES = "class-examples"


@dataclass
class RebuildPlan:
    """What a rebuild would do — the confirm screen's numbers."""

    collections: dict[str, int] = field(default_factory=dict)
    identities: int = 0
    recoverable_points: int = 0
    unrecoverable_points: int = 0
    #: identity manifest: [{identifier_key, identity_id, crops: [...]}]
    entries: list[dict] = field(default_factory=list)

    @property
    def total_points(self) -> int:
        return self.recoverable_points + self.unrecoverable_points


def plan_rebuild(session, vectors, config) -> RebuildPlan:
    """Catalogue the store: what exists, what can be re-embedded from
    where, and what cannot. Read-only."""
    plan = RebuildPlan()
    per_identity: dict[tuple[str, int], list[str]] = {}

    for name in vectors.collection_names():
        points = vectors.scroll_all(name)
        plan.collections[name] = len(points)
        if name.endswith(_PENDING_SUFFIX) or name == CLASS_EXAMPLES:
            continue
        for point in points:
            payload = point.payload or {}
            identity_id = payload.get("identity_id")
            crop = payload.get("crop_path")
            if identity_id is None:
                plan.unrecoverable_points += 1
                continue
            key = (name, int(identity_id))
            per_identity.setdefault(key, [])
            if crop and Path(crop).is_file():
                per_identity[key].append(crop)
                plan.recoverable_points += 1
            else:
                plan.unrecoverable_points += 1

    # The database knows crops the payloads may not (older points, and
    # identities whose gallery was thin): detections behind standing
    # links, then verified annotations — best first, the cover rule.
    from siteloom.web.identity_ops import cover_candidates

    for identity in session.execute(select(Identity)).scalars():
        key = (identity.identifier_key, identity.id)
        known = per_identity.setdefault(key, [])
        cap = _max_vectors_for(config, identity.identifier_key)
        if len(known) >= cap:
            continue
        for crop in cover_candidates(session, identity, limit=cap * 2):
            if crop not in known and Path(crop).is_file():
                known.append(crop)
            if len(known) >= cap:
                break

    for (identifier_key, identity_id), crops in sorted(per_identity.items()):
        plan.entries.append({
            "identifier_key": identifier_key,
            "identity_id": identity_id,
            "crops": crops[: _max_vectors_for(config, identifier_key)],
        })
    plan.identities = len(plan.entries)
    return plan


def _max_vectors_for(config, identifier_key: str) -> int:
    ident = config.identity.identifiers.get(identifier_key)
    return ident.max_vectors_per_identity if ident else 20


@dataclass
class RebuildReport:
    identities: int = 0
    vectors_written: int = 0
    crops_unreadable: int = 0
    unrecoverable_points: int = 0
    resumed_past: int = 0


def run_rebuild(
    Session,
    vectors,
    config,
    *,
    progress,
    embedder_for: Callable[[str], Any] | None = None,
    work_dir: str | Path | None = None,
    resume: bool = False,
) -> RebuildReport:
    """Execute the reset-and-rebuild. `progress` is a ProgressReporter;
    every phase heartbeats and honours interrupts. `embedder_for` maps an
    identifier key to an embedder (defaults to the shared
    `identifier_embedder` rule); tests inject a fake."""
    if embedder_for is None:
        from siteloom.web.identity_ops import identifier_embedder

        cache: dict = {}

        def embedder_for(key: str):
            return identifier_embedder(config, key, cache)

    work = Path(work_dir or Path(config.storage.media_dir) / "identity-rebuild")
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "manifest.json"
    done_path = work / "manifest.done"
    report = RebuildReport()

    with Session() as session:
        if resume and manifest_path.is_file():
            entries = json.loads(manifest_path.read_text())["entries"]
        else:
            with progress.phase("Cataloguing vectors"):
                plan = plan_rebuild(session, vectors, config)
                report.unrecoverable_points = plan.unrecoverable_points
                entries = plan.entries
                # The manifest is written BEFORE anything is dropped —
                # it is the only copy of the provenance once the
                # collections are gone. Biometric-adjacent (identity ids
                # + crop paths): it lives under media_dir and nothing
                # serves it.
                manifest_path.write_text(json.dumps({
                    "stamp": compute_stamp(config),
                    "unrecoverable_points": plan.unrecoverable_points,
                    "entries": entries,
                }, indent=2))
                done_path.write_text("")

            with progress.phase("Clearing collections"):
                for name in vectors.collection_names():
                    vectors.drop(name)
                for identity in session.execute(select(Identity)).scalars():
                    identity.vector_count = 0
                write_stamp(config.identity.vector_db_path, compute_stamp(config))
                session.commit()
                # Degraded-and-honest from here: empty galleries in the
                # new space, never a mix of two spaces.

        done = set(done_path.read_text().split()) if done_path.is_file() else set()
        with progress.phase("Re-embedding from stored crops", total=len(entries)):
            embedders: dict[str, Any] = {}
            for entry in entries:
                marker = f"{entry['identifier_key']}:{entry['identity_id']}"
                if marker in done:
                    report.resumed_past += 1
                    progress.advance(skipped=1)
                    continue
                embedder = embedders.get(entry["identifier_key"])
                if embedder is None:
                    embedder = embedders[entry["identifier_key"]] = embedder_for(
                        entry["identifier_key"]
                    )
                identity = session.get(Identity, entry["identity_id"])
                written = 0
                for crop in entry["crops"]:
                    vector = embed_crop_file(embedder, crop)
                    if vector is None:
                        report.crops_unreadable += 1
                        continue
                    vectors.add(
                        entry["identifier_key"], vector,
                        entry["identity_id"], crop_path=crop,
                    )
                    written += 1
                if identity is not None:
                    identity.vector_count = written
                report.vectors_written += written
                report.identities += 1
                session.commit()
                with done_path.open("a") as fh:
                    fh.write(marker + "\n")
                progress.advance(identities=1, vectors=written)
                progress.check_interrupt()

        with progress.phase("Reconciling"):
            # The store is the count authority after a bulk write.
            from siteloom.web.identity_ops import refresh_vector_count

            for identity in session.execute(select(Identity)).scalars():
                refresh_vector_count(session, vectors, identity)
            session.commit()

            # Custom classes re-embed from their annotation crops via the
            # machinery that already owns them.
            try:
                from siteloom.identity.classes import CustomClassifier

                generic = embedder_for("_generic")

                def embed_crop(path):
                    return embed_crop_file(generic, path)

                CustomClassifier(vectors).rebuild(session, embed_crop)
            except Exception:  # pragma: no cover — degrades to empty
                log.exception("class-examples rebuild failed; run "
                              "`siteloom classes rebuild` separately")
    return report
