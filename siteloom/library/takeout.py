"""Google Photos Takeout importer.

Takeout gives each media file a JSON sidecar containing, among other
things, a `people` array — the names from Google Photos' own face
grouping. That is a genuinely valuable training signal, but note what it
is NOT: the tags are per *photo*, never per face box. A photo of three
people tagged with two names does not say which box is whom.

So assignment runs in two passes, which is what makes the ambiguous
majority usable:

  Pass 1 — unambiguous. Exactly one detected face and exactly one person
           tag: that face is that person. Logically certain (modulo a
           wrong detection), so these are auto-verified by default and
           become the seed gallery.
  Pass 2 — constrained matching. For photos with several faces and/or
           several names, embed each face and match it against the pass-1
           gallery *restricted to the names tagged on that same photo*.
           Closed-set matching over 2-5 candidates is far easier than
           open-set recognition, so this recovers most of the archive.
           These are proposals only — they always require human review.

Everything lands in Annotation rows as `proposed_name` + `proposal_basis`
with verified=False (except pass 1). Nothing reaches the identity store
or a training export until a human confirms it in the UI.

Sidecar naming is genuinely messy across Takeout versions (truncation,
`(1)` counters that migrate into the JSON name, older `.json` suffixes),
so matching is done primarily by the sidecar's own `title` field, with
filename heuristics as fallback.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from siteloom.adapters.file import IMAGE_EXTS, VIDEO_EXTS
from siteloom.store import Annotation, ItemTag, LibraryItem  # noqa: F401

log = logging.getLogger(__name__)

SIDECAR_SUFFIXES = (
    ".supplemental-metadata.json",
    ".supplemental-meta.json",
    ".suppl.json",
    ".json",
)


@dataclass
class Sidecar:
    path: Path
    title: str
    people: list[str]
    taken_at: datetime | None
    description: str = ""


@dataclass
class ImportStats:
    items_seen: int = 0
    sidecars_matched: int = 0
    people_tags: int = 0
    faces_detected: int = 0
    unambiguous: int = 0
    matched: int = 0
    unresolved: int = 0
    skipped_derivative: int = 0
    people: dict[str, int] = field(default_factory=lambda: defaultdict(int))


# Google Photos exports edited copies as "<name>-edited.jpg" with no
# sidecar of their own. They depict the same moment as the original, so
# importing both would seed the gallery with near-duplicates — which
# inflates per-person coverage counts and biases a fine-tune toward
# whichever photos happened to be edited.
DERIVATIVE_MARKERS = ("-edited", "-EDITED", "-edit")


def is_derivative(path: Path) -> bool:
    return any(marker in path.stem for marker in DERIVATIVE_MARKERS)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parse_sidecar(path: Path) -> Sidecar | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError) as exc:
        log.debug("unreadable sidecar %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or "title" not in data:
        return None  # album metadata.json, print-subscriptions.json, etc.
    people = [
        p["name"].strip()
        for p in data.get("people", []) or []
        if isinstance(p, dict) and p.get("name", "").strip()
    ]
    taken_at = None
    for key in ("photoTakenTime", "creationTime"):
        stamp = (data.get(key) or {}).get("timestamp")
        if stamp:
            try:
                taken_at = datetime.fromtimestamp(int(stamp), tz=timezone.utc).replace(
                    tzinfo=None
                )
                break
            except (ValueError, OSError):
                pass
    return Sidecar(
        path=path,
        title=str(data.get("title", "")),
        people=people,
        taken_at=taken_at,
        description=str(data.get("description", "") or ""),
    )


def index_sidecars(directory: Path) -> dict[str, Sidecar]:
    """Map media filename -> sidecar for one directory.

    Keyed on the sidecar's `title` (authoritative) and on filename-derived
    guesses (for sidecars whose title was itself truncated).
    """
    index: dict[str, Sidecar] = {}
    for json_path in directory.glob("*.json"):
        sidecar = parse_sidecar(json_path)
        if sidecar is None:
            continue
        if sidecar.title:
            index.setdefault(sidecar.title, sidecar)
        # Filename form: strip the sidecar suffix to recover the media name.
        name = json_path.name
        for suffix in SIDECAR_SUFFIXES:
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                index.setdefault(stem, sidecar)
                # Counter migration: "IMG_1.jpg(1)" sidecar <-> "IMG_1(1).jpg" media
                if stem.endswith(")") and "(" in stem:
                    base, _, counter = stem.rpartition("(")
                    media_stem = Path(base)
                    index.setdefault(
                        f"{media_stem.stem}({counter[:-1]}){media_stem.suffix}", sidecar
                    )
                break
    return index


def find_sidecar(media: Path, index: dict[str, Sidecar]) -> Sidecar | None:
    if media.name in index:
        return index[media.name]
    # Takeout truncates long names; fall back to longest prefix match.
    candidates = [
        (key, sc)
        for key, sc in index.items()
        if key and (media.name.startswith(key) or key.startswith(media.stem))
    ]
    if candidates:
        return max(candidates, key=lambda kv: len(kv[0]))[1]
    return None


class TakeoutImporter:
    """Imports a Takeout tree into the library with face proposals."""

    def __init__(self, indexer, face_embedder=None, auto_verify_unambiguous=True):
        self.indexer = indexer
        self.Session = indexer.Session
        self.config = indexer.config
        self._face = face_embedder
        self.auto_verify_unambiguous = auto_verify_unambiguous
        self.faces_dir = indexer.media_dir / "library" / "faces"
        self.faces_dir.mkdir(parents=True, exist_ok=True)

    @property
    def face(self):
        if self._face is None:
            from siteloom.identity.embedders import FaceEmbedder

            self._face = FaceEmbedder(
                projection_path=self.config.identity.face_projection_path or None
            )
        return self._face

    def import_tree(
        self,
        root: str | Path,
        limit: int | None = None,
        name: str = "",
        batch_size: int = 200,
        include_derivatives: bool = False,
    ) -> ImportStats:
        """Register items + people tags, then run the two assignment passes.

        Resumable at item granularity: items that already carry face
        annotations are skipped, and work is committed every
        `batch_size` items. A 30k-photo archive can therefore be
        interrupted and resumed without losing progress.
        """
        root = Path(root).expanduser().resolve()
        source = self.indexer.add_source(root, name=name or root.name, kind="takeout")
        stats = ImportStats()

        media_files: list[Path] = []
        sidecar_indexes: dict[Path, dict[str, Sidecar]] = {}
        for file in sorted(root.rglob("*")):
            if file.suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS:
                continue
            if not include_derivatives and is_derivative(file):
                stats.skipped_derivative += 1
                continue
            media_files.append(file)
            if file.parent not in sidecar_indexes:
                sidecar_indexes[file.parent] = index_sidecars(file.parent)
            if limit is not None and len(media_files) >= limit:
                break

        log.info(
            "registering %d media files (%d edited derivatives skipped)",
            len(media_files),
            stats.skipped_derivative,
        )

        # -- register items + people tags ---------------------------------
        with self.Session() as session:
            for i, media in enumerate(media_files, 1):
                stats.items_seen += 1
                sidecar = find_sidecar(media, sidecar_indexes[media.parent])
                item = session.scalar(select(LibraryItem).filter_by(path=str(media)))
                if item is None:
                    item = LibraryItem(
                        source_id=source.id,
                        path=str(media),
                        kind=(
                            "image"
                            if media.suffix.lower() in IMAGE_EXTS
                            else "video"
                        ),
                        mtime=datetime.fromtimestamp(media.stat().st_mtime),
                        status="pending",
                    )
                    session.add(item)
                    session.flush()
                if sidecar is None:
                    continue
                stats.sidecars_matched += 1
                if sidecar.taken_at:
                    item.taken_at = sidecar.taken_at
                existing = {t.value for t in item.tags if t.kind == "person"}
                for person in sidecar.people:
                    stats.people[person] += 1
                    if person not in existing:
                        session.add(
                            ItemTag(item_id=item.id, kind="person", value=person)
                        )
                        stats.people_tags += 1
                if i % (batch_size * 10) == 0:
                    session.commit()
                    log.info("registered %d/%d", i, len(media_files))
            session.commit()

        # -- pass 1: detect every face once, name the certain ones --------
        gallery = self._pass_one(stats, batch_size)
        # -- pass 2: constrained matching over the saved crops ------------
        self._pass_two(gallery, stats, batch_size)
        return stats

    # ------------------------------------------------------------------

    def _tagged_item_ids(self, session: Session) -> list[int]:
        """Ids only — holding 23k ORM objects across a long run is what
        makes large imports fall over."""
        return list(
            session.scalars(
                select(LibraryItem.id)
                .join(ItemTag, ItemTag.item_id == LibraryItem.id)
                .filter(ItemTag.kind == "person")
                .distinct()
            ).all()
        )

    def _detect_faces(self, item: LibraryItem) -> list[tuple[np.ndarray, np.ndarray]]:
        """[(yunet_row, embedding)] for every face found in the item."""
        if item.kind != "image":
            return []  # video face import is a later pass; photos first
        image = cv2.imread(item.path)
        if image is None:
            return []
        out = []
        for face in self.face.detect(image):
            embedding = self.face.embed_face(image, face)
            if embedding is not None:
                out.append((face, embedding, image))
        return out

    def _face_annotation(
        self,
        session: Session,
        item: LibraryItem,
        face: np.ndarray,
        image: np.ndarray,
        seq: int,
        name: str | None,
        basis: str | None,
        verified: bool,
        similarity: float = 0.0,
    ) -> Annotation:
        h, w = image.shape[:2]
        x, y, fw, fh = face[:4]
        x1, y1 = max(0.0, float(x)), max(0.0, float(y))
        x2, y2 = min(float(w), x1 + float(fw)), min(float(h), y1 + float(fh))
        crop = image[int(y1) : int(y2), int(x1) : int(x2)]
        crop_path = None
        if crop.size:
            path = self.faces_dir / f"{item.id}_{seq}.jpg"
            cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            crop_path = str(path)
        annotation = Annotation(
            item_id=item.id,
            frame_index=0,
            bbox=json.dumps([x1 / w, y1 / h, x2 / w, y2 / h]),
            class_name="face",
            confidence=float(face[-1]),
            proposed_name=name,
            proposal_basis=basis,
            source="import",
            verified=verified,
            crop_path=crop_path,
            created_at=_now(),
        )
        session.add(annotation)
        return annotation

    def _pass_one(
        self, stats: ImportStats, batch_size: int = 200
    ) -> dict[str, list[np.ndarray]]:
        """Detect every face once and name the certain ones.

        Faces on ambiguous photos are persisted here too (with no
        proposed_name), so pass 2 never has to decode a full photo again —
        it works from the small saved face crops instead.
        """
        gallery: dict[str, list[np.ndarray]] = defaultdict(list)
        with self.Session() as session:
            item_ids = self._tagged_item_ids(session)
            log.info("pass 1: %d tagged items", len(item_ids))
            done = 0
            for item_id in item_ids:
                item = session.get(LibraryItem, item_id)
                if item is None:
                    continue
                names = [t.value for t in item.tags if t.kind == "person"]
                if any(a.class_name == "face" for a in item.annotations):
                    # Already processed — rebuild the gallery from prior
                    # certain assignments so a resumed run matches as well
                    # as a fresh one.
                    for a in item.annotations:
                        if (
                            a.class_name == "face"
                            and a.proposal_basis == "unambiguous"
                            and a.proposed_name
                            and a.crop_path
                        ):
                            vector = self._embed_crop(a.crop_path)
                            if vector is not None:
                                gallery[a.proposed_name].append(vector)
                    continue

                faces = self._detect_faces(item)
                stats.faces_detected += len(faces)
                certain = len(faces) == 1 and len(names) == 1
                for fi, (face, embedding, image) in enumerate(faces):
                    self._face_annotation(
                        session,
                        item,
                        face,
                        image,
                        fi,
                        name=names[0] if certain else None,
                        basis="unambiguous" if certain else None,
                        verified=certain and self.auto_verify_unambiguous,
                    )
                    if certain:
                        gallery[names[0]].append(embedding)
                        stats.unambiguous += 1
                done += 1
                if done % batch_size == 0:
                    session.commit()
                    log.info(
                        "pass 1: %d/%d items, %d faces, %d certain",
                        done,
                        len(item_ids),
                        stats.faces_detected,
                        stats.unambiguous,
                    )
            session.commit()
        log.info(
            "pass 1 done: %d unambiguous faces across %d people",
            stats.unambiguous,
            len(gallery),
        )
        return gallery

    def _embed_crop(self, crop_path: str) -> np.ndarray | None:
        """Embed an already-saved face crop (no full-photo decode)."""
        crop = cv2.imread(crop_path)
        if crop is None:
            return None
        faces = self.face.detect(crop)
        if faces:
            return self.face.embed_face(crop, max(faces, key=lambda f: f[-1]))
        # The crop is already tightly bound to a face; embed it directly
        # rather than losing the sample to a failed re-detection.
        try:
            resized = cv2.resize(crop, (112, 112))
            feature = (
                self.face._recognizer.feature(resized).flatten().astype(np.float32)
            )
            return self.face._finish(feature)
        except cv2.error:
            return None

    def _pass_two(
        self,
        gallery: dict[str, list[np.ndarray]],
        stats: ImportStats,
        batch_size: int = 200,
    ) -> None:
        """Name the faces pass 1 left blank, matching against the gallery
        but restricted to the names tagged on that same photo."""
        threshold = self.config.identity.identifiers["face"].threshold
        with self.Session() as session:
            pending = list(
                session.scalars(
                    select(Annotation.item_id)
                    .filter(
                        Annotation.class_name == "face",
                        Annotation.proposed_name.is_(None),
                        Annotation.rejected.is_(False),
                    )
                    .distinct()
                ).all()
            )
            log.info("pass 2: %d items with unnamed faces", len(pending))
            done = 0
            for item_id in pending:
                item = session.get(LibraryItem, item_id)
                if item is None:
                    continue
                names = [t.value for t in item.tags if t.kind == "person"]
                faces = [
                    a
                    for a in item.annotations
                    if a.class_name == "face"
                    and a.proposed_name is None
                    and not a.rejected
                ]
                if not faces or not names:
                    stats.unresolved += len(faces)
                    continue
                # Names already fixed on this photo can't be reused.
                used_names = {
                    a.proposed_name
                    for a in item.annotations
                    if a.class_name == "face" and a.proposed_name
                }
                embeddings = {
                    a.id: self._embed_crop(a.crop_path) for a in faces if a.crop_path
                }
                candidates = {
                    n: gallery[n]
                    for n in names
                    if n not in used_names and gallery.get(n)
                }

                scores = []
                for annotation in faces:
                    embedding = embeddings.get(annotation.id)
                    if embedding is None:
                        continue
                    for name, vectors in candidates.items():
                        best = max(float(embedding @ v) for v in vectors)
                        scores.append((best, annotation.id, name))
                scores.sort(reverse=True)

                assigned: dict[int, tuple[str, float]] = {}
                for score, annotation_id, name in scores:
                    if score < threshold or annotation_id in assigned:
                        continue
                    if name in used_names:
                        continue
                    assigned[annotation_id] = (name, score)
                    used_names.add(name)

                for annotation in faces:
                    name, score = assigned.get(annotation.id, (None, 0.0))
                    basis = "matched" if name else None
                    if name is None:
                        remaining = [n for n in names if n not in used_names]
                        # One face, one name still free: safe even with no
                        # gallery evidence.
                        if len(faces) == 1 and len(remaining) == 1:
                            name, basis = remaining[0], "single-candidate"
                            used_names.add(name)
                    if name:
                        annotation.proposed_name = name
                        annotation.proposal_basis = basis
                        stats.matched += 1
                        embedding = embeddings.get(annotation.id)
                        if embedding is not None:
                            gallery[name].append(embedding)
                    else:
                        stats.unresolved += 1
                done += 1
                if done % batch_size == 0:
                    session.commit()
                    log.info(
                        "pass 2: %d/%d items, %d named, %d unresolved",
                        done,
                        len(pending),
                        stats.matched,
                        stats.unresolved,
                    )
            session.commit()
        log.info(
            "pass 2 done: %d proposals, %d faces left unassigned",
            stats.matched,
            stats.unresolved,
        )
