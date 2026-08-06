"""Local media library indexing.

Two deliberately separate phases, which is what makes indexing *partial*
and resumable over archives too large to process in one sitting:

1. `scan()`  — cheap. Walks the directory, inserts a LibraryItem row per
   file with status="pending". Re-running only adds new/changed files.
2. `process()` — expensive. Takes the next N pending items, runs the
   detection module over them (and the identity module where enabled),
   writes Annotations, flips status to "indexed".

Because phase 2 is bounded by `limit`, you can index 200 photos, label
them, come back and index 200 more; nothing is lost or repeated. Items
that fail are marked "failed" with the error, not silently dropped, so
they surface in the UI instead of vanishing.

The detection pass reuses the same JobDispatcher and modules as live
ingestion — improvements to detection accuracy apply to both (PRD §6.6).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from siteloom.adapters.file import IMAGE_EXTS, VIDEO_EXTS
from siteloom.config import SiteConfig
from siteloom.dispatch import Job, JobDispatcher
from siteloom.store import (
    Annotation,
    LibraryItem,
    LibrarySource,
)

log = logging.getLogger(__name__)

THUMB_MAX = 480


@dataclass
class ScanResult:
    added: int
    updated: int
    skipped: int
    total_pending: int


@dataclass
class ProcessResult:
    processed: int
    failed: int
    annotations: int
    remaining: int


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class LibraryIndexer:
    def __init__(
        self,
        config: SiteConfig,
        session_factory,
        dispatcher: JobDispatcher,
        resolver=None,
    ):
        self.config = config
        self.Session = session_factory
        self.dispatcher = dispatcher
        self.resolver = resolver
        self.media_dir = Path(config.storage.media_dir)
        self.thumbs_dir = self.media_dir / "library" / "thumbs"
        self.crops_dir = self.media_dir / "library" / "crops"
        for d in (self.thumbs_dir, self.crops_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- phase 1 -----------------------------------------------------------

    def add_source(
        self, path: str | Path, name: str = "", kind: str = "directory"
    ) -> LibrarySource:
        path = Path(path).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(path)
        with self.Session() as session:
            source = session.scalar(select(LibrarySource).filter_by(path=str(path)))
            if source is None:
                source = LibrarySource(
                    path=str(path),
                    name=name or path.name,
                    kind=kind,
                    added_at=_now(),
                )
                session.add(source)
                session.commit()
            return source

    def scan(self, source_id: int, limit: int | None = None) -> ScanResult:
        """Register files as pending items. Cheap — no decoding."""
        added = updated = skipped = 0
        with self.Session() as session:
            source = session.get(LibrarySource, source_id)
            if source is None:
                raise KeyError(f"no library source {source_id}")
            root = Path(source.path)
            known = {
                path: (item_id, mtime)
                for path, item_id, mtime in session.execute(
                    select(LibraryItem.path, LibraryItem.id, LibraryItem.mtime)
                    .filter_by(source_id=source_id)
                )
            }
            for file in sorted(root.rglob("*")):
                suffix = file.suffix.lower()
                if suffix not in IMAGE_EXTS | VIDEO_EXTS:
                    continue
                if limit is not None and added >= limit:
                    break
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                existing = known.get(str(file))
                if existing is None:
                    session.add(
                        LibraryItem(
                            source_id=source_id,
                            path=str(file),
                            kind="image" if suffix in IMAGE_EXTS else "video",
                            mtime=mtime,
                            status="pending",
                        )
                    )
                    added += 1
                elif existing[1] != mtime:
                    # File changed on disk — re-index it, drop stale auto
                    # annotations but keep human work (handled in process).
                    item = session.get(LibraryItem, existing[0])
                    item.mtime = mtime
                    item.status = "pending"
                    updated += 1
                else:
                    skipped += 1
            source.last_indexed_at = _now()
            session.commit()
            pending = session.scalar(
                select(func.count())
                .select_from(LibraryItem)
                .filter_by(source_id=source_id, status="pending")
            )
        return ScanResult(added, updated, skipped, pending or 0)

    # -- phase 2 -----------------------------------------------------------

    def process(
        self,
        source_id: int | None = None,
        limit: int = 100,
        identify: bool = True,
        progress=None,
    ) -> ProcessResult:
        """Run detection (+ identification) over pending items."""
        from siteloom.library.takeout import _NullProgress

        progress = progress or _NullProgress()
        processed = failed = annotation_count = 0
        with self.Session() as session:
            q = select(LibraryItem).filter_by(status="pending")
            if source_id is not None:
                q = q.filter_by(source_id=source_id)
            items = session.scalars(q.order_by(LibraryItem.id).limit(limit)).all()

            with progress.phase("Indexing media", total=len(items)):
                for item in items:
                    boxes = 0
                    try:
                        boxes = self._process_item(session, item, identify)
                        annotation_count += boxes
                        item.status = "indexed"
                        item.error = None
                        processed += 1
                    except Exception as exc:
                        item.status = "failed"
                        item.error = f"{type(exc).__name__}: {exc}"
                        failed += 1
                        log.warning("indexing failed for %s: %s", item.path, exc)
                    item.indexed_at = _now()
                    session.commit()
                    progress.advance(
                        boxes=boxes, failed=1 if item.status == "failed" else 0
                    )
                    progress.check_interrupt()

            remaining = session.scalar(
                select(func.count()).select_from(LibraryItem).filter_by(status="pending")
            )
        return ProcessResult(processed, failed, annotation_count, remaining or 0)

    def _process_item(self, session: Session, item: LibraryItem, identify: bool) -> int:
        # Drop stale machine annotations; human work is never discarded.
        for existing in list(item.annotations):
            if existing.source == "auto" and not existing.verified:
                session.delete(existing)

        frames = self._frames_for(item)
        count = 0
        for frame_index, image in frames:
            if frame_index == 0:
                item.height, item.width = image.shape[:2]
                item.thumb_path = self._save_thumb(item, image)
            count += self._detect_frame(session, item, frame_index, image, identify)
        return count

    def _frames_for(self, item: LibraryItem) -> list[tuple[int, "cv2.Mat"]]:
        """Images give one frame; short videos give a few evenly spaced
        frames — enough to catch everyone present without processing
        every frame of an archive."""
        if item.kind == "image":
            image = cv2.imread(item.path)
            if image is None:
                raise IOError("could not decode image")
            return [(0, image)]

        cap = cv2.VideoCapture(item.path)
        if not cap.isOpened():
            raise IOError("could not open video")
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            item.duration_s = total / fps if fps else 0.0
            n = self.config.library.video_frames
            targets = (
                [0]
                if total <= 1
                else [round(i * (total - 1) / max(1, n - 1)) for i in range(n)]
            )
            frames = []
            for target in sorted(set(targets)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, image = cap.read()
                if ok:
                    frames.append((target, image))
            if not frames:
                raise IOError("no readable frames")
            return frames
        finally:
            cap.release()

    def _detect_frame(
        self,
        session: Session,
        item: LibraryItem,
        frame_index: int,
        image,
        identify: bool,
    ) -> int:
        ok, jpeg = cv2.imencode(".jpg", image)
        if not ok:
            raise IOError("frame encode failed")
        result = self.dispatcher.submit_and_wait(
            Job(
                module="detection",
                payload={
                    "image_jpeg": jpeg.tobytes(),
                    "camera_id": f"library-{item.source_id}",
                    "timestamp": (item.taken_at or item.mtime).isoformat(),
                    "zones": [],
                    "require_zone": False,
                },
            )
        )
        if not result.ok:
            raise RuntimeError(result.error or "detection failed")

        h, w = image.shape[:2]
        count = 0
        for det in result.result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            crop_path = self._save_crop(item, frame_index, count, det)
            annotation = Annotation(
                item_id=item.id,
                frame_index=frame_index,
                bbox=json.dumps([x1 / w, y1 / h, x2 / w, y2 / h]),
                class_name=det["class_name"],
                confidence=det["confidence"],
                source="auto",
                crop_path=crop_path,
                created_at=_now(),
            )
            session.add(annotation)
            session.flush()
            if identify and self.resolver is not None and det.get("crop_jpeg"):
                self._identify_annotation(session, annotation, det)
            count += 1
        return count

    def _identify_annotation(
        self, session: Session, annotation: Annotation, det: dict
    ) -> None:
        """Resolve identity for a library crop against the SAME identity
        store live camera events use — a face enrolled from a photo
        archive is recognized on a camera immediately."""
        result = self.dispatcher.submit_and_wait(
            Job(
                module="identity",
                payload={
                    "crop_jpeg": det["crop_jpeg"],
                    "class_name": det["class_name"],
                },
            )
        )
        if not result.ok:
            log.warning("identity failed on library crop: %s", result.error)
            return
        registry = self.config.identity.identifiers
        for emb in result.result["embeddings"]:
            ident_cfg = registry.get(emb["identifier"])
            resolution = self.resolver.resolve(
                session,
                identifier_key=emb["identifier"],
                class_name=det["class_name"],
                vector=emb["vector"],
                plate=emb["plate"],
                timestamp=_now(),
                crop_path=annotation.crop_path,
                threshold=ident_cfg.threshold if ident_cfg else None,
                max_vectors=ident_cfg.max_vectors_per_identity if ident_cfg else 20,
            )
            # Face identity wins the annotation link when present; it is
            # the strongest signal available for a person crop.
            if annotation.identity_id is None or emb["identifier"] == "face":
                annotation.identity_id = resolution.identity.id

    # -- media helpers -----------------------------------------------------

    def _save_thumb(self, item: LibraryItem, image) -> str:
        h, w = image.shape[:2]
        scale = min(1.0, THUMB_MAX / max(h, w))
        thumb = (
            cv2.resize(image, (round(w * scale), round(h * scale)))
            if scale < 1.0
            else image
        )
        path = self.thumbs_dir / f"{item.id}.jpg"
        cv2.imwrite(str(path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return str(path)

    def _save_crop(
        self, item: LibraryItem, frame_index: int, seq: int, det: dict
    ) -> str | None:
        crop_jpeg = det.get("crop_jpeg")
        if not crop_jpeg:
            return None
        path = self.crops_dir / f"{item.id}_{frame_index}_{seq}.jpg"
        path.write_bytes(crop_jpeg)
        return str(path)
