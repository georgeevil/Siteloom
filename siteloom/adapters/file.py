"""FileAdapter: local video files as a camera source.

Serves two purposes: development/testing without live cameras, and the
seed of the backfill module (PRD §6.6) — backfill is "iterate a directory
through the same pipeline as live ingest," not a parallel path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from collections.abc import Iterator

import cv2

from siteloom.adapters.base import CameraAdapter, Frame, FrameSource, StreamInfo

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}


class ImageSource(FrameSource):
    """A single still photo as a one-frame source, so photo archives run
    through the identical pipeline as video (PRD §6.6)."""

    def __init__(self, path: Path):
        super().__init__(str(path), source_id=str(path), is_file=True)
        self._path = path

    def frames(self, sample_fps: float = 2.0) -> Iterator[Frame]:
        image = cv2.imread(str(self._path))
        if image is None:
            raise IOError(f"could not read image {self._path}")
        ts = datetime.fromtimestamp(self._path.stat().st_mtime, tz=timezone.utc)
        yield Frame(image=image, timestamp=ts, source_id=str(self._path))


class FileAdapter(CameraAdapter):
    def __init__(self, source: str):
        self._source = Path(source)
        self._files: list[Path] = []

    def connect(self) -> None:
        if not self._source.exists():
            raise FileNotFoundError(self._source)
        if self._source.is_dir():
            self._files = sorted(
                p
                for p in self._source.rglob("*")
                if p.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS
            )
        else:
            self._files = [self._source]

    def list_streams(self) -> list[StreamInfo]:
        return [StreamInfo(id=str(p), name=p.name, kind="file") for p in self._files]

    def get_live_stream(self, stream_id: str) -> FrameSource:
        path = Path(stream_id)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() in IMAGE_EXTS:
            return ImageSource(path)
        # Use the file's mtime as the clip's base timestamp so backfilled
        # events land at roughly the right point on the timeline.
        base = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return FrameSource(str(path), source_id=str(path), is_file=True, base_time=base)
