"""FileAdapter: local video files as a camera source.

Serves two purposes: development/testing without live cameras, and the
seed of the backfill module (PRD §6.6) — backfill is "iterate a directory
through the same pipeline as live ingest," not a parallel path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from siteloom.adapters.base import CameraAdapter, FrameSource, StreamInfo

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


class FileAdapter(CameraAdapter):
    def __init__(self, source: str):
        self._source = Path(source)
        self._files: list[Path] = []

    def connect(self) -> None:
        if not self._source.exists():
            raise FileNotFoundError(self._source)
        if self._source.is_dir():
            self._files = sorted(
                p for p in self._source.rglob("*") if p.suffix.lower() in VIDEO_EXTS
            )
        else:
            self._files = [self._source]

    def list_streams(self) -> list[StreamInfo]:
        return [StreamInfo(id=str(p), name=p.name, kind="file") for p in self._files]

    def get_live_stream(self, stream_id: str) -> FrameSource:
        path = Path(stream_id)
        if not path.exists():
            raise FileNotFoundError(path)
        # Use the file's mtime as the clip's base timestamp so backfilled
        # events land at roughly the right point on the timeline.
        base = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return FrameSource(str(path), source_id=str(path), is_file=True, base_time=base)
