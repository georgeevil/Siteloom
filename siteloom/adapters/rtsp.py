"""Generic RTSP fallback adapter (PRD §6.1).

Stream-only, no event API — but keeps the system from being blocked on a
brand without a dedicated adapter.
"""

from __future__ import annotations

from siteloom.adapters.base import CameraAdapter, FrameSource, StreamInfo


class RtspAdapter(CameraAdapter):
    def __init__(self, source: str):
        self._url = source

    def connect(self) -> None:
        pass  # nothing to authenticate; the URL carries credentials if any

    def list_streams(self) -> list[StreamInfo]:
        return [StreamInfo(id=self._url, name=self._url, kind="live")]

    def get_live_stream(self, stream_id: str) -> FrameSource:
        return FrameSource(stream_id, source_id=stream_id, is_file=False)
