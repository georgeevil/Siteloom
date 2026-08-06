"""Camera adapter layer (PRD §6.1).

One adapter per brand behind a shared interface. Adapters produce
`FrameSource`s — anything OpenCV can decode (RTSP URL, file path) — so the
rest of the pipeline never knows what brand a frame came from.
"""

from __future__ import annotations

import abc
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np


@dataclass
class StreamInfo:
    id: str
    name: str
    kind: str  # "live" | "file"


@dataclass
class Frame:
    """A sampled frame plus enough metadata to build a serializable Job."""

    image: np.ndarray  # BGR
    timestamp: datetime
    source_id: str


class FrameSource:
    """Wraps a cv2.VideoCapture and yields frames sampled at a target fps.

    For live streams every frame is grabbed (to keep the buffer drained)
    but only decoded/yielded at the sample rate. For files we seek by
    grab()-skipping so decoding cost tracks the sample rate, not the
    native frame rate.
    """

    def __init__(
        self,
        target: str,
        source_id: str,
        *,
        is_file: bool = False,
        base_time: datetime | None = None,
    ):
        self._target = target
        self._is_file = is_file
        self.source_id = source_id
        self._base_time = base_time

    @property
    def target(self) -> str:
        """The URL or file path this source reads (e.g. for cleanup of
        temporary clip files after processing)."""
        return self._target

    #: Live-stream open/read timeouts. Without them a network drop leaves
    #: grab() blocked inside FFmpeg forever, so the ingest reconnect loop
    #: never regains control; with them the stream ends and the caller
    #: reconnects.
    LIVE_TIMEOUT_MS = 15_000

    def frames(self, sample_fps: float = 2.0) -> Iterator[Frame]:
        if self._is_file:
            cap = cv2.VideoCapture(self._target)
        else:
            cap = cv2.VideoCapture(
                self._target,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    self.LIVE_TIMEOUT_MS,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    self.LIVE_TIMEOUT_MS,
                ],
            )
        if not cap.isOpened():
            raise IOError(f"could not open video source {self._target!r}")
        try:
            if self._is_file:
                yield from self._file_frames(cap, sample_fps)
            else:
                yield from self._live_frames(cap, sample_fps)
        finally:
            cap.release()

    def _file_frames(self, cap: cv2.VideoCapture, sample_fps: float) -> Iterator[Frame]:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(native_fps / sample_fps))
        base = self._base_time or datetime.now(timezone.utc)
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                return
            if idx % step == 0:
                ok, image = cap.retrieve()
                if not ok:
                    return
                ts = base.timestamp() + idx / native_fps
                yield Frame(
                    image=image,
                    timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
                    source_id=self.source_id,
                )
            idx += 1

    def _live_frames(self, cap: cv2.VideoCapture, sample_fps: float) -> Iterator[Frame]:
        interval = 1.0 / sample_fps
        next_at = time.monotonic()
        while True:
            ok = cap.grab()  # always grab, so the stream buffer stays drained
            if not ok:
                return
            now = time.monotonic()
            if now >= next_at:
                ok, image = cap.retrieve()
                if not ok:
                    return
                next_at = now + interval
                yield Frame(
                    image=image,
                    timestamp=datetime.now(timezone.utc),
                    source_id=self.source_id,
                )


class CameraAdapter(abc.ABC):
    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def list_streams(self) -> list[StreamInfo]: ...

    @abc.abstractmethod
    def get_live_stream(self, stream_id: str) -> FrameSource: ...

    def get_historical_clip(
        self, stream_id: str, start: datetime, end: datetime
    ) -> FrameSource:
        raise NotImplementedError(f"{type(self).__name__} has no historical access")

    def close(self) -> None:  # noqa: B027 — optional hook
        pass
