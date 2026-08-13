"""UniFi Protect adapter (PRD §6.1) via the `uiprotect` client.

`uiprotect` is asyncio-based; this adapter wraps it behind the synchronous
CameraAdapter interface. connect() starts a private event loop on a
background thread and keeps the authenticated client alive on it, so the
historical calls (event listing, clip export) reuse one session instead of
re-authenticating per request. close() tears both down.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from siteloom.adapters.base import CameraAdapter, FrameSource, StreamInfo
from siteloom.config import UniFiConfig

log = logging.getLogger(__name__)


@dataclass
class RecordingEvent:
    """One Protect NVR recording trigger (motion / smart detection).

    `id` is Protect's own event id — stable across queries, which is what
    makes backfill dedupe possible.
    """

    id: str
    camera_id: str
    type: str  # "motion" | "smartDetectZone" | "smartDetectLine" | ...
    start: datetime
    end: datetime
    score: int = 0
    smart_types: list[str] = field(default_factory=list)


class UniFiProtectAdapter(CameraAdapter):
    def __init__(self, source: str = "", *, unifi: UniFiConfig):
        # `source` (the per-camera config value) is the Protect camera id;
        # it is resolved in get_live_stream, not here, so one adapter
        # instance can serve every camera on the console.
        self._cfg = unifi
        self._client = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._rtsp_urls: dict[str, str] = {}
        self._names: dict[str, str] = {}

    # -- connection lifecycle ---------------------------------------------

    def connect(self) -> None:
        if self._loop is not None:
            return
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever, name="uiprotect-loop", daemon=True
        )
        thread.start()
        self._loop, self._thread = loop, thread
        try:
            self._run(self._connect())
        except Exception:
            self.close()
            raise

    def _run(self, coro):
        """Run a coroutine on the adapter's loop from sync code."""
        if self._loop is None:
            raise RuntimeError("adapter not connected")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _connect(self) -> None:
        from uiprotect import ProtectApiClient

        client = ProtectApiClient(
            self._cfg.host,
            self._cfg.port,
            self._cfg.username,
            self._cfg.password,
            api_key=self._cfg.api_key or None,
            verify_ssl=self._cfg.verify_ssl,
        )
        await client.update()
        self._client = client
        for cam_id, cam in client.bootstrap.cameras.items():
            self._names[cam_id] = cam.name or cam_id
            # Prefer the highest-quality channel with RTSP enabled.
            url = None
            for channel in cam.channels:
                if channel.is_rtsp_enabled and channel.rtsps_url:
                    url = channel.rtsps_url
                    break
            if url:
                self._rtsp_urls[cam_id] = url

    def close(self) -> None:
        if self._loop is None:
            return
        if self._client is not None:
            try:
                self._run(self._client.close_session())
            except Exception:  # pragma: no cover — best-effort teardown
                log.debug("uiprotect session close failed", exc_info=True)
            self._client = None
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop.close()
        self._loop = self._thread = None

    def nvr_timezone(self) -> str:
        """The NVR's own IANA timezone name (CLD-100).

        `uiprotect` exposes it as a `ZoneInfo` on the bootstrap's NVR
        model; the `.key` is the IANA name the site setting stores. The
        NVR already knows what wall clock its cameras live on, which is
        what makes this the second rung of the timezone resolution chain.
        """
        if self._client is None:
            raise RuntimeError("adapter not connected")
        zone = self._client.bootstrap.nvr.timezone
        key = getattr(zone, "key", None)
        if not key:
            raise RuntimeError(f"NVR reports no usable timezone ({zone!r})")
        return str(key)

    # -- live -------------------------------------------------------------

    def list_streams(self) -> list[StreamInfo]:
        return [
            StreamInfo(id=cam_id, name=self._names.get(cam_id, cam_id), kind="live")
            for cam_id in self._rtsp_urls
        ]

    def get_live_stream(self, stream_id: str) -> FrameSource:
        url = self._rtsp_urls.get(stream_id)
        if url is None:
            raise KeyError(
                f"camera {stream_id!r} not found or has no RTSP-enabled channel; "
                f"known cameras: {list(self._rtsp_urls)}"
            )
        return FrameSource(url, source_id=stream_id, is_file=False)

    # -- historical (PRD §6.6) --------------------------------------------

    #: Protect event types that mean "the NVR recorded something moving".
    #: Rings, audio detections etc. are deliberately absent — they carry
    #: no video worth running the visual pipeline over.
    RECORDING_EVENT_TYPES = ("motion", "smartDetectZone", "smartDetectLine")

    def list_recording_events(
        self,
        start: datetime,
        end: datetime,
        stream_id: str | None = None,
    ) -> list[RecordingEvent]:
        """Motion/smart-detect windows the NVR recorded in [start, end].

        These windows are what makes historical backfill tractable:
        Protect already did motion gating 24/7, so indexing its recordings
        means processing minutes per day, not the day itself.
        """
        from uiprotect.data import EventType

        types = [EventType(t) for t in self.RECORDING_EVENT_TYPES]
        events = self._run(
            self._client.get_events(start=start, end=end, types=types, sorting="asc")
        )
        out = []
        for ev in events:
            if ev.camera_id is None or ev.end is None:
                continue  # device-less or still in progress
            if stream_id is not None and ev.camera_id != stream_id:
                continue
            out.append(
                RecordingEvent(
                    id=str(ev.id),
                    camera_id=ev.camera_id,
                    type=ev.type.value,
                    start=ev.start,
                    end=ev.end,
                    score=ev.score or 0,
                    smart_types=[t.value for t in (ev.smart_detect_types or [])],
                )
            )
        return out

    def download_clip(
        self, stream_id: str, start: datetime, end: datetime, output: Path
    ) -> Path:
        """Export the NVR's recording for [start, end] as an MP4 file.

        Protect's export boundaries are approximate (±a few seconds);
        callers stamping frame times should use `start` as the base and
        accept that fuzz rather than trying to correct for it.
        """
        output.parent.mkdir(parents=True, exist_ok=True)
        self._run(
            self._client.get_camera_video(stream_id, start, end, output_file=output)
        )
        return output

    def get_historical_clip(
        self, stream_id: str, start: datetime, end: datetime
    ) -> FrameSource:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        # mkstemp is only used for a collision-free name: close its fd
        # immediately (uiprotect writes the file itself) and hand the
        # file to the FrameSource, which deletes it once consumed/closed.
        fd, name = tempfile.mkstemp(suffix=".mp4", prefix="siteloom-clip-")
        os.close(fd)
        path = Path(name)
        try:
            self.download_clip(stream_id, start, end, path)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return FrameSource(
            str(path),
            source_id=stream_id,
            is_file=True,
            base_time=start,
            owns_file=True,
        )
