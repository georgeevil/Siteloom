"""Live-view frame hub: one RTSP reader per camera, shared by all viewers.

Browsers cannot play RTSP, so the live page streams MJPEG. The naive
version — one FFmpeg session per <img> tag — multiplies camera and CPU
load by the number of open tabs; instead each camera gets at most one
reader thread, holding the latest JPEG, and every viewer is fanned out
from that. Readers start on the first viewer and stop shortly after the
last one leaves, so an unwatched console costs nothing.

This is view-only plumbing: nothing here touches the detection pipeline,
which keeps its own per-camera ingest connections.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import cv2

from siteloom.config import CameraConfig, SiteConfig
from siteloom.ingest import build_adapter

log = logging.getLogger(__name__)

#: A reader with no viewers keeps running this long (page reloads and
#: layout switches reconnect within seconds; re-opening RTSP is slow).
IDLE_STOP_S = 20.0
RECONNECT_S = 3.0
#: Viewer-facing rate/size caps: the console is a monitor wall, not a
#: recorder — 8 fps at 1280 wide reads as live and keeps a 3×3 grid of
#: 4K cameras from saturating a LAN.
VIEW_FPS = 8.0
MAX_WIDTH = 1280
JPEG_QUALITY = 70
OPEN_TIMEOUT_MS = 15_000


class _Feed(threading.Thread):
    """One camera's reader: RTSP -> latest JPEG, under a condition."""

    def __init__(self, hub: "LiveHub", cam: CameraConfig):
        super().__init__(name=f"live-{cam.id}", daemon=True)
        self.hub = hub
        self.cam = cam
        self.cond = threading.Condition()
        self.frame: bytes | None = None
        self.seq = 0
        self.error = ""
        self.clients = 0
        self.last_client = time.monotonic()
        self.stopping = threading.Event()

    def _done(self) -> bool:
        return self.stopping.is_set() or (
            self.clients == 0 and time.monotonic() - self.last_client > IDLE_STOP_S
        )

    def run(self) -> None:
        while not self._done():
            try:
                url = self.hub._resolve(self.cam)
                self._pump(url)
            except Exception as exc:
                with self.cond:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.cond.notify_all()
                log.warning("live feed %s: %s", self.cam.id, self.error)
                # Protect rotates stream URLs; a failed session's URL is
                # the first suspect.
                self.hub._invalidate(self.cam.id)
            self.stopping.wait(RECONNECT_S)
        self.hub._drop(self.cam.id, self)
        with self.cond:
            self.cond.notify_all()  # release any waiter watching a dying feed

    def _pump(self, url: str) -> None:
        cap = cv2.VideoCapture(
            url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                OPEN_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                OPEN_TIMEOUT_MS,
            ],
        )
        if not cap.isOpened():
            raise IOError(f"could not open stream for camera {self.cam.id}")
        try:
            interval = 1.0 / VIEW_FPS
            next_at = 0.0
            while not self._done():
                if not cap.grab():  # grab every frame to keep buffers drained
                    raise IOError("stream ended")
                now = time.monotonic()
                if now < next_at:
                    continue
                ok, image = cap.retrieve()
                if not ok:
                    raise IOError("frame decode failed")
                next_at = now + interval
                h, w = image.shape[:2]
                if w > MAX_WIDTH:
                    image = cv2.resize(image, (MAX_WIDTH, round(h * MAX_WIDTH / w)))
                ok, jpeg = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if not ok:
                    continue
                with self.cond:
                    self.frame = jpeg.tobytes()
                    self.seq += 1
                    self.error = ""
                    self.cond.notify_all()
        finally:
            cap.release()


def _has_new_frame(feed: _Feed, seen: int) -> Callable[[], bool]:
    """A `Condition.wait_for` predicate bound to *this* feed and *this* seq.

    `LiveHub.frames` rebinds both names inside its loop — a feed that dies
    is replaced mid-stream, and `seen` advances with every frame — and a
    closure captures the variable, not its value. An inline lambda would
    therefore ask whichever feed the loop currently holds about whichever
    seq it currently has: on a multi-camera site, the wrong camera's
    reader (CLD-99). Passing them through a call freezes both.
    """
    return lambda: feed.seq != seen


class HubStopped(RuntimeError):
    """Raised instead of starting a reader once the hub is going down."""


class LiveHub:
    def __init__(self, config: SiteConfig):
        self.config = config
        self._feeds: dict[str, _Feed] = {}
        self._lock = threading.Lock()
        self._urls: dict[str, str] = {}
        self._resolve_lock = threading.Lock()
        #: Set once, by `stop()`, and never cleared: a hub that has been
        #: released belongs to a process that is exiting. `frames()` and
        #: `snapshot()` both watch it, which is the only thing that can
        #: end an MJPEG response — see the note on `frames`.
        self._shutdown = threading.Event()

    def cameras(self) -> list[CameraConfig]:
        """Cameras with a live stream to show (files have none)."""
        return [c for c in self.config.cameras if c.adapter != "file"]

    # -- URL resolution ----------------------------------------------------

    def _resolve(self, cam: CameraConfig) -> str:
        if cam.adapter == "rtsp":
            return cam.source
        with self._resolve_lock:
            url = self._urls.get(cam.id)
            if url is None:
                adapter = build_adapter(cam, self.config)
                adapter.connect()
                try:
                    # One Protect bootstrap knows every camera — cache all
                    # of them so the next tile skips this round-trip.
                    for other in self.cameras():
                        if other.adapter != cam.adapter:
                            continue
                        try:
                            self._urls[other.id] = adapter.get_live_stream(
                                other.source
                            ).target
                        except KeyError:
                            log.warning(
                                "live: camera %s has no RTSP-enabled channel",
                                other.id,
                            )
                finally:
                    adapter.close()
                url = self._urls.get(cam.id)
            if url is None:
                raise KeyError(f"camera {cam.id} has no live stream")
            return url

    def _invalidate(self, cam_id: str) -> None:
        with self._resolve_lock:
            self._urls.pop(cam_id, None)

    # -- feed lifecycle ----------------------------------------------------

    def _ensure(self, camera_id: str) -> _Feed:
        cam = next((c for c in self.cameras() if c.id == camera_id), None)
        if cam is None:
            raise KeyError(f"no live camera {camera_id!r}")
        if self._shutdown.is_set():
            # `stop()` joins the readers it stopped; a viewer that lost the
            # race and rebuilt one here would resurrect the thread we are
            # shutting down, after the join had already passed it.
            raise HubStopped("live hub is shutting down")
        with self._lock:
            feed = self._feeds.get(camera_id)
            if feed is None or not feed.is_alive():
                feed = _Feed(self, cam)
                self._feeds[camera_id] = feed
                feed.start()
            return feed

    def _drop(self, camera_id: str, feed: _Feed) -> None:
        with self._lock:
            if self._feeds.get(camera_id) is feed:
                del self._feeds[camera_id]

    def stop(self) -> None:
        """Stop every reader **and release every viewer**.

        The flag is set first, before a single feed is touched: `frames()`
        reads it, so a viewer whose feed dies underneath it returns
        instead of rebuilding the reader this call is trying to stop.

        Each feed is then notified as well as stopped. Without that, a
        viewer parked in `cond.wait_for(..., timeout=2.0)` notices on its
        next timeout; with it, shutdown is immediate — which is the
        difference between a Ctrl-C that returns the prompt and one that
        appears to hang.
        """
        self._shutdown.set()
        with self._lock:
            feeds = list(self._feeds.values())
        for feed in feeds:
            feed.stopping.set()
            with feed.cond:
                feed.cond.notify_all()
        for feed in feeds:
            feed.join(timeout=5)

    def viewers(self) -> dict[str, int]:
        """Open viewer count per camera — what a shutdown is waiting on.

        uvicorn's "Waiting for connections to close" names no connection,
        which is unhelpful precisely when it matters. `serve` logs this
        alongside it so the operator can see that the thing holding the
        server open is two browser tiles and not the database.
        """
        with self._lock:
            return {cid: f.clients for cid, f in self._feeds.items() if f.clients}

    # -- viewers -----------------------------------------------------------

    def frames(self, camera_id: str):
        """Yield JPEG frames as they arrive; blocks between frames.

        Runs in the serving threadpool for as long as the viewer keeps
        the response open; the finally block is what lets an idle feed
        notice its last viewer left.

        **The `_shutdown` checks are load-bearing, not defensive** (CLD-132).
        This loop only *yields* when a new frame arrives, so a feed that is
        alive but silent — a stalled camera, a reader between reconnects —
        spins here without ever returning from `next()`. Starlette drives a
        sync generator from an anyio worker thread, and those threads are
        not daemons: a generator that never returns is an interpreter that
        cannot exit, no matter how the shutdown was asked for. Checking a
        flag every ≤2 s is what bounds that.
        """
        feed = self._ensure(camera_id)
        with feed.cond:
            feed.clients += 1
        try:
            seen = 0
            while not self._shutdown.is_set():
                with feed.cond:
                    feed.cond.wait_for(_has_new_frame(feed, seen), timeout=2.0)
                    frame, seq = feed.frame, feed.seq
                    alive = feed.is_alive() and not feed.stopping.is_set()
                if seq != seen and frame is not None:
                    seen = seq
                    yield frame
                elif not alive:
                    if self._shutdown.is_set():
                        return  # going down for good — do not rebuild
                    # Lost a race with idle-stop; rebuild onto a fresh feed.
                    with feed.cond:
                        feed.clients -= 1
                        feed.last_client = time.monotonic()
                    feed = self._ensure(camera_id)
                    with feed.cond:
                        feed.clients += 1
                    seen = 0
        finally:
            with feed.cond:
                feed.clients -= 1
                feed.last_client = time.monotonic()

    def snapshot(self, camera_id: str, timeout: float = 12.0) -> bytes | None:
        """One current JPEG (poster images, quick health checks)."""
        feed = self._ensure(camera_id)
        deadline = time.monotonic() + timeout
        with feed.cond:
            feed.clients += 1
        try:
            while time.monotonic() < deadline and not self._shutdown.is_set():
                with feed.cond:
                    if feed.frame is not None:
                        return feed.frame
                    feed.cond.wait(timeout=0.5)
                if not feed.is_alive():
                    return None
            return None
        finally:
            with feed.cond:
                feed.clients -= 1
                feed.last_client = time.monotonic()
