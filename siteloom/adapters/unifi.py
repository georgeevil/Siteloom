"""UniFi Protect adapter (PRD §6.1) via the `uiprotect` client.

`uiprotect` is asyncio-based; this adapter wraps it behind the synchronous
CameraAdapter interface by resolving each camera's RTSPS stream URL once,
then handing frames off to OpenCV like any other stream.
"""

from __future__ import annotations

import asyncio

from siteloom.adapters.base import CameraAdapter, FrameSource, StreamInfo
from siteloom.config import UniFiConfig


class UniFiProtectAdapter(CameraAdapter):
    def __init__(self, source: str = "", *, unifi: UniFiConfig):
        # `source` (the per-camera config value) is the Protect camera id;
        # it is resolved in get_live_stream, not here, so one adapter
        # instance can serve every camera on the console.
        self._cfg = unifi
        self._client = None
        self._rtsp_urls: dict[str, str] = {}
        self._names: dict[str, str] = {}

    def connect(self) -> None:
        asyncio.run(self._connect())

    async def _connect(self) -> None:
        from uiprotect import ProtectApiClient

        client = ProtectApiClient(
            self._cfg.host,
            self._cfg.port,
            self._cfg.username,
            self._cfg.password,
            verify_ssl=self._cfg.verify_ssl,
        )
        await client.update()
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
        await client.close_session()

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
