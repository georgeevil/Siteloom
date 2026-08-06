from siteloom.adapters.base import CameraAdapter, FrameSource, StreamInfo
from siteloom.adapters.file import FileAdapter
from siteloom.adapters.rtsp import RtspAdapter
from siteloom.adapters.unifi import UniFiProtectAdapter

ADAPTERS = {
    "file": FileAdapter,
    "rtsp": RtspAdapter,
    "unifi": UniFiProtectAdapter,
}

__all__ = [
    "CameraAdapter",
    "FrameSource",
    "StreamInfo",
    "FileAdapter",
    "RtspAdapter",
    "UniFiProtectAdapter",
    "ADAPTERS",
]
