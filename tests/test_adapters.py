import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from siteloom.adapters import FileAdapter
from siteloom.adapters.base import FrameSource
from siteloom.adapters.unifi import UniFiProtectAdapter
from siteloom.config import UniFiConfig


def test_file_adapter_single_file(sample_video):
    adapter = FileAdapter(str(sample_video))
    adapter.connect()
    streams = adapter.list_streams()
    assert len(streams) == 1
    assert streams[0].kind == "file"


def test_file_adapter_directory(sample_video):
    adapter = FileAdapter(str(sample_video.parent))
    adapter.connect()
    assert [s.name for s in adapter.list_streams()] == [sample_video.name]


def test_file_adapter_missing_path():
    adapter = FileAdapter("/nonexistent/path")
    with pytest.raises(FileNotFoundError):
        adapter.connect()


def test_frame_sampling_rate(sample_video):
    """30 frames @15fps sampled at 5fps → every 3rd frame → 10 frames."""
    adapter = FileAdapter(str(sample_video))
    adapter.connect()
    source = adapter.get_live_stream(str(sample_video))
    frames = list(source.frames(sample_fps=5.0))
    assert len(frames) == 10
    assert frames[0].image.shape == (240, 320, 3)
    # timestamps increase monotonically
    ts = [f.timestamp for f in frames]
    assert ts == sorted(ts)


# -- historical clip lifetime (CLD-60) ------------------------------------
#
# get_historical_clip downloads to a temp MP4 the FrameSource owns; the
# file must be gone once the source is consumed/closed, with no live NVR
# involved — download_clip is stubbed to copy the synthetic sample video.

T0 = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)


class ClipStubAdapter(UniFiProtectAdapter):
    """Serves clip exports from the sample video without a console."""

    def __init__(self, sample_video):
        super().__init__(unifi=UniFiConfig())
        self._sample = sample_video

    def download_clip(self, stream_id, start, end, output):
        shutil.copy(self._sample, output)
        return output


def test_historical_clip_deleted_after_consumption(sample_video):
    adapter = ClipStubAdapter(sample_video)
    source = adapter.get_historical_clip("cam-1", T0, T0 + timedelta(seconds=2))
    path = Path(source.target)
    assert path.exists()
    frames = list(source.frames(sample_fps=5.0))
    assert frames  # the clip actually decoded before being discarded
    assert frames[0].timestamp >= T0  # base_time stamped from clip start
    assert not path.exists()
    source.close()  # idempotent after frames() already cleaned up


def test_historical_clip_context_manager_deletes_unconsumed(sample_video):
    adapter = ClipStubAdapter(sample_video)
    with adapter.get_historical_clip(
        "cam-1", T0, T0 + timedelta(seconds=2)
    ) as source:
        path = Path(source.target)
        assert path.exists()
    # never iterated — close() must still remove the download
    assert not path.exists()


def test_historical_clip_failed_download_leaves_no_file(sample_video):
    class FailingAdapter(ClipStubAdapter):
        def download_clip(self, stream_id, start, end, output):
            self.attempted = output
            raise IOError("export failed")

    adapter = FailingAdapter(sample_video)
    with pytest.raises(IOError):
        adapter.get_historical_clip("cam-1", T0, T0 + timedelta(seconds=2))
    assert not adapter.attempted.exists()


def test_unowned_source_never_deletes_its_file(sample_video):
    # Backfill hands FrameSource paths it manages itself (its own
    # TemporaryDirectory) — consuming/closing must not touch them.
    source = FrameSource(str(sample_video), source_id="cam-1", is_file=True)
    list(source.frames(sample_fps=5.0))
    source.close()
    assert sample_video.exists()
