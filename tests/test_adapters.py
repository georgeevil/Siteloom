import pytest

from siteloom.adapters import FileAdapter


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
