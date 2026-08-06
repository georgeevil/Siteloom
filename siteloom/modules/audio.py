"""Audio loud-duration module (PRD §6.5).

The NoiseAware/Minut model: an episode is RMS level above a threshold
sustained for a minimum duration. Levels are dBFS (relative to digital
full scale) — thresholds get calibrated per deployment, they are not
absolute SPL. Classification and transcription are intentionally NOT
here (transcription is legally gated, PRD §9/NFR5).

Job payload:
    media_path: str — file with an audio stream (video or audio file)
    base_time:  str (ISO) — timestamp of the file's start
Result:
    {"episodes": [{start_s, end_s, peak_db, mean_db}]}
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from siteloom.config import AudioConfig
from siteloom.dispatch.base import Job

SILENCE_DB = -90.0


def rms_db(samples: np.ndarray) -> float:
    """dBFS of a window of float samples in [-1, 1]."""
    if samples.size == 0:
        return SILENCE_DB
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * math.log10(rms) if rms > 1e-9 else SILENCE_DB


def detect_episodes(
    levels_db: list[float],
    *,
    window_s: float,
    threshold_db: float,
    min_duration_s: float,
    release_s: float,
) -> list[dict[str, float]]:
    """Find sustained loud episodes in a series of per-window dB levels.

    An episode opens when the level crosses threshold_db, tolerates up to
    release_s of continuous quiet without closing, and only counts if its
    loud span lasts at least min_duration_s.
    """
    episodes: list[dict[str, float]] = []
    start_idx: int | None = None
    last_loud_idx = -1
    release_windows = max(1, round(release_s / window_s))

    def _close(end_idx: int) -> None:
        nonlocal start_idx
        duration = (end_idx - start_idx + 1) * window_s
        if duration >= min_duration_s:
            span = levels_db[start_idx : end_idx + 1]
            episodes.append(
                {
                    "start_s": start_idx * window_s,
                    "end_s": (end_idx + 1) * window_s,
                    "peak_db": max(span),
                    "mean_db": sum(span) / len(span),
                }
            )
        start_idx = None

    for i, level in enumerate(levels_db):
        if level >= threshold_db:
            if start_idx is None:
                start_idx = i
            last_loud_idx = i
        elif start_idx is not None and i - last_loud_idx >= release_windows:
            _close(last_loud_idx)
    if start_idx is not None:
        _close(last_loud_idx)
    return episodes


def media_levels(media_path: str, window_s: float) -> list[float]:
    """Decode the first audio stream and return per-window dBFS levels."""
    import av

    levels: list[float] = []
    with av.open(media_path) as container:
        if not container.streams.audio:
            return levels
        stream = container.streams.audio[0]
        rate = stream.rate or 48000
        window_samples = max(1, int(rate * window_s))
        buffer = np.empty(0, dtype=np.float32)
        for frame in container.decode(stream):
            samples = frame.to_ndarray()
            if samples.dtype.kind == "i":  # normalize ints to [-1, 1]
                samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
            if samples.ndim > 1:  # downmix channels
                samples = samples.mean(axis=0)
            buffer = np.concatenate([buffer, samples.astype(np.float32)])
            while buffer.size >= window_samples:
                levels.append(rms_db(buffer[:window_samples]))
                buffer = buffer[window_samples:]
        if buffer.size:
            levels.append(rms_db(buffer))
    return levels


class AudioModule:
    def __init__(self, cfg: AudioConfig | None = None):
        self.cfg = cfg or AudioConfig()

    def process(self, job: Job) -> dict[str, Any]:
        levels = media_levels(job.payload["media_path"], self.cfg.window_s)
        episodes = detect_episodes(
            levels,
            window_s=self.cfg.window_s,
            threshold_db=self.cfg.threshold_db,
            min_duration_s=self.cfg.min_duration_s,
            release_s=self.cfg.release_s,
        )
        return {"episodes": episodes}
