import math

import numpy as np

from siteloom.modules.audio import SILENCE_DB, detect_episodes, rms_db

LOUD, QUIET = -10.0, -60.0


def test_rms_db_full_scale_sine():
    t = np.linspace(0, 1, 48000, dtype=np.float32)
    sine = np.sin(2 * np.pi * 440 * t)
    assert abs(rms_db(sine) - 20 * math.log10(1 / math.sqrt(2))) < 0.1


def test_rms_db_silence():
    assert rms_db(np.zeros(1000, dtype=np.float32)) == SILENCE_DB


def _detect(levels, **kw):
    defaults = dict(window_s=1.0, threshold_db=-25.0, min_duration_s=30.0, release_s=5.0)
    defaults.update(kw)
    return detect_episodes(levels, **defaults)


def test_sustained_episode_detected():
    levels = [QUIET] * 10 + [LOUD] * 40 + [QUIET] * 10
    episodes = _detect(levels)
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep["start_s"] == 10.0
    assert ep["end_s"] == 50.0
    assert ep["peak_db"] == LOUD


def test_short_burst_ignored():
    """A door slam (loud but brief) must not create a noise event."""
    levels = [QUIET] * 10 + [LOUD] * 5 + [QUIET] * 40
    assert _detect(levels) == []


def test_brief_dip_does_not_split_episode():
    """A pause shorter than release_s keeps one continuous episode."""
    levels = [LOUD] * 20 + [QUIET] * 3 + [LOUD] * 20
    episodes = _detect(levels)
    assert len(episodes) == 1
    assert episodes[0]["end_s"] == 43.0


def test_long_gap_splits_and_short_halves_dropped():
    levels = [LOUD] * 20 + [QUIET] * 10 + [LOUD] * 20
    # each half is 20s < min_duration 30s -> nothing qualifies
    assert _detect(levels) == []
    # but with a lower min duration both halves count
    assert len(_detect(levels, min_duration_s=15.0)) == 2


def test_episode_running_at_end_of_file_closes():
    levels = [QUIET] * 5 + [LOUD] * 35
    episodes = _detect(levels)
    assert len(episodes) == 1
    assert episodes[0]["end_s"] == 40.0
