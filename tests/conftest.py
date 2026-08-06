from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """A tiny synthetic mp4: 30 frames, 320x240, with a moving rectangle."""
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (320, 240)
    )
    for i in range(30):
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        x = 20 + i * 8
        cv2.rectangle(frame, (x, 100), (x + 60, 200), (0, 200, 255), -1)
        writer.write(frame)
    writer.release()
    return path
