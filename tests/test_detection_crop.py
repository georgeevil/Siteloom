"""Crop geometry (no YOLO involved — expand_box is the whole contract).

The crop is stored for display *and* fed to the identity embedders, so
what this function returns is what a detection looks like everywhere.
"""

from __future__ import annotations

from siteloom.config import DetectionConfig
from siteloom.modules.detection import expand_box


def test_zero_margin_is_the_box_itself():
    assert expand_box((10.0, 20.0, 30.0, 60.0), 0.0, 640, 480) == (10, 20, 30, 60)


def test_margin_grows_each_side_by_a_fraction_of_that_dimension():
    # 100 wide, 200 tall, 10% margin -> 10px left/right, 20px top/bottom.
    assert expand_box((100.0, 100.0, 200.0, 300.0), 0.1, 640, 480) == (
        90,
        80,
        210,
        320,
    )


def test_shape_is_preserved_so_a_person_stays_tall():
    x1, y1, x2, y2 = expand_box((100.0, 100.0, 140.0, 260.0), 0.25, 640, 480)
    assert (x2 - x1) / (y2 - y1) == (140 - 100) / (260 - 100)


def test_clamped_to_the_frame():
    assert expand_box((2.0, 1.0, 60.0, 40.0), 0.5, 64, 48) == (0, 0, 64, 48)


def test_rounds_outward_so_the_box_is_never_eaten():
    # Fractional bounds must floor down / ceil up, not truncate inward.
    assert expand_box((10.6, 10.6, 20.4, 20.4), 0.0, 640, 480) == (10, 10, 21, 21)


def test_degenerate_box_stays_a_valid_slice():
    # A zero-area box at the frame edge must not produce x2 < x1: the
    # caller slices with these and would otherwise hand cv2 garbage.
    x1, y1, x2, y2 = expand_box((640.0, 480.0, 640.0, 480.0), 0.2, 640, 480)
    assert x1 <= x2 and y1 <= y2


def test_default_config_carries_context():
    assert DetectionConfig().crop_margin > 0
