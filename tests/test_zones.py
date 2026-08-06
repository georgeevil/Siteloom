from siteloom.modules.detection import _zones_hit

# Zone covering the left half of a 100x100 frame.
LEFT = {"name": "left", "points": [[0, 0], [0.5, 0], [0.5, 1.0], [0, 1.0]]}


def test_bottom_center_inside_zone():
    # bbox bottom-center at (25, 90) → inside left half
    assert _zones_hit([LEFT], (10, 40, 40, 90), 100, 100) == ["left"]


def test_bottom_center_outside_zone():
    # bbox overlaps the zone but its bottom-center (75, 90) is right half
    assert _zones_hit([LEFT], (40, 40, 110, 90), 100, 100) == []


def test_no_zones_configured():
    assert _zones_hit([], (0, 0, 10, 10), 100, 100) == []


def test_multiple_zone_hits():
    top = {"name": "top", "points": [[0, 0], [1.0, 0], [1.0, 0.95], [0, 0.95]]}
    hits = _zones_hit([LEFT, top], (10, 40, 40, 90), 100, 100)
    assert hits == ["left", "top"]
