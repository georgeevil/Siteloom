"""Site timezone: store UTC by contract, convert at display (CLD-100).

The properties worth holding, each the reason a rung or a boundary
exists:

* the helper renders UTC verbatim while no zone is configured — the
  entire pre-CLD-100 console is the unset case, which is why no other
  rendered-timestamp assertion in this suite had to move — and converts
  through the site zone once one is set, on both sides of a DST boundary,
* the admin set path validates against the IANA registry — a typo is
  refused with nothing stored in memory or YAML — and the panel names
  the rung that supplied the setting,
* the NVR detect is a one-shot action that fails politely with no NVR
  (stubbed here: no live NVR in tests, ever),
* the browser seed applies only while the setting is unset — whoever
  opens the panel next must not quietly move every rendered timestamp,
* operator-typed `datetime-local` input (backfill range, manual
  bookings) is read as the *site's* wall clock, not the server
  process's, and stored as naive UTC,
* the incident export names its zone — it leaves the console and must
  carry its frame with it,
* and every write is admin by prefix, through the one middleware.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import select

from siteloom import localtime
from siteloom.config import CameraConfig, SiteConfig, StorageConfig, load_config
from siteloom.store import Booking, Camera, Event, get_session, init_db, make_engine
from siteloom.web import timezone_routes
from siteloom.web.app import create_app
from siteloom.web.auth import required_role
from siteloom.web.backfill_routes import parse_range
from siteloom.web.bookings_routes import BookingError, parse_window

#: Europe/Bucharest: UTC+2 in winter (EET), UTC+3 in summer (EEST) — a
#: zone where getting DST wrong is visible on both sides of the year.
ZONE = "Europe/Bucharest"
WINTER_UTC = datetime(2026, 1, 15, 12, 0, 0)
SUMMER_UTC = datetime(2026, 7, 15, 12, 0, 0)


def make_env(tmp_path, timezone_name="", source="", cameras=(), unifi_host=""):
    config = SiteConfig(
        site_id="t",
        site_name="Kai",
        timezone=timezone_name,
        timezone_source=source,
        cameras=list(cameras) or [CameraConfig(id="cam1", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/tz.db", media_dir=str(tmp_path / "m")
        ),
    )
    config.identity.enabled = False
    config.unifi.host = unifi_host
    engine = make_engine(config.storage.db_url)
    init_db(engine)
    Session = get_session(engine)
    with Session() as s:
        s.add(Camera(id="cam1", site_id="t", name="Gate"))
        s.commit()
    client = TestClient(create_app(config), follow_redirects=False)
    return client, Session, config


# -- the helper ------------------------------------------------------------


def test_unset_zone_renders_utc_verbatim():
    """The unset rung is the whole pre-CLD-100 console: rendered output
    must be byte-identical to a bare strftime."""
    assert localtime.display(WINTER_UTC, None) == "2026-01-15 12:00:00"
    assert localtime.display(WINTER_UTC, None, "%H:%M") == "12:00"
    assert localtime.display(None, None) == ""


def test_a_set_zone_converts_on_both_sides_of_dst():
    zone = ZoneInfo(ZONE)
    # Winter: EET, UTC+2.
    assert localtime.display(WINTER_UTC, zone, "%H:%M") == "14:00"
    # Summer: EEST, UTC+3 — a fixed-offset implementation fails here.
    assert localtime.display(SUMMER_UTC, zone, "%H:%M") == "15:00"


def test_input_conversion_inverts_display_on_both_sides_of_dst():
    zone = ZoneInfo(ZONE)
    assert localtime.as_utc(datetime(2026, 1, 15, 14, 0), zone) == WINTER_UTC
    assert localtime.as_utc(datetime(2026, 7, 15, 15, 0), zone) == SUMMER_UTC
    # Unset zone: site time *is* UTC, the conversion is the identity.
    assert localtime.as_utc(WINTER_UTC, None) == WINTER_UTC


def test_the_helper_owns_the_zone_not_the_format():
    """Screens keep their own precision; only the frame moves."""
    zone = ZoneInfo(ZONE)
    assert localtime.display(SUMMER_UTC, zone, "%m-%d %H:%M") == "07-15 15:00"
    assert (
        localtime.display(SUMMER_UTC, zone, "%Y-%m-%dT%H:%M") == "2026-07-15T15:00"
    )


# -- the setting and its validation ---------------------------------------


def test_a_typo_is_refused_at_the_config_boundary():
    with pytest.raises(ValueError):
        SiteConfig(site_id="t", timezone="Europe/Bukarest")
    # Empty is the unset state, not an error.
    assert SiteConfig(site_id="t", timezone="").timezone == ""
    assert SiteConfig(site_id="t", timezone=ZONE).timezone == ZONE


def test_a_typo_in_yaml_fails_the_load(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump({"site_id": "t", "timezone": "Mars/Olympus"}))
    with pytest.raises(Exception):
        load_config(path)


def test_admin_set_stores_validates_and_records_provenance(tmp_path):
    client, _, config = make_env(tmp_path)
    r = client.post("/classes/timezone", data={"timezone": ZONE})
    assert r.status_code == 303 and "tz=set" in r.headers["location"]
    assert (config.timezone, config.timezone_source) == (ZONE, "admin")
    page = client.get("/classes").text
    assert ZONE in page and "set by admin" in page

    # The refusal: nothing stored, nowhere, and the reason names the typo.
    r = client.post("/classes/timezone", data={"timezone": "Europe/Bukarest"})
    assert r.status_code == 400
    assert "Europe/Bukarest" in r.text
    assert (config.timezone, config.timezone_source) == (ZONE, "admin")


def test_the_set_persists_to_yaml_the_way_classes_do(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "site_id": "t",
                "storage": {
                    "db_url": f"sqlite:///{tmp_path}/p.db",
                    "media_dir": str(tmp_path / "m"),
                },
                "identity": {"enabled": False},
            }
        )
    )
    config = load_config(path)
    client = TestClient(create_app(config), follow_redirects=False)
    assert client.post("/classes/timezone", data={"timezone": ZONE}).status_code == 303
    written = yaml.safe_load(path.read_text())
    assert written["timezone"] == ZONE
    assert written["timezone_source"] == "admin"
    # And the round trip survives: the file loads with the zone intact.
    assert load_config(path).timezone == ZONE


def test_clearing_falls_back_to_utc_labelled_as_utc(tmp_path):
    client, _, config = make_env(tmp_path, timezone_name=ZONE, source="admin")
    r = client.post("/classes/timezone", data={"timezone": ""})
    assert r.status_code == 303 and "tz=cleared" in r.headers["location"]
    assert (config.timezone, config.timezone_source) == ("", "")
    page = client.get("/classes").text
    # Rung 4 is never silent: the frame is named even when it is UTC.
    assert "UTC" in page and "default UTC" in page


def test_the_writes_are_admin_by_prefix():
    """The one middleware gates these like /classes/detection — a new
    mutating path must land under ADMIN_PREFIXES, and does."""
    assert required_role("POST", "/classes/timezone") == "admin"
    assert required_role("POST", "/classes/timezone/detect") == "admin"
    assert required_role("POST", "/classes/timezone/seed") == "admin"


# -- rung 2: the NVR ------------------------------------------------------


def test_nvr_detect_reads_once_and_records_provenance(tmp_path, monkeypatch):
    client, _, config = make_env(tmp_path, unifi_host="nvr.local")
    calls = []

    def fake_read(cfg):
        calls.append(cfg)
        return ZONE

    monkeypatch.setattr(timezone_routes, "read_nvr_timezone", fake_read)
    r = client.post("/classes/timezone/detect")
    assert r.status_code == 303 and "tz=detected" in r.headers["location"]
    assert (config.timezone, config.timezone_source) == (ZONE, "nvr")
    # One connect-read-disconnect per click, never a per-request read.
    assert len(calls) == 1
    assert "from NVR" in client.get("/classes").text


def test_nvr_detect_fails_politely_with_no_nvr_configured(tmp_path):
    client, _, config = make_env(tmp_path)  # no unifi host
    r = client.post("/classes/timezone/detect")
    assert r.status_code == 303 and "tz=no-unifi" in r.headers["location"]
    assert config.timezone == ""
    assert "nothing to detect" in client.get("/classes?tz=no-unifi").text


def test_nvr_detect_fails_politely_when_unreachable(tmp_path, monkeypatch):
    client, _, config = make_env(tmp_path, unifi_host="nvr.local")

    def unreachable(cfg):
        raise TimeoutError("no route to host")

    monkeypatch.setattr(timezone_routes, "read_nvr_timezone", unreachable)
    r = client.post("/classes/timezone/detect")
    assert r.status_code == 303 and "tz=nvr-failed" in r.headers["location"]
    assert config.timezone == ""


# -- rung 3: the browser seed ---------------------------------------------


def test_browser_seed_applies_only_while_unset(tmp_path):
    client, _, config = make_env(tmp_path)
    r = client.post("/classes/timezone/seed", data={"timezone": ZONE})
    assert r.status_code == 303 and "tz=seeded" in r.headers["location"]
    assert (config.timezone, config.timezone_source) == (ZONE, "browser")
    assert "from browser" in client.get("/classes").text

    # A second browser proposing a different zone changes nothing: a seed
    # is a starting point, not an override.
    r = client.post("/classes/timezone/seed", data={"timezone": "America/Bogota"})
    assert r.status_code == 303 and "tz=already-set" in r.headers["location"]
    assert (config.timezone, config.timezone_source) == (ZONE, "browser")


def test_browser_seed_is_validated_like_any_other_write(tmp_path):
    client, _, config = make_env(tmp_path)
    assert (
        client.post(
            "/classes/timezone/seed", data={"timezone": "Not/AZone"}
        ).status_code
        == 400
    )
    assert client.post("/classes/timezone/seed", data={"timezone": ""}).status_code == 400
    assert config.timezone == ""


# -- display across the console -------------------------------------------


def test_screens_convert_once_a_zone_is_set(tmp_path):
    client, Session, config = make_env(tmp_path, timezone_name=ZONE, source="admin")
    with Session() as s:
        s.add(
            Booking(
                uid="b1", summary="Unit A", start=WINTER_UTC, end=SUMMER_UTC,
                source="ical",
            )
        )
        s.commit()
    page = client.get("/bookings").text
    # Stored 12:00 UTC renders as the site's wall clock, and the page
    # names the frame instead of the old hardcoded "UTC".
    assert "2026-01-15 14:00" in page
    assert ZONE in page
    assert "Window (UTC)" not in page


def test_unset_zone_renders_screens_exactly_as_before(tmp_path):
    client, Session, config = make_env(tmp_path)
    with Session() as s:
        s.add(
            Booking(
                uid="b1", summary="Unit A", start=WINTER_UTC, end=SUMMER_UTC,
                source="ical",
            )
        )
        s.commit()
    page = client.get("/bookings").text
    assert "2026-01-15 12:00" in page
    assert "Window (UTC)" in page


def test_datetime_local_prefill_matches_the_input_frame(tmp_path):
    """The edit form's value and what parse_window reads back must be one
    frame: saving an untouched form must not shift a booking."""
    client, Session, config = make_env(tmp_path, timezone_name=ZONE, source="admin")
    with Session() as s:
        s.add(
            Booking(
                uid=f"siteloom-manual:{0}", summary="Unit A",
                start=SUMMER_UTC, end=SUMMER_UTC, source="manual",
            )
        )
        s.commit()
    page = client.get("/bookings").text
    assert 'value="2026-07-15T15:00"' in page
    first, last = parse_window("2026-07-15T15:00", "2026-07-15T15:00", ZoneInfo(ZONE))
    assert (first, last) == (SUMMER_UTC, SUMMER_UTC)


# -- input boundaries ------------------------------------------------------


def test_backfill_range_is_read_in_the_site_zone():
    zone = ZoneInfo(ZONE)
    begins, finishes = parse_range("2026-07-15T15:00", "2026-07-15T16:00", zone)
    assert begins.utcoffset().total_seconds() == 3 * 3600
    assert begins.astimezone(timezone.utc).replace(tzinfo=None) == SUMMER_UTC
    # Unset zone: naive input is UTC — never the server process's zone.
    begins, _ = parse_range("2026-07-15T15:00", "2026-07-15T16:00")
    assert begins.tzinfo is not None
    assert begins.astimezone(timezone.utc).replace(tzinfo=None) == datetime(
        2026, 7, 15, 15, 0
    )


def test_booking_input_is_site_local_stored_as_utc(tmp_path):
    client, Session, config = make_env(tmp_path, timezone_name=ZONE, source="admin")
    r = client.post(
        "/bookings",
        data={"summary": "Unit A", "start": "2026-07-15T15:00", "end": "2026-07-15T18:00"},
    )
    assert r.status_code == 303
    with Session() as s:
        booking = s.scalars(select(Booking)).one()
        assert booking.start == SUMMER_UTC  # 15:00 EEST == 12:00 UTC
        assert booking.end == datetime(2026, 7, 15, 15, 0)


def test_booking_offsets_are_still_refused():
    with pytest.raises(BookingError):
        parse_window("2026-08-10T15:00+02:00", "2026-08-11T15:00", ZoneInfo(ZONE))


# -- the export carries its frame -----------------------------------------


def test_incident_export_names_its_zone_and_converts(tmp_path):
    client, Session, config = make_env(tmp_path, timezone_name=ZONE, source="admin")
    with Session() as s:
        s.add(
            Event(
                camera_id="cam1", track_id=1, class_name="car",
                first_seen=WINTER_UTC, last_seen=WINTER_UTC,
                detection_count=4, best_confidence=0.8,
            )
        )
        s.commit()
    r = client.post(
        "/events/1/escalate", data={"incident_id": "new", "title": "", "next_url": ""}
    )
    assert r.status_code == 303
    body = client.get("/incidents/1/export").text
    assert ZONE in body  # the frame travels with the file
    assert f"Time ({ZONE})" in body
    assert "2026-01-15 14:00:00" in body  # the event, on the site clock
    # The default title bakes prose about a moment — the site's moment.
    assert "2026-01-15 14:00" in body.split("<title>")[1].split("</title>")[0]


def test_incident_export_says_utc_when_unset(tmp_path):
    client, Session, config = make_env(tmp_path)
    with Session() as s:
        s.add(
            Event(
                camera_id="cam1", track_id=1, class_name="car",
                first_seen=WINTER_UTC, last_seen=WINTER_UTC,
                detection_count=4, best_confidence=0.8,
            )
        )
        s.commit()
    client.post(
        "/events/1/escalate", data={"incident_id": "new", "title": "T", "next_url": ""}
    )
    body = client.get("/incidents/1/export").text
    # Rung 4, labelled: UTC is named, never implied.
    assert "Time (UTC)" in body
    assert "2026-01-15 12:00:00" in body
