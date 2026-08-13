"""Site-local time: the display side of the naive-UTC contract (CLD-100).

The contract: **naive UTC everywhere in the store; convert at the display
boundary via the site zone; convert at input boundaries on the way in.**
Storage never changes shape — every timestamp column stays a naive UTC
datetime (`store/models._utcnow`) — and this module is the one place the
site's zone is applied, in both directions:

* `display()` turns a stored naive-UTC value into the site's wall clock
  for a screen or an export. The console renders **site-local for every
  viewer** — two operators discussing "the 21:40 event" mean the same
  moment — never the viewer's browser zone, which is used once, to *seed*
  the setting (see web/timezone_routes.py), and never again.
* `as_utc()` / `as_aware()` read a naive value an operator typed into a
  `datetime-local` input as the site's wall clock and convert it to what
  the consumer needs. Interpreting form input in the *server process's*
  zone (`.astimezone()` with no argument) is the bug this replaces: right
  only until the box serving the site sits in a different zone than the
  cameras.

The zone itself is `SiteConfig.timezone` (an IANA name), resolved by the
chain in web/timezone_routes.py: admin-set, else read from the UniFi NVR,
else seeded once from an admin's browser, else UTC — labelled as UTC,
never silently. Unset means every function here degrades to a straight
UTC read, which is exactly the behaviour the console had before the
setting existed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo, available_timezones

#: How each rung of the resolution chain announces itself on the admin
#: panel. The stored `timezone_source` values are the keys; "" is both
#: "never set" and, paired with an empty `timezone`, the default-UTC rung.
SOURCE_LABELS = {
    "admin": "set by admin",
    "nvr": "from NVR",
    "browser": "from browser",
    "": "default UTC",
}

#: The default format the Jinja global falls back to. Screens that want
#: another precision pass their own — the helper owns the *zone*, not a
#: single format string.
DEFAULT_FORMAT = "%Y-%m-%d %H:%M:%S"


def zone_options() -> list[str]:
    """Every IANA name this host can store, for the admin panel's
    datalist — picking beats typing, and typos get refused anyway."""
    return sorted(available_timezones())


def validate_timezone(name: str) -> str:
    """`name` as a stored-worthy IANA zone name, or ValueError.

    Checked against `zoneinfo.available_timezones()` so a typo is refused
    at the boundary rather than stored and blowing up at render time.
    Empty is valid: it is the unset state, which means UTC.
    """
    name = name.strip()
    if name and name not in available_timezones():
        raise ValueError(
            f"{name!r} is not an IANA timezone name (e.g. Europe/Bucharest)"
        )
    return name


def site_zone(config) -> ZoneInfo | None:
    """The configured zone, or None for the default-UTC rung.

    Resolved per call, never cached: the admin panel edits the live
    config object, and a zone captured at app construction would render
    the old wall clock until restart.
    """
    name = getattr(config, "timezone", "")
    return ZoneInfo(name) if name else None


def zone_name(config) -> str:
    """What to call the frame times are shown in — 'UTC' when unset.

    The unset rung is *labelled*, never silent: a bare timestamp with no
    frame is exactly the ambiguity the site zone exists to remove.
    """
    return getattr(config, "timezone", "") or "UTC"


def source_label(config) -> str:
    """Which rung of the resolution chain supplied the current setting."""
    if not getattr(config, "timezone", ""):
        return SOURCE_LABELS[""]
    source = getattr(config, "timezone_source", "")
    return SOURCE_LABELS.get(source, SOURCE_LABELS["admin"])


def display(value: datetime | None, zone: ZoneInfo | None, fmt: str = DEFAULT_FORMAT) -> str:
    """A stored naive-UTC value, formatted on the site's wall clock.

    With no zone configured this is `strftime` verbatim — the rendered
    output of an unconfigured site is byte-identical to what the console
    produced before the setting existed, which is what keeps every
    pre-existing rendered-timestamp assertion true.
    """
    if value is None:
        return ""
    if zone is None:
        return value.strftime(fmt)
    return value.replace(tzinfo=timezone.utc).astimezone(zone).strftime(fmt)


def as_utc(value: datetime, zone: ZoneInfo | None) -> datetime:
    """Operator-typed naive site-local time -> naive UTC for the store.

    The input boundary for anything that persists what a `datetime-local`
    input posted (manual bookings). An aware value is converted from its
    own offset; callers that refuse offsets do so before calling.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=zone or timezone.utc)
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def as_aware(value: datetime, zone: ZoneInfo | None) -> datetime:
    """Operator-typed naive site-local time -> aware, zone attached.

    For consumers that want an aware datetime (the NVR backfill API).
    A value that already carries an offset is honoured as it stands.
    """
    if value.tzinfo is not None:
        return value
    return value.replace(tzinfo=zone or timezone.utc)
