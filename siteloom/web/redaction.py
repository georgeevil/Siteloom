"""What a viewer may be told about who someone is (CLD-111).

`restricted` exists so a night-shift operator can judge whether an event
matters without browsing everyone's face (NFR5). CLD-31 closed the
screens that *list* people; CLD-103 closed the picker that listed them
inside a screen the rung deliberately keeps. What neither closed is every
place the console prints one name at a time — the triage queue's identity
column, the rail's linked identities, an incident title an operator
typed, a booking summary an iCal feed carried. Fifty rows of the first
is the directory `/identities` was closed to withhold, reassembled by
scrolling.

The product decision (2026-08-10): **a restricted operator sees whether
the system recognised someone, never who.** Recognised-vs-not is the
entire triage signal — "should I wake someone" — it is a boolean, and it
survives removing the name, so the rung stays staffable while the console
stops handing out a directory.

One decision (`may_name`) with three substitutions — and the predicate
itself, for markup that only means anything alongside a name — rather
than a judgement remade on every screen. A per-screen judgement is what
produced the surfaces this closes; seven of them landed in one session:

* An identity's name becomes `Known person` / `Known vehicle` /
  `Known <class>`. Unmatched is untouched: it never carried a name, and
  telling the two apart is the one thing that must survive.
* An identity's **plate** goes the same way. "Known vehicle · ABC123"
  identifies a household as surely as a name does.
* Free text — an incident title an operator authored, a booking summary
  a feed supplied — is withheld **wholesale** and replaced by its row's
  number (`Incident 12 · 3 events`). Free text cannot be sanitised, only
  replaced: a Writer can type anything into an incident title, and
  Airbnb/Booking.com iCal summaries routinely carry the guest's name,
  which makes /bookings a dated guest register. What stays is what tells
  a restricted operator an arrival was expected — the window, the source
  and the event counts.

**Event crops stay visible**, explicitly decided. A face on the event
being triaged is what a guard watching a monitor sees anyway, the gallery
under `media_dir/library/` is gated already (`auth.is_gallery_media`),
and withholding the crop too would leave nothing to judge from.

Both halves are server-side. The Jinja globals `install()` binds resolve
the viewer from the request themselves, so a template calls
`identity_name(x)` and never reaches for `x.display_name`; a `{% if %}`
in the template would render nothing and still ship every name in the
payload, which is the lesson CLD-103 already paid for.

The floor is asked of `auth.required_role("GET", "/identities")` rather
than written here as a second literal rung, for the same reason
`nav.items` asks it: the screen that lists people *is* the definition of
"may be told a name", so moving that floor moves both or neither.
"""

from __future__ import annotations

from typing import Any

from jinja2 import pass_context

from siteloom.store import Identity, User
from siteloom.web import auth

#: The screen whose read floor decides this. Not a rung spelled out here.
NAMING_FLOOR_PATH = "/identities"

#: Which noun a recognised subject is announced with. Keyed on the
#: identifier first because that is what an Identity row is *for*: a face
#: identity and an appearance identity are both a person, while a vehicle
#: identity's detection class is car or truck or bus.
IDENTIFIER_NOUNS = {"face": "person", "person": "person", "vehicle": "vehicle"}
#: The classes that are a vehicle under another name, for an identity
#: whose identifier is not one of the three above. Mirrors
#: CLASS_KINDS["vehicles"] in web/app.py, which cannot be imported here
#: (app.py imports this module).
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
#: Neither an identifier nor a class this build knows. Never blank: an
#: empty identity cell reads as "unmatched", which is the one distinction
#: this module exists to preserve.
FALLBACK_NOUN = "subject"


def may_name(user: User | None) -> bool:
    """Whether this viewer is told *who*, or only *that the system knew*.

    `None` is the open single-operator mode (no User rows at all), where
    nothing is gated and the console behaves exactly as it did before
    auth existed — a feature that needs an account to work is a bug.
    """
    return user is None or auth.has_role(
        user, auth.required_role("GET", NAMING_FLOOR_PATH)
    )


def subject_noun(identity: Identity) -> str:
    """`person`, `vehicle`, or whatever class this identity is.

    Falling through to the detection class is what makes a dynamically
    added class read as `Known dog` with no code change here — adding a
    class to `detection.classes` is meant to be the only step (NFR3).
    """
    key = (identity.identifier_key or "").strip().lower()
    if key in IDENTIFIER_NOUNS:
        return IDENTIFIER_NOUNS[key]
    class_name = (identity.class_name or "").strip().lower()
    if class_name in VEHICLE_CLASSES:
        return "vehicle"
    return class_name or key or FALLBACK_NOUN


def recognised_as(identity: Identity) -> str:
    """All a withheld identity says: the system has seen this one before.

    True of an unlabeled row as much as a named one — an identity exists
    because sightings were grouped, which is the recognition the operator
    is being told about. Whether it also has a name is not their rung.
    """
    return f"Known {subject_noun(identity)}"


def name_for(identity: Identity | None, allowed: bool) -> str:
    """The identity name this viewer gets. `allowed` is `may_name`'s."""
    if identity is None:
        return ""
    return identity.display_name if allowed else recognised_as(identity)


def plate_for(identity: Identity | None, allowed: bool) -> str | None:
    """The plate this viewer gets — None below the floor, so the template
    prints no separator either, rather than a bare dot with a gap."""
    if identity is None or not allowed:
        return None
    return identity.plate


def text_for(value: str | None, withheld: str, allowed: bool) -> str | None:
    """Free text, or the stand-in that names its row instead.

    An empty value is passed through: there is nothing to withhold, and
    substituting there would invent a row description where the screen
    means to show its own "—".
    """
    if allowed or not value:
        return value
    return withheld


def _may_name_in(context: Any) -> bool:
    """The gate for one template render.

    A context with no `request` is a template rendered outside the
    console's request path. Withholding there is the safe direction: a
    rendering mistake must not be able to turn into a disclosure, and
    guessing "open mode" would make it one.
    """
    request = context.get("request")
    if request is None:
        return False
    return may_name(getattr(request.state, "user", None))


def install(templates) -> None:
    """Bind the substitutions into Jinja as viewer-aware globals.

    Globals rather than per-screen context, for the same reason
    `nav_items` is one: these are called from templates rendered by four
    different modules, and a screen that has to be handed the helper is a
    screen that renders a name the day someone forgets.
    """

    @pass_context
    def identity_name(context: Any, identity: Identity | None) -> str:
        return name_for(identity, _may_name_in(context))

    @pass_context
    def identity_plate(context: Any, identity: Identity | None) -> str | None:
        return plate_for(identity, _may_name_in(context))

    @pass_context
    def free_text(context: Any, value: str | None, withheld: str) -> str | None:
        return text_for(value, withheld, _may_name_in(context))

    @pass_context
    def may_name(context: Any) -> bool:
        """The gate itself, for markup that only makes sense with a name.

        The rail styles an unnamed claim differently from a named one.
        Below the floor both read `Known vehicle`, so keeping that
        styling would say which of the two the system can name — and
        dropping it unconditionally would grey out every claim for a
        viewer who can read the names perfectly well. The distinction is
        the viewer's, so the condition asks about the viewer.
        """
        return _may_name_in(context)

    templates.env.globals["identity_name"] = identity_name
    templates.env.globals["identity_plate"] = identity_plate
    templates.env.globals["free_text"] = free_text
    templates.env.globals["may_name"] = may_name
