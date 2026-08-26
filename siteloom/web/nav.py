"""The console's left-hand navigation, as data rather than markup.

Every screen in the console is one entry in one list. Keeping that list
in a template meant each new screen edited the same block of Jinja, which
is both a merge conflict waiting to happen and an easy thing to forget —
a route with no way to reach it is indistinguishable from a route that
does not exist.

So a route module declares its own entry next to the routes it
registers, by calling `add()` from its `register()`. The ordering is
still deliberate (`after=` names the entry to sit behind), because the
sidebar reads top to bottom as a workflow: watch, then look up, then
judge, then operate.

`register()` is called once per app, but tests build many apps in one
process, so `add()` is idempotent on `href` — re-registering replaces
rather than duplicates.

`items()` filters the list to what the viewer may actually open (CLD-103).
The floor is asked of `auth.required_role`, the same function the
middleware enforces with, so a sidebar entry and its 403 can never
disagree — a hardcoded list of restricted entries here would be a second
copy of the rule, wrong from the first time a prefix moves.
"""

from __future__ import annotations

from dataclasses import dataclass

from siteloom.store import User
from siteloom.web import auth


@dataclass(frozen=True)
class NavItem:
    #: Path the entry links to. Also its identity in the registry.
    href: str
    label: str
    #: Two-letter code shown in the collapsed rail and on phones.
    code: str
    #: The entry's screens as in-page sub-tabs, (href, label) pairs with
    #: the entry's own screen among them. One sidebar row can own several
    #: screens (Training owns labelling, the library, classes and
    #: models); base.html renders the strip from this, once, instead of
    #: every member template carrying its own copy. Empty means a
    #: single-screen entry and no strip. Every tab must share the entry's
    #: read floor — `items()` filters by the entry's href alone, so a tab
    #: on a different floor would be advertised to a viewer whose floor
    #: was never checked.
    tabs: tuple[tuple[str, str], ...] = ()

    @staticmethod
    def owns(href: str, path: str) -> bool:
        """Whether `path` is inside the section rooted at `href`.

        "/" would prefix-match everything, so the root section owns
        exactly the event screens instead. Every other section owns its
        own path and anything beneath it — `/library/import` is inside
        `/library`.
        """
        if href == "/":
            return path == "/" or path.startswith("/events")
        return path == href or path.startswith(href + "/")

    def active_for(self, path: str) -> bool:
        """Whether `path` is inside this entry's section or any tab's."""
        return self.owns(self.href, path) or any(
            self.owns(href, path) for href, _ in self.tabs
        )


#: The screens the console shipped with, in sidebar order — grouped into
#: workflow entries, not one row per screen (fourteen rows was a table of
#: contents, not a navigation). Route modules split out of app.py attach
#: their screens via `add()`, either as rows or as tabs of these.
NAV: list[NavItem] = [
    NavItem(
        "/",
        "Events",
        "EV",
        tabs=(("/", "Events"), ("/noise", "Noise")),
    ),
    NavItem("/live", "Live view", "LV"),
    NavItem("/identities", "Identities", "ID"),
    NavItem(
        "/training",
        "Training",
        "TR",
        tabs=(
            ("/training", "Training data"),
            ("/library", "Media library"),
            ("/classes", "Classes"),
        ),
    ),
    NavItem("/stats", "Accuracy", "AC"),
    NavItem("/jobs", "Jobs", "JB"),
]


def add(
    href: str,
    label: str,
    code: str,
    after: str | None = None,
    tab_of: str | None = None,
) -> None:
    """Register a screen in the sidebar, or update one already there.

    `after` names an existing href to sit behind; an unknown one (or
    None) appends, so an entry never vanishes because the screen it
    wanted to follow was removed.

    `tab_of` names an entry to join as an in-page tab instead of taking a
    row — the screen appends to that entry's strip (idempotently, for the
    many apps one test process builds) and any row it held on its own is
    dropped. The tab must share the entry's read floor (see
    `NavItem.tabs`). An unknown parent falls through to a plain row,
    for the same reason `after` does.
    """
    if tab_of is not None:
        for i, entry in enumerate(NAV):
            if entry.href == tab_of:
                tabs = entry.tabs or ((entry.href, entry.label),)
                tabs = tuple(t for t in tabs if t[0] != href) + ((href, label),)
                NAV[i] = NavItem(entry.href, entry.label, entry.code, tabs)
                NAV[:] = [e for e in NAV if e.href != href]
                return
    item = NavItem(href, label, code)
    existing = [i for i, e in enumerate(NAV) if e.href == href]
    if existing:
        NAV[existing[0]] = item
        return
    if after is not None:
        for i, entry in enumerate(NAV):
            if entry.href == after:
                NAV.insert(i + 1, item)
                return
    NAV.append(item)


def items(user: User | None = None) -> list[NavItem]:
    """The sidebar as this viewer may use it.

    Exposed to Jinja as a global so base.html needs no per-route context
    — a screen that forgot to pass it would render a bare page.

    `user` is None in the open single-operator mode (no User rows), where
    nothing is gated and the whole list stands. Otherwise each entry is
    kept only if its href is readable at the viewer's rung, so a
    `restricted` operator is not offered Identities, Media library and
    Models and then refused at the door: an entry that 403s is a console
    describing a system the operator cannot see.

    Filtering, never rewriting: an entry whose floor this viewer clears
    renders exactly as it does for everyone else, so `active_for` and the
    route it points at stay the list's own business.
    """
    if user is None:
        return list(NAV)
    return [i for i in NAV if auth.has_role(user, auth.required_role("GET", i.href))]
