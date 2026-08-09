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
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NavItem:
    #: Path the entry links to. Also its identity in the registry.
    href: str
    label: str
    #: Two-letter code shown in the collapsed rail and on phones.
    code: str

    def active_for(self, path: str) -> bool:
        """Whether `path` is inside this entry's section.

        "/" would prefix-match everything, so the root entry owns exactly
        the event screens instead. Every other entry owns its own path
        and anything beneath it — `/library/import` lights up Library.
        """
        if self.href == "/":
            return path == "/" or path.startswith("/events")
        return path == self.href or path.startswith(self.href + "/")


#: The screens the console shipped with, in sidebar order. Route modules
#: split out of app.py append to this via `add()`.
NAV: list[NavItem] = [
    NavItem("/", "Events", "EV"),
    NavItem("/live", "Live view", "LV"),
    NavItem("/library", "Media library", "MD"),
    NavItem("/identities", "Identities", "ID"),
    NavItem("/classes", "Classes", "CL"),
    NavItem("/training", "Training data", "TR"),
    NavItem("/stats", "Accuracy", "AC"),
    NavItem("/jobs", "Jobs", "JB"),
    NavItem("/noise", "Noise", "NO"),
]


def add(href: str, label: str, code: str, after: str | None = None) -> None:
    """Register a screen in the sidebar, or update one already there.

    `after` names an existing href to sit behind; an unknown one (or
    None) appends, so an entry never vanishes because the screen it
    wanted to follow was removed.
    """
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


def items() -> list[NavItem]:
    """Exposed to Jinja as a global so base.html needs no per-route
    context — a screen that forgot to pass it would render a bare page."""
    return list(NAV)
