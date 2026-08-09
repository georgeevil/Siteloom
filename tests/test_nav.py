"""The sidebar is the console's only index of itself (CLD-87).

Two failures matter and neither shows up in a page-renders test: an entry
pointing at a route nobody registered, and a route that exists with no
entry pointing at it. The first is a dead link; the second is a feature
the operator will never find. Both are asserted against the live app
rather than against the list, because the list is exactly the thing that
could be wrong.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.store import init_db, make_engine
from siteloom.web import nav
from siteloom.web.app import create_app


@pytest.fixture
def app(tmp_path):
    config = SiteConfig(
        site_id="t",
        cameras=[CameraConfig(id="c", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/n.db", media_dir=str(tmp_path / "m")
        ),
    )
    config.identity.enabled = False
    init_db(make_engine(config.storage.db_url))
    return create_app(config)


def test_every_sidebar_entry_resolves_to_a_registered_route(app):
    routes = {r.path for r in app.routes}
    missing = [i.href for i in nav.items() if i.href not in routes]
    assert not missing, f"sidebar links with no route: {missing}"


def test_every_sidebar_entry_actually_renders(app):
    client = TestClient(app)
    for item in nav.items():
        r = client.get(item.href)
        assert r.status_code == 200, f"{item.href} -> {r.status_code}"


def test_the_current_screen_is_marked_active(app):
    client = TestClient(app)
    for path, expected in (("/", "Events"), ("/identities", "Identities")):
        body = client.get(path).text
        # The active class is what tells an operator where they are; a
        # sidebar where nothing is lit is a sidebar that lost the plot.
        assert body.count("sl-nav-item active") == 1
        assert expected in body


def test_a_subpath_lights_up_its_section(app):
    """`/library/import` is inside Library, and `/` must not claim it."""
    assert nav.NavItem("/library", "x", "x").active_for("/library/import")
    assert not nav.NavItem("/", "x", "x").active_for("/library/import")
    assert nav.NavItem("/", "x", "x").active_for("/events/3")


def test_a_sibling_prefix_is_not_a_subpath():
    """`/live` must not light up for `/livestream` — string-prefix
    matching is the same bug shape as the media-root containment one."""
    assert not nav.NavItem("/live", "x", "x").active_for("/livestream")


def test_add_is_idempotent_so_a_second_app_does_not_duplicate_entries():
    """register() runs once per app and tests build many per process."""
    before = len(nav.items())
    nav.add("/probe", "Probe", "PB")
    nav.add("/probe", "Probe", "PB")
    try:
        assert len(nav.items()) == before + 1
    finally:
        nav.NAV[:] = [i for i in nav.NAV if i.href != "/probe"]


def test_add_places_an_entry_where_it_was_asked_to_go():
    nav.add("/probe", "Probe", "PB", after="/live")
    try:
        hrefs = [i.href for i in nav.items()]
        assert hrefs[hrefs.index("/live") + 1] == "/probe"
    finally:
        nav.NAV[:] = [i for i in nav.NAV if i.href != "/probe"]


def test_an_unknown_anchor_appends_rather_than_dropping_the_entry():
    nav.add("/probe", "Probe", "PB", after="/nonexistent")
    try:
        assert nav.items()[-1].href == "/probe"
    finally:
        nav.NAV[:] = [i for i in nav.NAV if i.href != "/probe"]
