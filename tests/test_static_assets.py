"""Self-hosted fonts (CLD-86).

The claim is not "we added some files" — it is that the console renders
correctly on a host with no route to the internet, and reaches no third
party while doing it. That is a property of every page, so it is asserted
by walking the templates rather than by spot-checking one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from siteloom.config import CameraConfig, SiteConfig, StorageConfig
from siteloom.web.app import STATIC_DIR, create_app
from siteloom.store import init_db, make_engine

TEMPLATES = Path(__file__).parent.parent / "siteloom" / "web" / "templates"

#: Anything that would make a browser fetch from another host.
EXTERNAL = re.compile(
    r"""(?:src|href)\s*=\s*["'](?:https?:)?//""", re.I
)


@pytest.fixture
def client(tmp_path):
    config = SiteConfig(
        site_id="t",
        cameras=[CameraConfig(id="c", adapter="file", source="x")],
        storage=StorageConfig(
            db_url=f"sqlite:///{tmp_path}/s.db", media_dir=str(tmp_path / "m")
        ),
    )
    config.identity.enabled = False
    init_db(make_engine(config.storage.db_url))
    return TestClient(create_app(config))


def test_no_template_references_an_external_host():
    """One template reaching a CDN is enough to break an air-gapped
    install, so this walks all of them rather than sampling."""
    offenders = {}
    for path in TEMPLATES.rglob("*.html"):
        hits = EXTERNAL.findall(path.read_text())
        if hits:
            offenders[path.name] = len(hits)
    assert not offenders, f"external asset references: {offenders}"


def test_font_css_is_served_and_points_only_at_local_files(client):
    r = client.get("/static/fonts.css")
    assert r.status_code == 200
    # Comments name the CDN this replaced, and a comment is not a fetch.
    css = re.sub(r"/\*.*?\*/", "", r.text, flags=re.S)
    assert "@font-face" in css
    assert "gstatic.com" not in css
    assert "googleapis.com" not in css
    assert "/static/fonts/" in css


def test_every_font_the_css_asks_for_actually_exists(client):
    """A stylesheet naming a file that was never vendored fails exactly
    like the CDN it replaced, only more quietly."""
    css = client.get("/static/fonts.css").text
    refs = set(re.findall(r"url\((/static/fonts/[^)]+)\)", css))
    assert refs, "no font files referenced"
    for ref in refs:
        assert client.get(ref).status_code == 200, ref


def test_fonts_are_licensed_alongside_the_binaries():
    """OFL requires the license travel with the fonts."""
    licenses = list((STATIC_DIR / "fonts").glob("OFL-*.txt"))
    assert len(licenses) >= 2
    for path in licenses:
        assert "SIL OPEN FONT LICENSE" in path.read_text().upper()


def test_static_mount_does_not_serve_outside_its_directory(client):
    """It ships a fixed package directory, but it is still a mount taking
    a path from the URL."""
    for attempt in (
        "/static/../app.py",
        "/static/%2e%2e/app.py",
        "/static/..%2f..%2fconfig.py",
    ):
        assert client.get(attempt).status_code in (404, 403), attempt


def test_a_rendered_page_links_the_local_stylesheet(client):
    body = client.get("/").text
    assert '/static/fonts.css' in body
    assert "googleapis" not in body
