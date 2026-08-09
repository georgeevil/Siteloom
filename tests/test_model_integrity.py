"""Face weights are pinned by digest and verified on every load (CLD-50).

The YuNet/SFace ONNX files decide who the system thinks a face belongs
to, and they land in a cache directory under the operator's home that
any local process can write. The old size heuristic only caught a
truncated download; a same-size substitution walked straight into
`cv2.FaceRecognizerSF`.

Everything here runs offline: the "upstream" is a `file://` URL over a
temp file, which exercises the real download path (urlopen → staging
file → verify → rename) without touching the network or needing weights.
"""

from __future__ import annotations

import hashlib

import pytest

from siteloom import health
from siteloom.health import FAIL, OK, WARN, Report
from siteloom.identity import embedders
from siteloom.identity.embedders import ModelIntegrityError, ModelSpec, ensure_model

GOOD = b"pretend-onnx-weights" * 64


def spec_for(payload: bytes, path) -> ModelSpec:
    return ModelSpec(
        url=path.as_uri(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )


@pytest.fixture
def upstream(tmp_path):
    """A published artifact and the spec that pins it."""
    remote = tmp_path / "upstream"
    remote.mkdir()
    path = remote / "face_model.onnx"
    path.write_bytes(GOOD)
    return spec_for(GOOD, path)


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "cache"


def no_network(monkeypatch, reason="the network must not be touched here"):
    def explode(url, dest, timeout=None):
        raise AssertionError(reason)

    monkeypatch.setattr(embedders, "_fetch", explode)


# -- downloading ------------------------------------------------------------


def test_a_matching_digest_is_downloaded_and_cached(upstream, cache):
    path = ensure_model(upstream, cache)
    assert path == cache / "face_model.onnx"
    assert path.read_bytes() == GOOD


def test_a_wrong_digest_leaves_nothing_behind(upstream, cache):
    """A bad file left in the cache is loaded by the next run without
    even attempting a re-download — so the cache must stay empty."""
    tampered = ModelSpec(url=upstream.url, sha256="0" * 64, size=upstream.size)

    with pytest.raises(ModelIntegrityError) as exc:
        ensure_model(tampered, cache)

    assert list(cache.iterdir()) == []  # no weights, no .part
    assert "sha256" in str(exc.value)


def test_the_failure_says_which_kind_of_failure_it_is(upstream, cache):
    """An operator has to tell a corrupted download from an upstream
    that republished the file, and must not be tempted to repin blind."""
    tampered = ModelSpec(url=upstream.url, sha256="0" * 64, size=upstream.size)
    with pytest.raises(ModelIntegrityError) as exc:
        ensure_model(tampered, cache)
    message = str(exc.value)
    assert "re-download" in message
    assert "Do not edit the pinned digest" in message
    assert str(cache) in message  # where to pre-seed


def test_a_truncated_download_is_caught_by_size_before_hashing(tmp_path, cache):
    """git-lfs pointers were the original failure mode; they are a
    size mismatch, and must still be reported as one."""
    remote = tmp_path / "remote"
    remote.mkdir()
    pointer = remote / "face_model.onnx"
    pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    spec = ModelSpec(url=pointer.as_uri(), sha256="a" * 64, size=len(GOOD))

    with pytest.raises(ModelIntegrityError) as exc:
        ensure_model(spec, cache)
    assert "size" in str(exc.value)
    assert list(cache.iterdir()) == []


def test_an_unreachable_url_explains_pre_seeding(upstream, cache, monkeypatch):
    def offline(url, dest, timeout=None):
        raise OSError("Network is unreachable")

    monkeypatch.setattr(embedders, "_fetch", offline)
    with pytest.raises(IOError) as exc:
        ensure_model(upstream, cache)
    assert "pre-seed" in str(exc.value)
    assert "SITELOOM_MODELS_DIR" in str(exc.value)
    assert list(cache.glob("*.part")) == []  # no half-written staging file


# -- the cached file, on every subsequent run --------------------------------


def test_a_verified_cache_needs_no_network(upstream, cache, monkeypatch):
    """Air-gapped installs and the test suite depend on this: a
    pre-seeded, correct file is used without any download attempt."""
    cache.mkdir()
    (cache / "face_model.onnx").write_bytes(GOOD)
    no_network(monkeypatch)

    assert ensure_model(upstream, cache).read_bytes() == GOOD


def test_a_second_call_does_not_re_download(upstream, cache, monkeypatch):
    ensure_model(upstream, cache)
    no_network(monkeypatch)
    assert ensure_model(upstream, cache).read_bytes() == GOOD


def test_a_tampered_cache_entry_of_the_right_size_is_caught(upstream, cache):
    """The whole point of the digest: same size, different bytes. The
    old min_bytes check passed this file happily."""
    cache.mkdir()
    cached = cache / "face_model.onnx"
    swapped = b"x" * len(GOOD)
    cached.write_bytes(swapped)
    assert len(swapped) == upstream.size

    path = ensure_model(upstream, cache)
    assert path.read_bytes() == GOOD  # replaced from upstream


def test_a_tampered_cache_entry_is_deleted_even_if_redownload_fails(
    upstream, cache, monkeypatch
):
    """Removal must not be conditional on a successful replacement, or a
    host that is offline keeps loading the bad file forever."""
    cache.mkdir()
    cached = cache / "face_model.onnx"
    cached.write_bytes(b"x" * len(GOOD))

    def offline(url, dest, timeout=None):
        raise OSError("Network is unreachable")

    monkeypatch.setattr(embedders, "_fetch", offline)
    with pytest.raises(IOError):
        ensure_model(upstream, cache)
    assert not cached.exists()


# -- where the models live ---------------------------------------------------


def test_models_dir_honours_the_offline_override(tmp_path, monkeypatch):
    monkeypatch.delenv(embedders.MODELS_DIR_ENV, raising=False)
    assert embedders.models_dir() == embedders.DEFAULT_MODELS_DIR
    monkeypatch.setenv(embedders.MODELS_DIR_ENV, str(tmp_path / "seeded"))
    assert embedders.models_dir() == tmp_path / "seeded"


def test_the_pinned_specs_are_plausible():
    """A wrong pin breaks every fresh install, so at least assert the
    shape: real sha256 hex, a real size, an https source."""
    for spec in embedders.FACE_MODELS:
        assert len(spec.sha256) == 64
        assert set(spec.sha256) <= set("0123456789abcdef")
        assert spec.size > 100_000
        assert spec.url.startswith("https://")
        assert spec.filename.endswith(".onnx")


# -- the health check --------------------------------------------------------


def use_test_models(monkeypatch, cache, spec):
    monkeypatch.setattr(embedders, "FACE_MODELS", (spec,))
    monkeypatch.setenv(embedders.MODELS_DIR_ENV, str(cache))


def status_of(report: Report) -> tuple[str, str, str]:
    check = report.checks[0]
    return check.status, check.detail, check.remedy


def test_doctor_reports_a_corrupt_model_as_a_failed_check(
    upstream, cache, monkeypatch
):
    """A check that raised would hide every other check in the report."""
    cache.mkdir()
    (cache / "face_model.onnx").write_bytes(b"x" * len(GOOD))
    use_test_models(monkeypatch, cache, upstream)

    report = Report()
    health.check_face_models(report, None)  # must not raise

    status, detail, remedy = status_of(report)
    assert status == FAIL
    assert "face_model.onnx" in detail
    assert str(cache / "face_model.onnx") in remedy


def test_doctor_does_not_delete_what_it_diagnoses(upstream, cache, monkeypatch):
    """`doctor` is safe to run at any time, including as an
    ExecStartPre — it reports, ensure_model is what cleans up."""
    cache.mkdir()
    bad = cache / "face_model.onnx"
    bad.write_bytes(b"x" * len(GOOD))
    use_test_models(monkeypatch, cache, upstream)

    health.check_face_models(Report(), None)
    assert bad.exists()


def test_doctor_warns_when_models_are_merely_absent(upstream, cache, monkeypatch):
    use_test_models(monkeypatch, cache, upstream)
    report = Report()
    health.check_face_models(report, None)
    status, _, remedy = status_of(report)
    assert status == WARN
    assert "pre-seed" in remedy


def test_doctor_passes_on_verified_models(upstream, cache, monkeypatch):
    cache.mkdir()
    (cache / "face_model.onnx").write_bytes(GOOD)
    use_test_models(monkeypatch, cache, upstream)
    report = Report()
    health.check_face_models(report, None)
    status, detail, _ = status_of(report)
    assert status == OK
    assert "verified" in detail
