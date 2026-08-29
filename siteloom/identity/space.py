"""The embedding-space stamp (CLD-106): which settings the vectors are in.

Some settings are not settings. `crop_margin` changes what every crop
looks like, and the face projection changes what every face vector *is*
— after either moves, stored vectors and new ones are incomparable,
matching quietly degrades, and nothing errors. CLD-106's decision:
record the space the vectors were built in, notice when the config
disagrees, and offer the reset-and-rebuild (labels survive, vectors are
derived and rebuilt from the stored crops).

This module is the ONE place that says which settings are poisoning —
a future field gets classified here deliberately, not by whoever
notices (`POISONING`). The stamp lives as `embedding-space.json`
*inside* the vector store directory: it describes exactly what sits
next to it, `siteloom reset` wipes the two together, and a store copied
to another machine carries its own provenance. (Named "stamp", not
"fingerprint" — `identity/fingerprint.py` is the CLD-254 vehicle-color
read.)

Dimensions come from the embedder classes' static knowledge (class
attributes plus the projection file's shape) — computing a stamp never
loads a model, so `doctor` can afford the check.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

STAMP_FILE = "embedding-space.json"

#: The settings that define the vector space. Adding a field here is a
#: deliberate act: it means "changing this invalidates every stored
#: vector", and the doctor check + rebuild flow pick it up unchanged.
POISONING = (
    "detection.crop_margin",
    "identity.face_projection_path (and the file's content)",
    "the embedder algorithms and their output dimensions",
)


def _projection_state(path_text: str | None) -> dict:
    """The face projection's identity: path, content hash, and the face
    dimension it implies. A configured-but-missing file is the silent
    base-SFace fallback (CLD-299) — recorded as absent, so a projection
    appearing later reads as the space change it is."""
    state: dict = {"path": path_text or None, "sha256": None}
    face_dim = 128  # SFace base
    if path_text:
        path = Path(path_text)
        if path.is_file():
            state["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            try:
                import numpy as np

                face_dim = int(np.load(path, mmap_mode="r").shape[1])
            except Exception:  # pragma: no cover — corrupt file
                log.warning("unreadable face projection %s", path)
    return {**state, "face_dim": face_dim}


def compute_stamp(config) -> dict:
    """The space the current configuration would embed into."""
    from siteloom.identity.embedders import GenericEmbedder

    projection = _projection_state(config.identity.face_projection_path)
    return {
        "crop_margin": config.detection.crop_margin,
        "face_projection": {
            "path": projection["path"], "sha256": projection["sha256"],
        },
        "embedders": {
            "face": projection["face_dim"],
            "generic": GenericEmbedder.dim,
        },
    }


def _stamp_path(vector_db_path: str | Path) -> Path:
    return Path(vector_db_path) / STAMP_FILE


def read_stamp(vector_db_path: str | Path) -> dict | None:
    path = _stamp_path(vector_db_path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        log.warning("corrupt embedding-space stamp at %s", path)
        return None


def write_stamp(vector_db_path: str | Path, stamp: dict) -> None:
    path = _stamp_path(vector_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamp, indent=2))


def stamp_diff(recorded: dict, current: dict) -> list[str]:
    """The poisoning fields that moved, as human-readable lines. Empty
    means the stored vectors match what the config would produce."""
    diffs: list[str] = []
    if recorded.get("crop_margin") != current.get("crop_margin"):
        diffs.append(
            f"crop_margin: {recorded.get('crop_margin')} → "
            f"{current.get('crop_margin')}"
        )
    rec_proj = recorded.get("face_projection") or {}
    cur_proj = current.get("face_projection") or {}
    if rec_proj.get("sha256") != cur_proj.get("sha256"):
        def word(p):
            return "none" if not p.get("sha256") else (
                f"{p.get('path')} ({p['sha256'][:12]}…)"
            )
        diffs.append(
            f"face projection: {word(rec_proj)} → {word(cur_proj)}"
        )
    rec_dims = recorded.get("embedders") or {}
    cur_dims = current.get("embedders") or {}
    for algo in sorted(set(rec_dims) | set(cur_dims)):
        if rec_dims.get(algo) != cur_dims.get(algo):
            diffs.append(
                f"{algo} embedding dimension: {rec_dims.get(algo)} → "
                f"{cur_dims.get(algo)}"
            )
    return diffs
