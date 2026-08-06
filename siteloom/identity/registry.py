"""Identifier registry: which identification algorithms run for which
detection class — including classes added dynamically at runtime.

Configured identifiers come from `identity.identifiers` in site YAML.
When a detection class arrives with no identifier and `auto_add_classes`
is on, the registry manufactures a generic identifier for it on the spot:
its own registry key, its own vector-store collection (created lazily by
the store), the default threshold. Adding "deer" to detection.classes is
therefore the ONLY step needed to start re-identifying deer.
"""

from __future__ import annotations

import logging

from siteloom.config import IdentifierConfig, IdentityConfig

log = logging.getLogger(__name__)


class IdentifierRegistry:
    def __init__(self, cfg: IdentityConfig, device: str = "mps"):
        self.cfg = cfg
        self._device = device
        self._identifiers: dict[str, IdentifierConfig] = dict(cfg.identifiers)
        self._embedders: dict[str, object] = {}  # algo -> shared instance

    def identifiers_for(self, class_name: str) -> list[tuple[str, IdentifierConfig]]:
        """All (key, config) pairs that apply to a detection class."""
        matches = [
            (key, ident)
            for key, ident in self._identifiers.items()
            if class_name in ident.applies_to
        ]
        if not matches and self.cfg.auto_add_classes:
            if class_name in self.cfg.auto_add_exclude:
                return []
            ident = IdentifierConfig(
                algo="generic",
                applies_to=[class_name],
                threshold=self.cfg.auto_add_threshold,
            )
            self._identifiers[class_name] = ident
            log.info("dynamically added generic identifier for class %r", class_name)
            matches = [(class_name, ident)]
        return matches

    def get(self, key: str) -> IdentifierConfig:
        return self._identifiers[key]

    def embedder_for(self, key: str):
        """Shared embedder instance for an identifier (one per algo —
        generic identifiers all reuse one backbone; collections keep
        their classes apart, not the embedder)."""
        algo = self._identifiers[key].algo
        embedder = self._embedders.get(algo)
        if embedder is None:
            from siteloom.identity.embedders import build_embedder

            embedder = build_embedder(
                algo,
                device=self._device,
                projection_path=self.cfg.face_projection_path or None,
            )
            self._embedders[algo] = embedder
        return embedder

    @property
    def known(self) -> dict[str, IdentifierConfig]:
        return dict(self._identifiers)
