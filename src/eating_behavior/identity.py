from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CatProfile:
    name: str
    coat_class: str


@dataclass(frozen=True)
class IdentityDecision:
    cat_name: str
    method: str
    confidence: float
    reason: str
    candidates: tuple[str, ...]


class HouseholdRegistry:
    def __init__(self, cats: list[dict[str, Any]], unknown_name: str = "unknown") -> None:
        self.unknown_name = unknown_name
        self.cats = [
            CatProfile(name=str(item["name"]), coat_class=str(item["coat_class"]))
            for item in cats
            if item.get("name") and item.get("coat_class")
        ]

    def candidates_for_coat(self, coat_class: str) -> list[CatProfile]:
        return [cat for cat in self.cats if cat.coat_class == coat_class]


class IdentityService:
    """Identity confirmation placeholder.

    The first formal version supports the business routing rule: if only one
    registered cat has the predicted coat class, confirm directly. When several
    cats share the same coat class, this class returns unknown until a gallery
    Re-ID backend is attached.
    """

    def __init__(
        self,
        registry: HouseholdRegistry,
        direct_when_single_same_coat: bool = True,
    ) -> None:
        self.registry = registry
        self.direct_when_single_same_coat = direct_when_single_same_coat

    def decide(self, coat_class: str, crop: np.ndarray | None = None) -> IdentityDecision:
        candidates = self.registry.candidates_for_coat(coat_class)
        candidate_names = tuple(cat.name for cat in candidates)
        if self.direct_when_single_same_coat and len(candidates) == 1:
            return IdentityDecision(
                cat_name=candidates[0].name,
                method="direct_by_single_same_coat",
                confidence=1.0,
                reason="only_one_registered_cat_has_this_coat",
                candidates=candidate_names,
            )
        if len(candidates) > 1:
            return IdentityDecision(
                cat_name=self.registry.unknown_name,
                method="needs_identity_confirmation",
                confidence=0.0,
                reason="multiple_registered_cats_share_this_coat",
                candidates=candidate_names,
            )
        return IdentityDecision(
            cat_name=self.registry.unknown_name,
            method="unregistered_coat",
            confidence=0.0,
            reason="no_registered_cat_has_this_coat",
            candidates=(),
        )

