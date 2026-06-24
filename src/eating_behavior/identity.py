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
    household_id: str = "default"


class HouseholdRegistry:
    def __init__(
        self,
        cats: list[dict[str, Any]],
        unknown_name: str = "unknown",
        household_id: str = "default",
        source_ids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.unknown_name = unknown_name
        self.household_id = household_id
        self.source_ids = tuple(normalize_source_key(source_id) for source_id in (source_ids or ()))
        self.sources = self.source_ids
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
                household_id=self.registry.household_id,
            )
        if len(candidates) > 1:
            return IdentityDecision(
                cat_name=self.registry.unknown_name,
                method="needs_identity_confirmation",
                confidence=0.0,
                reason="multiple_registered_cats_share_this_coat",
                candidates=candidate_names,
                household_id=self.registry.household_id,
            )
        return IdentityDecision(
            cat_name=self.registry.unknown_name,
            method="unregistered_coat",
            confidence=0.0,
            reason="no_registered_cat_has_this_coat",
            candidates=(),
            household_id=self.registry.household_id,
        )


def normalize_source_key(source: str) -> str:
    return str(source).strip()


def source_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple, set)):
        return tuple(normalize_source_key(item) for item in value if str(item).strip())
    raise TypeError(f"source must be a string or a list of strings, got {type(value).__name__}")


class SourceIdentityService:
    """Route identity decisions to the household registered for each stable source_id."""

    def __init__(
        self,
        default_registry: HouseholdRegistry,
        registries_by_source: dict[str, HouseholdRegistry] | None = None,
        direct_when_single_same_coat: bool = True,
    ) -> None:
        self.default_registry = default_registry
        self.default_service = IdentityService(default_registry, direct_when_single_same_coat)
        self.direct_when_single_same_coat = direct_when_single_same_coat
        self.services_by_source: dict[str, IdentityService] = {}
        for source, registry in (registries_by_source or {}).items():
            source_key = normalize_source_key(source)
            if not source_key:
                continue
            if source_key in self.services_by_source:
                raise ValueError(f"Duplicate identity source mapping: {source_key}")
            self.services_by_source[source_key] = IdentityService(registry, direct_when_single_same_coat)

    @classmethod
    def from_config(cls, identity_config: dict[str, Any]) -> "SourceIdentityService":
        direct_when_single_same_coat = bool(identity_config.get("direct_when_single_same_coat", True))
        default_unknown_name = str(identity_config.get("unknown_name", "unknown"))
        default_registry = HouseholdRegistry(
            cats=list(identity_config.get("cats") or []),
            unknown_name=default_unknown_name,
            household_id=str(identity_config.get("household_id", "default")),
        )
        registries_by_source: dict[str, HouseholdRegistry] = {}
        for household in list(identity_config.get("households") or []):
            if not isinstance(household, dict):
                raise ValueError("identity.households items must be mappings")
            household_id = str(
                household.get("household_id")
                or household.get("id")
                or household.get("name")
                or f"household_{len(registries_by_source) + 1}"
            )
            sources = source_list(
                household.get(
                    "source_ids",
                    household.get(
                        "camera_ids",
                        household.get("camera_id", household.get("sources", household.get("source"))),
                    ),
                )
            )
            registry = HouseholdRegistry(
                cats=list(household.get("cats") or []),
                unknown_name=str(household.get("unknown_name", default_unknown_name)),
                household_id=household_id,
                source_ids=list(sources),
            )
            for source in sources:
                if source in registries_by_source:
                    raise ValueError(f"Duplicate identity source mapping: {source}")
                registries_by_source[source] = registry
        return cls(
            default_registry=default_registry,
            registries_by_source=registries_by_source,
            direct_when_single_same_coat=direct_when_single_same_coat,
        )

    def decide(self, source: str, coat_class: str, crop: np.ndarray | None = None) -> IdentityDecision:
        source_key = normalize_source_key(source)
        service = self.services_by_source.get(source_key)
        if service is not None:
            return service.decide(coat_class, crop)
        if self.default_registry.cats:
            return self.default_service.decide(coat_class, crop)
        return IdentityDecision(
            cat_name=self.default_registry.unknown_name,
            method="unregistered_source",
            confidence=0.0,
            reason="no_household_registered_for_source",
            candidates=(),
            household_id=self.default_registry.household_id,
        )

    def household_id_for_source(self, source: str) -> str:
        source_key = normalize_source_key(source)
        service = self.services_by_source.get(source_key)
        if service is not None:
            return service.registry.household_id
        if self.default_registry.cats:
            return self.default_registry.household_id
        return ""
