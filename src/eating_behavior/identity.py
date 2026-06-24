"""Cat identity routing.

The current identity stage is household-scoped coat routing. A stable
source_id/camera_id first selects the customer household, then coat_class is
matched only against cats registered in that household.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CatProfile:
    """Registered cat profile inside one household."""

    name: str
    coat_class: str


@dataclass(frozen=True)
class IdentityDecision:
    """Result of one identity decision.

    cat_name may be an internal unknown value. Public event output converts
    unconfirmed identities to an empty string.
    """

    cat_name: str
    method: str
    confidence: float
    reason: str
    candidates: tuple[str, ...]
    household_id: str = "default"


class HouseholdRegistry:
    """Registered cats and source IDs for one household."""

    def __init__(
        self,
        cats: list[dict[str, Any]],
        unknown_name: str = "unknown",
        household_id: str = "default",
        source_ids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.unknown_name = unknown_name
        self.household_id = household_id
        # source_ids are stable camera IDs owned by this household. sources is
        # kept as a backward-compatible alias for older code/config.
        self.source_ids = tuple(normalize_source_key(source_id) for source_id in (source_ids or ()))
        self.sources = self.source_ids
        # Ignore incomplete cat entries so bad config does not enter hot paths.
        self.cats = [
            CatProfile(name=str(item["name"]), coat_class=str(item["coat_class"]))
            for item in cats
            if item.get("name") and item.get("coat_class")
        ]

    def candidates_for_coat(self, coat_class: str) -> list[CatProfile]:
        """Return cats in this household with the predicted coat_class."""
        return [cat for cat in self.cats if cat.coat_class == coat_class]


class IdentityService:
    """Household-local identity service.

    If exactly one registered cat has the predicted coat_class, confirm it
    directly. Multiple same-coat cats stay unknown until a Re-ID backend is
    attached.
    """

    def __init__(
        self,
        registry: HouseholdRegistry,
        direct_when_single_same_coat: bool = True,
    ) -> None:
        self.registry = registry
        self.direct_when_single_same_coat = direct_when_single_same_coat

    def decide(self, coat_class: str, crop: np.ndarray | None = None) -> IdentityDecision:
        """Decide identity from a household registry and a coat prediction."""
        candidates = self.registry.candidates_for_coat(coat_class)
        candidate_names = tuple(cat.name for cat in candidates)
        if self.direct_when_single_same_coat and len(candidates) == 1:
            # This is the only current path that confirms a public cat name.
            return IdentityDecision(
                cat_name=candidates[0].name,
                method="direct_by_single_same_coat",
                confidence=1.0,
                reason="only_one_registered_cat_has_this_coat",
                candidates=candidate_names,
                household_id=self.registry.household_id,
            )
        if len(candidates) > 1:
            # Do not guess between multiple cats with the same coat_class.
            return IdentityDecision(
                cat_name=self.registry.unknown_name,
                method="needs_identity_confirmation",
                confidence=0.0,
                reason="multiple_registered_cats_share_this_coat",
                candidates=candidate_names,
                household_id=self.registry.household_id,
            )
        # No registered cat has this coat in the selected household.
        return IdentityDecision(
            cat_name=self.registry.unknown_name,
            method="unregistered_coat",
            confidence=0.0,
            reason="no_registered_cat_has_this_coat",
            candidates=(),
            household_id=self.registry.household_id,
        )


def normalize_source_key(source: str) -> str:
    """Normalize source_id/camera_id strings used as dictionary keys."""
    return str(source).strip()


def source_list(value: Any) -> tuple[str, ...]:
    """Normalize source_id config values into a tuple.

    Supports a string, comma-separated string, list, tuple, or set.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple, set)):
        return tuple(normalize_source_key(item) for item in value if str(item).strip())
    raise TypeError(f"source must be a string or a list of strings, got {type(value).__name__}")


class SourceIdentityService:
    """Route identity decisions by stable source_id/camera_id."""

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
            # One source_id must map to exactly one household.
            self.services_by_source[source_key] = IdentityService(registry, direct_when_single_same_coat)

    @classmethod
    def from_config(cls, identity_config: dict[str, Any]) -> "SourceIdentityService":
        """Build source_id -> household identity services from config."""
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
                        # camera_ids/camera_id are aliases for database naming.
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
        """Select household by source_id, then decide identity inside it."""
        source_key = normalize_source_key(source)
        service = self.services_by_source.get(source_key)
        if service is not None:
            return service.decide(coat_class, crop)
        if self.default_registry.cats:
            # Default household keeps single-customer/debug configs working.
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
        """Return household_id for source_id even when cat identity is unknown."""
        source_key = normalize_source_key(source)
        service = self.services_by_source.get(source_key)
        if service is not None:
            return service.registry.household_id
        if self.default_registry.cats:
            return self.default_registry.household_id
        return ""
