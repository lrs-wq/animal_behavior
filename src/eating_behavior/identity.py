"""Cat identity routing.

The current identity stage is household-scoped coat routing. A stable
source_id/camera_id first selects the customer household, then coat_class is
matched only against cats registered in that household.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import config_bool
from .media import normalize_source_id


def normalized_confidence(value: float | None, *, default: float) -> float:
    """Clamp a finite confidence to [0, 1], rejecting invalid model values."""
    if value is None:
        return default
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(confidence):
        return 0.0
    return max(0.0, min(1.0, confidence))


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
        profiles: list[CatProfile] = []
        seen_names: set[str] = set()
        for item in cats:
            if not isinstance(item, dict):
                raise ValueError("identity cats must contain mappings")
            name = str(item.get("name") or "").strip()
            coat_class = str(item.get("coat_class") or "").strip()
            if not name or not coat_class:
                raise ValueError("identity cat entries require name and coat_class")
            if name in seen_names:
                raise ValueError(f"Duplicate cat name in household: {name}")
            seen_names.add(name)
            profiles.append(CatProfile(name=name, coat_class=coat_class))
        self.cats = profiles

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

    def decide(
        self,
        coat_class: str,
        crop: np.ndarray | None = None,
        *,
        evidence_confidence: float | None = None,
    ) -> IdentityDecision:
        """Decide identity from a household registry and a coat prediction."""
        candidates = self.registry.candidates_for_coat(coat_class)
        candidate_names = tuple(cat.name for cat in candidates)
        if self.direct_when_single_same_coat and len(candidates) == 1:
            # This is the only current path that confirms a public cat name.
            return IdentityDecision(
                cat_name=candidates[0].name,
                method="direct_by_single_same_coat",
                confidence=normalized_confidence(evidence_confidence, default=1.0),
                reason="only_one_registered_cat_has_this_coat",
                candidates=candidate_names,
                household_id=self.registry.household_id,
            )
        if len(candidates) == 1:
            # The operator explicitly disabled coat-only confirmation. Preserve
            # the candidate for a future Re-ID stage instead of incorrectly
            # reporting that the coat is unregistered.
            return IdentityDecision(
                cat_name=self.registry.unknown_name,
                method="needs_identity_confirmation",
                confidence=0.0,
                reason="direct_single_same_coat_confirmation_disabled",
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
    return normalize_source_id(source)


def source_list(value: Any) -> tuple[str, ...]:
    """Normalize source_id config values into a tuple.

    Supports a string, comma-separated string, list, tuple, or set.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        normalized = (normalize_source_key(item) for item in value.split(","))
        return tuple(item for item in normalized if item)
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
        allow_default_fallback: bool | None = None,
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
        # A routing table represents a multi-household boundary, so unknown
        # sources fail closed unless fallback was explicitly enabled.
        self.allow_default_fallback = (
            not bool(self.services_by_source) if allow_default_fallback is None else allow_default_fallback
        )

    @classmethod
    def from_config(cls, identity_config: dict[str, Any]) -> "SourceIdentityService":
        """Build source_id -> household identity services from config."""
        direct_when_single_same_coat = config_bool(
            identity_config.get("direct_when_single_same_coat", True),
            field_name="identity.direct_when_single_same_coat",
        )
        default_unknown_name = str(identity_config.get("unknown_name", "unknown"))
        default_cats = identity_config.get("cats") or []
        if not isinstance(default_cats, list):
            raise ValueError("identity.cats must be a list of mappings")
        default_registry = HouseholdRegistry(
            cats=default_cats,
            unknown_name=default_unknown_name,
            household_id=str(identity_config.get("household_id", "default")),
        )
        registries_by_source: dict[str, HouseholdRegistry] = {}
        households_value = identity_config.get("households") or []
        if not isinstance(households_value, list):
            raise ValueError("identity.households must be a list of mappings")
        households = list(households_value)
        fallback_value = identity_config.get("allow_default_fallback", not bool(households))
        if not isinstance(fallback_value, bool):
            raise ValueError("identity.allow_default_fallback must be a boolean")
        seen_household_ids: set[str] = set()
        for household_index, household in enumerate(households, start=1):
            if not isinstance(household, dict):
                raise ValueError("identity.households items must be mappings")
            household_id = str(
                household.get("household_id")
                or household.get("id")
                or household.get("name")
                or f"household_{household_index}"
            ).strip()
            if household_id in seen_household_ids:
                raise ValueError(f"Duplicate identity household_id: {household_id}")
            seen_household_ids.add(household_id)
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
            household_cats = household.get("cats") or []
            if not isinstance(household_cats, list):
                raise ValueError(
                    f"identity household cats must be a list: {household_id}"
                )
            registry = HouseholdRegistry(
                cats=household_cats,
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
            allow_default_fallback=fallback_value,
        )

    def decide(
        self,
        source: str,
        coat_class: str,
        crop: np.ndarray | None = None,
        *,
        evidence_confidence: float | None = None,
    ) -> IdentityDecision:
        """Select household by source_id, then decide identity inside it."""
        source_key = normalize_source_key(source)
        service = self.services_by_source.get(source_key)
        if service is not None:
            return service.decide(
                coat_class,
                crop,
                evidence_confidence=evidence_confidence,
            )
        if self.allow_default_fallback and self.default_registry.cats:
            # Default household keeps single-customer/debug configs working.
            return self.default_service.decide(
                coat_class,
                crop,
                evidence_confidence=evidence_confidence,
            )
        return IdentityDecision(
            cat_name=self.default_registry.unknown_name,
            method="unregistered_source",
            confidence=0.0,
            reason="no_household_registered_for_source",
            candidates=(),
            household_id="",
        )

    def reject(
        self,
        source: str,
        *,
        method: str,
        reason: str,
        candidates: tuple[str, ...] = (),
    ) -> IdentityDecision:
        """Return an explicitly unconfirmed decision in the routed household."""
        source_key = normalize_source_key(source)
        service = self.services_by_source.get(source_key)
        if service is None and self.allow_default_fallback and self.default_registry.cats:
            service = self.default_service
        if service is None:
            return IdentityDecision(
                cat_name=self.default_registry.unknown_name,
                method="unregistered_source",
                confidence=0.0,
                reason="no_household_registered_for_source",
                candidates=(),
                household_id="",
            )
        return IdentityDecision(
            cat_name=service.registry.unknown_name,
            method=method,
            confidence=0.0,
            reason=reason,
            candidates=candidates,
            household_id=service.registry.household_id,
        )

    def household_id_for_source(self, source: str) -> str:
        """Return household_id for source_id even when cat identity is unknown."""
        source_key = normalize_source_key(source)
        service = self.services_by_source.get(source_key)
        if service is not None:
            return service.registry.household_id
        if self.allow_default_fallback and self.default_registry.cats:
            return self.default_registry.household_id
        return ""

    def validate_configured_sources(self, source_ids: list[str] | tuple[str, ...]) -> None:
        """Require every configured source to cross a strict household boundary."""
        if not self.services_by_source or self.allow_default_fallback:
            return
        missing = [
            normalize_source_key(source_id)
            for source_id in source_ids
            if normalize_source_key(source_id) not in self.services_by_source
        ]
        if missing:
            raise ValueError(
                "Configured source_id has no household identity mapping: "
                + ", ".join(missing)
            )
