"""Tests for household and source_id identity routing."""

import pytest

from eating_behavior.identity import HouseholdRegistry, IdentityService, SourceIdentityService


def test_single_same_coat_confirms_identity() -> None:
    """A single cat with the predicted coat can be confirmed directly."""
    registry = HouseholdRegistry([{"name": "blue", "coat_class": "blue_cat"}])
    decision = IdentityService(registry).decide("blue_cat")
    assert decision.cat_name == "blue"
    assert decision.method == "direct_by_single_same_coat"


def test_multiple_same_coat_requires_identity_confirmation() -> None:
    """Multiple cats with the same coat must stay unconfirmed."""
    registry = HouseholdRegistry(
        [
            {"name": "silver_a", "coat_class": "silver_gray_white"},
            {"name": "silver_b", "coat_class": "silver_gray_white"},
        ]
    )
    decision = IdentityService(registry).decide("silver_gray_white")
    assert decision.cat_name == "unknown"
    assert decision.method == "needs_identity_confirmation"


def test_single_same_coat_stays_unknown_when_direct_confirmation_is_disabled() -> None:
    """A disabled direct route must not masquerade as an unregistered coat."""
    registry = HouseholdRegistry([{"name": "blue", "coat_class": "blue_cat"}])

    decision = IdentityService(registry, direct_when_single_same_coat=False).decide("blue_cat")

    assert decision.cat_name == "unknown"
    assert decision.method == "needs_identity_confirmation"
    assert decision.reason == "direct_single_same_coat_confirmation_disabled"
    assert decision.candidates == ("blue",)


def test_source_identity_routes_same_coat_to_source_household() -> None:
    """The same coat_class can map to different cats by source_id."""
    service = SourceIdentityService.from_config(
        {
            "households": [
                {
                    "household_id": "customer_a_home",
                    "source_ids": ["cam_customer_a_01"],
                    "cats": [{"name": "customer_a_blue", "coat_class": "blue_cat"}],
                },
                {
                    "household_id": "customer_b_home",
                    "camera_ids": ["cam_customer_b_01"],
                    "cats": [{"name": "customer_b_blue", "coat_class": "blue_cat"}],
                },
            ]
        }
    )

    customer_a = service.decide("cam_customer_a_01", "blue_cat")
    customer_b = service.decide("cam_customer_b_01", "blue_cat")

    assert customer_a.cat_name == "customer_a_blue"
    assert customer_a.household_id == "customer_a_home"
    assert customer_b.cat_name == "customer_b_blue"
    assert customer_b.household_id == "customer_b_home"


def test_source_identity_rejects_unregistered_source_without_default_household() -> None:
    """An unknown source_id must not reuse another household."""
    service = SourceIdentityService.from_config(
        {
            "households": [
                {
                    "household_id": "customer_a_home",
                    "source_ids": ["cam_customer_a_01"],
                    "cats": [{"name": "customer_a_blue", "coat_class": "blue_cat"}],
                }
            ]
        }
    )

    decision = service.decide("cam_customer_b_01", "blue_cat")

    assert decision.cat_name == "unknown"
    assert decision.method == "unregistered_source"
    assert decision.household_id == ""
    assert service.household_id_for_source("cam_customer_b_01") == ""


def test_multi_household_unknown_source_does_not_implicitly_use_default_cats() -> None:
    """Default cats cannot cross a configured household routing boundary."""
    service = SourceIdentityService.from_config(
        {
            "household_id": "legacy_default",
            "cats": [{"name": "default_blue", "coat_class": "blue_cat"}],
            "households": [
                {
                    "household_id": "customer_a_home",
                    "source_ids": ["cam_customer_a_01"],
                    "cats": [{"name": "customer_a_blue", "coat_class": "blue_cat"}],
                }
            ],
        }
    )

    decision = service.decide("cam_unknown", "blue_cat")

    assert decision.cat_name == "unknown"
    assert decision.method == "unregistered_source"
    assert decision.household_id == ""
    assert service.household_id_for_source("cam_unknown") == ""


def test_multi_household_can_explicitly_allow_default_fallback() -> None:
    """An explicit opt-in preserves intentional legacy fallback behavior."""
    service = SourceIdentityService.from_config(
        {
            "allow_default_fallback": True,
            "household_id": "legacy_default",
            "cats": [{"name": "default_blue", "coat_class": "blue_cat"}],
            "households": [
                {
                    "household_id": "customer_a_home",
                    "source_ids": ["cam_customer_a_01"],
                    "cats": [{"name": "customer_a_blue", "coat_class": "blue_cat"}],
                }
            ],
        }
    )

    decision = service.decide("cam_unknown", "blue_cat")

    assert decision.cat_name == "default_blue"
    assert decision.household_id == "legacy_default"
    assert service.household_id_for_source("cam_unknown") == "legacy_default"


def test_allow_default_fallback_requires_a_boolean() -> None:
    with pytest.raises(ValueError, match="allow_default_fallback must be a boolean"):
        SourceIdentityService.from_config({"allow_default_fallback": "false"})


def test_strict_household_routing_validates_configured_sources() -> None:
    service = SourceIdentityService.from_config(
        {
            "households": [
                {
                    "household_id": "customer_a_home",
                    "source_ids": ["cam_customer_a_01"],
                    "cats": [{"name": "blue", "coat_class": "blue_cat"}],
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="no household identity mapping"):
        service.validate_configured_sources(("cam_customer_a_01", "cam_unknown"))


def test_quoted_false_disables_direct_confirmation() -> None:
    service = SourceIdentityService.from_config(
        {
            "direct_when_single_same_coat": "false",
            "cats": [{"name": "blue", "coat_class": "blue_cat"}],
        }
    )

    decision = service.decide("cam_debug", "blue_cat")

    assert decision.cat_name == "unknown"
    assert decision.reason == "direct_single_same_coat_confirmation_disabled"


def test_identity_cats_require_a_list_of_complete_unique_mappings() -> None:
    with pytest.raises(ValueError, match="identity.cats must be a list"):
        SourceIdentityService.from_config(
            {"cats": {"name": "blue", "coat_class": "blue_cat"}}
        )
    with pytest.raises(ValueError, match="require name and coat_class"):
        SourceIdentityService.from_config({"cats": [{"name": "blue"}]})
    with pytest.raises(ValueError, match="Duplicate cat name"):
        SourceIdentityService.from_config(
            {
                "cats": [
                    {"name": "blue", "coat_class": "blue_cat"},
                    {"name": "blue", "coat_class": "other"},
                ]
            }
        )


def test_source_identity_rejects_duplicate_source_mapping() -> None:
    """One source_id cannot belong to two households."""
    with pytest.raises(ValueError, match="Duplicate identity source mapping"):
        SourceIdentityService.from_config(
            {
                "households": [
                    {
                        "household_id": "customer_a_home",
                        "source_ids": ["cam_shared_01"],
                        "cats": [{"name": "customer_a_blue", "coat_class": "blue_cat"}],
                    },
                    {
                        "household_id": "customer_b_home",
                        "camera_ids": ["cam_shared_01"],
                        "cats": [{"name": "customer_b_blue", "coat_class": "blue_cat"}],
                    },
                ]
            }
        )


def test_source_identity_rejects_duplicate_household_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate identity household_id"):
        SourceIdentityService.from_config(
            {
                "households": [
                    {
                        "household_id": "shared",
                        "source_ids": ["cam_a"],
                        "cats": [{"name": "a", "coat_class": "blue_cat"}],
                    },
                    {
                        "household_id": "shared",
                        "source_ids": ["cam_b"],
                        "cats": [{"name": "b", "coat_class": "blue_cat"}],
                    },
                ]
            }
        )
