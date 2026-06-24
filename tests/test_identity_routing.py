import pytest

from eating_behavior.identity import HouseholdRegistry, IdentityService, SourceIdentityService


def test_single_same_coat_confirms_identity() -> None:
    registry = HouseholdRegistry([{"name": "blue", "coat_class": "blue_cat"}])
    decision = IdentityService(registry).decide("blue_cat")
    assert decision.cat_name == "blue"
    assert decision.method == "direct_by_single_same_coat"


def test_multiple_same_coat_requires_identity_confirmation() -> None:
    registry = HouseholdRegistry(
        [
            {"name": "silver_a", "coat_class": "silver_gray_white"},
            {"name": "silver_b", "coat_class": "silver_gray_white"},
        ]
    )
    decision = IdentityService(registry).decide("silver_gray_white")
    assert decision.cat_name == "unknown"
    assert decision.method == "needs_identity_confirmation"


def test_source_identity_routes_same_coat_to_source_household() -> None:
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


def test_source_identity_rejects_duplicate_source_mapping() -> None:
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
