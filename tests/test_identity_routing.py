from eating_behavior.identity import HouseholdRegistry, IdentityService


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

