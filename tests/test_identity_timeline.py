from eating_behavior.identity import IdentityDecision
from eating_behavior.models import CoatPrediction
from eating_behavior.pipeline import IdentityTimeline, event_cat_name


def make_identity(cat_name: str, method: str = "direct_by_single_same_coat") -> IdentityDecision:
    return IdentityDecision(
        cat_name=cat_name,
        method=method,
        confidence=1.0 if cat_name != "unknown" else 0.0,
        reason="test",
        candidates=(cat_name,) if cat_name != "unknown" else (),
    )


def make_coat(class_name: str) -> CoatPrediction:
    return CoatPrediction(
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        second_class_name="unknown",
        second_confidence=0.0,
    )


def test_event_cat_name_is_blank_for_unconfirmed_identity() -> None:
    identity = make_identity("unknown", method="needs_identity_confirmation")
    assert event_cat_name(identity) == ""


def test_identity_timeline_ignores_unconfirmed_identity() -> None:
    timeline = IdentityTimeline()
    timeline.add("cam01", 1.0, make_coat("silver_gray_white"), make_identity("unknown", "unregistered_coat"))
    assert timeline.infer("cam01", 2.0, 8.0) is None


def test_identity_timeline_infers_recent_same_coat_identity() -> None:
    timeline = IdentityTimeline()
    timeline.add("cam01", 1.0, make_coat("silver_gray_white"), make_identity("silver_pure"))
    timeline.add("cam01", 2.0, make_coat("blue_cat"), make_identity("blue"))

    observation = timeline.infer("cam01", 4.0, 8.0, coat_class="silver_gray_white")

    assert observation is not None
    assert observation.cat_name == "silver_pure"


def test_identity_timeline_prunes_old_observations() -> None:
    timeline = IdentityTimeline()
    timeline.add("cam01", 1.0, make_coat("silver_gray_white"), make_identity("silver_pure"))

    timeline.prune("cam01", 20.0, 8.0)

    assert timeline.infer("cam01", 20.0, 8.0, coat_class="silver_gray_white") is None
