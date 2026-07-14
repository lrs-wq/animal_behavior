"""Tests for pure spatial feeding-head to cat association."""

import pytest

from eating_behavior.association import associate_feeding_to_cat
from eating_behavior.identity import IdentityDecision
from eating_behavior.models import CoatPrediction, Detection


def make_cat(
    name: str,
    xyxy: tuple[int, int, int, int],
) -> tuple[Detection, CoatPrediction, IdentityDecision]:
    coat = CoatPrediction(0, f"{name}_coat", 0.9, "other", 0.1)
    identity = IdentityDecision(
        cat_name=name,
        method="direct_by_single_same_coat",
        confidence=1.0,
        reason="test",
        candidates=(name,),
    )
    return Detection(0, "cat", 0.9, xyxy), coat, identity


def feeding(xyxy: tuple[int, int, int, int]) -> Detection:
    return Detection(1, "feeding_head", 0.9, xyxy)


def test_center_containment_and_overlap_select_cat() -> None:
    result = associate_feeding_to_cat(
        feeding((4, 4, 8, 8)),
        [make_cat("inside", (0, 0, 20, 20))],
    )

    assert result is not None
    assert result.identity.cat_name == "inside"
    assert result.feeding_center_inside is True
    assert result.center_distance == 0.0
    assert result.feeding_coverage == 1.0


def test_partial_overlap_is_spatial_evidence_when_center_is_outside() -> None:
    result = associate_feeding_to_cat(
        feeding((8, 2, 14, 8)),
        [make_cat("overlap", (0, 0, 10, 10))],
    )

    assert result is not None
    assert result.identity.cat_name == "overlap"
    assert result.feeding_center_inside is False
    assert result.feeding_coverage == pytest.approx(1 / 3)


def test_far_feeding_is_not_forced_to_a_cat() -> None:
    result = associate_feeding_to_cat(
        feeding((100, 100, 104, 104)),
        [make_cat("far", (0, 0, 20, 20))],
        max_distance_ratio=0.5,
    )

    assert result is None


def test_tiny_overlap_does_not_bypass_maximum_distance() -> None:
    result = associate_feeding_to_cat(
        feeding((9, 0, 1009, 10)),
        [make_cat("barely_overlapping", (0, 0, 10, 10))],
        max_distance_ratio=0.5,
    )

    assert result is None


def test_regression_axis_aligned_far_box_does_not_look_close() -> None:
    # The old formula took the minimum distance to any rectangle edge. Because
    # the feeding center is aligned with the cat's y-range, it returned a tiny
    # distance despite the 81-pixel horizontal gap.
    result = associate_feeding_to_cat(
        feeding((90, 0, 92, 2)),
        [make_cat("left", (0, 0, 10, 10))],
        max_distance_ratio=0.5,
    )

    assert result is None


def test_nearly_equal_overlapping_candidates_are_ambiguous() -> None:
    result = associate_feeding_to_cat(
        feeding((8, 8, 12, 12)),
        [
            make_cat("first", (0, 0, 20, 20)),
            make_cat("second", (0, 0, 20, 20)),
        ],
        ambiguity_margin=0.1,
    )

    assert result is None


def test_clearer_candidate_wins_in_multi_cat_scene() -> None:
    result = associate_feeding_to_cat(
        feeding((5, 5, 9, 9)),
        [
            make_cat("containing", (0, 0, 20, 20)),
            make_cat("nearby", (20, 0, 40, 20)),
        ],
        max_distance_ratio=1.0,
        ambiguity_margin=0.1,
    )

    assert result is not None
    assert result.identity.cat_name == "containing"


@pytest.mark.parametrize(
    ("max_distance_ratio", "ambiguity_margin"),
    [(-0.1, 0.1), (float("inf"), 0.1), (0.5, -0.1), (0.5, float("nan"))],
)
def test_invalid_thresholds_are_rejected(
    max_distance_ratio: float,
    ambiguity_margin: float,
) -> None:
    with pytest.raises(ValueError):
        associate_feeding_to_cat(
            feeding((0, 0, 2, 2)),
            [],
            max_distance_ratio=max_distance_ratio,
            ambiguity_margin=ambiguity_margin,
        )
