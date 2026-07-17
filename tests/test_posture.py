"""Pure AP-10K cat-posture classifier tests."""

from __future__ import annotations

import math

import pytest

from eating_behavior.pose import AP10K_KEYPOINT_NAMES, PoseKeypoint, PosePrediction
from eating_behavior.posture import (
    CAT_POSTURE_CHINESE_LABELS,
    CatPosture,
    CatPostureClassifier,
    PostureConfig,
    PostureObservation,
)


def pose(
    stance: str = "standing",
    *,
    confidence: float = 0.9,
    low_indices: tuple[int, ...] = (),
) -> PosePrediction:
    """Build a fixed-camera side-view AP-10K skeleton."""
    paw_y = {
        "standing": (145.0, 140.0),
        "crouching": (116.0, 116.0),
        "lying": (94.0, 94.0),
    }[stance]
    if stance == "standing":
        front_joints = ((55.0, 110.0), (65.0, 110.0))
        hind_joints = ((135.0, 108.0), (145.0, 108.0))
    elif stance == "crouching":
        front_joints = ((85.0, 92.0), (35.0, 92.0))
        hind_joints = ((165.0, 92.0), (115.0, 92.0))
    else:
        front_joints = ((70.0, 86.0), (50.0, 86.0))
        hind_joints = ((150.0, 86.0), (130.0, 86.0))

    coordinates = (
        (25.0, 62.0),  # eyes, nose, neck, tail root
        (31.0, 62.0),
        (28.0, 68.0),
        (50.0, 80.0),
        (140.0, 80.0),
        (55.0, 80.0),
        front_joints[0],
        (55.0, paw_y[0]),
        (65.0, 80.0),
        front_joints[1],
        (65.0, paw_y[0]),
        (135.0, 80.0),
        hind_joints[0],
        (135.0, paw_y[1]),
        (145.0, 80.0),
        hind_joints[1],
        (145.0, paw_y[1]),
    )
    points = tuple(
        PoseKeypoint(
            index,
            AP10K_KEYPOINT_NAMES[index],
            x,
            y,
            0.1 if index in low_indices else confidence,
        )
        for index, (x, y) in enumerate(coordinates)
    )
    return PosePrediction((0, 0, 200, 200), points, confidence)


def subject(**overrides) -> CatPostureClassifier:
    return CatPostureClassifier(PostureConfig(enabled=True, **overrides))


@pytest.mark.parametrize(
    ("stance", "feeding_confidence", "expected"),
    [
        ("standing", 0.9, CatPosture.EATING),
        ("lying", 0.0, CatPosture.LYING),
        ("standing", 0.0, CatPosture.STANDING),
        ("crouching", 0.0, CatPosture.CROUCHING),
    ],
)
def test_classifier_outputs_all_supported_postures(
    stance: str,
    feeding_confidence: float,
    expected: CatPosture,
) -> None:
    observation = subject().classify(
        pose(stance),
        "accepted",
        feeding_confidence,
    )

    assert observation is not None
    assert observation.posture is expected
    assert observation.chinese_label == CAT_POSTURE_CHINESE_LABELS[expected]
    assert 0.0 < observation.confidence <= 1.0
    assert dict(observation.features)["visible_keypoints"] == 17.0


def test_eating_requires_an_unambiguous_feeding_association() -> None:
    classifier = subject(min_feeding_confidence=0.5)

    without_association = classifier.classify(pose(), "accepted", 0.0)
    below_threshold = classifier.classify(pose(), "accepted", 0.499)
    exact_threshold = classifier.classify(pose(), "accepted", 0.5)

    assert without_association is not None
    assert without_association.posture is CatPosture.STANDING
    assert below_threshold is not None
    assert below_threshold.posture is CatPosture.STANDING
    assert exact_threshold is not None
    assert exact_threshold.posture is CatPosture.EATING
    assert exact_threshold.evidence_source == "feeding_head_association"


def test_zero_configured_feeding_threshold_still_requires_positive_evidence() -> None:
    observation = subject(min_feeding_confidence=0.0).classify(
        pose(),
        "accepted",
        0.0,
    )

    assert observation is not None
    assert observation.posture is CatPosture.STANDING


@pytest.mark.parametrize("status", ["held", "missing", "rejected", ""])
def test_stale_or_rejected_pose_cannot_advance_posture(status: str) -> None:
    assert subject().classify(pose(), status, 0.9) is None


def test_accepted_and_raw_are_the_only_advancing_statuses() -> None:
    classifier = subject()

    assert classifier.classify(pose(), "accepted") is not None
    assert classifier.classify(pose(), "raw") is not None


def test_missing_keypoint_quality_or_geometry_returns_no_observation() -> None:
    classifier = subject()

    assert classifier.classify(pose(confidence=0.2), "accepted") is None
    assert (
        classifier.classify(
            pose(low_indices=(4, 11, 14)),
            "accepted",
        )
        is None
    )


def test_inverted_paws_are_not_misclassified_as_lying() -> None:
    prediction = pose("lying")
    inverted = PosePrediction(
        prediction.cat_xyxy,
        tuple(
            PoseKeypoint(point.index, point.name, point.x, 70.0, point.confidence)
            if point.index in {7, 10, 13, 16}
            else point
            for point in prediction.keypoints
        ),
        prediction.mean_confidence,
    )

    assert subject().classify(inverted, "accepted") is None


def test_geometric_confidence_is_conservatively_capped() -> None:
    observation = subject().classify(pose(confidence=1.0), "accepted")

    assert observation is not None
    assert observation.posture is CatPosture.STANDING
    assert observation.confidence <= 0.85


def test_disabled_classifier_returns_no_observation() -> None:
    classifier = CatPostureClassifier(PostureConfig(enabled=False))

    assert classifier.classify(pose(), "accepted", 0.9) is None


@pytest.mark.parametrize("value", [True, math.nan, math.inf, -0.1, 1.1])
def test_feeding_confidence_is_strictly_validated(value: float) -> None:
    with pytest.raises(ValueError, match="feeding_confidence"):
        subject().classify(pose(), "accepted", value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"enabled": 1}, "posture.enabled"),
        ({"min_keypoint_confidence": True}, "min_keypoint_confidence"),
        ({"min_keypoint_confidence": math.nan}, "min_keypoint_confidence"),
        ({"min_feeding_confidence": math.inf}, "min_feeding_confidence"),
        ({"min_body_length_ratio": -0.1}, "min_body_length_ratio"),
        ({"min_body_length_ratio": 0.0}, "min_body_length_ratio"),
        ({"min_visible_keypoints": True}, "min_visible_keypoints"),
        ({"min_visible_keypoints": 18}, "min_visible_keypoints"),
        ({"min_supporting_paws": 1}, "min_supporting_paws"),
        ({"min_complete_leg_chains": 0}, "min_complete_leg_chains"),
        (
            {"transition_confirm_seconds": math.nan},
            "transition_confirm_seconds",
        ),
        (
            {"transition_confirm_seconds": -1.0},
            "transition_confirm_seconds",
        ),
        ({"min_confirmed_observations": False}, "min_confirmed_observations"),
        ({"min_confirmed_observations": 0}, "min_confirmed_observations"),
        ({"max_anonymous_subjects_per_source": True}, "max_anonymous"),
        ({"max_anonymous_subjects_per_source": 0}, "max_anonymous"),
        (
            {
                "crouching_min_clearance_ratio": 0.3,
                "crouching_max_clearance_ratio": 0.2,
            },
            "crouching_min_clearance_ratio",
        ),
        (
            {
                "lying_max_clearance_ratio": 0.3,
                "standing_min_clearance_ratio": 0.2,
            },
            "lying_max_clearance_ratio",
        ),
        (
            {
                "crouching_min_clearance_ratio": 0.3,
                "standing_min_clearance_ratio": 0.2,
                "crouching_max_clearance_ratio": 0.4,
            },
            "crouching_min_clearance_ratio",
        ),
    ],
)
def test_config_rejects_bool_nonfinite_invalid_and_inverted_values(
    kwargs: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PostureConfig(**kwargs)


def test_observation_rejects_unsafe_public_values() -> None:
    with pytest.raises(ValueError, match="confidence"):
        PostureObservation(CatPosture.LYING, math.nan, "pose_geometry")
    with pytest.raises(ValueError, match="evidence_source"):
        PostureObservation(CatPosture.LYING, 0.5, " ")
    with pytest.raises(ValueError, match="feature values"):
        PostureObservation(
            CatPosture.LYING,
            0.5,
            "pose_geometry",
            (("bad", math.inf),),
        )
