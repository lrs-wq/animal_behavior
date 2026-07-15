"""Source-local cat tracking and bbox stabilization tests."""

from __future__ import annotations

import math

import pytest

from eating_behavior.models import Detection
from eating_behavior.tracking import CatTracker, TrackingConfig


def cat(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> Detection:
    return Detection(0, "cat", confidence, (x1, y1, x2, y2))


def tracker(**overrides) -> CatTracker:
    return CatTracker(TrackingConfig(enabled=True, **overrides))


def test_tracker_preserves_ids_when_detector_order_changes() -> None:
    subject = tracker()
    first = subject.update(
        "cam-a",
        0.0,
        [cat(0, 0, 100, 100), cat(300, 0, 400, 100)],
        (500, 500, 3),
    )
    second = subject.update(
        "cam-a",
        0.05,
        [cat(302, 0, 402, 100), cat(2, 0, 102, 100)],
        (500, 500, 3),
    )

    assert [item.track_id for item in first] == [1, 2]
    assert [item.track_id for item in second] == [2, 1]


def test_tracker_state_is_scoped_by_source() -> None:
    subject = tracker()

    first = subject.update("cam-a", 0.0, [cat(0, 0, 20, 20)], (100, 100, 3))
    second = subject.update("cam-b", 0.0, [cat(0, 0, 20, 20)], (100, 100, 3))

    assert first[0].track_id == 1
    assert second[0].track_id == 1


def test_tracker_smooths_pose_box_with_time_based_ema() -> None:
    subject = tracker(bbox_smoothing_seconds=0.1)
    subject.update("cam", 0.0, [cat(0, 0, 100, 100)], (300, 300, 3))

    result = subject.update(
        "cam",
        0.1,
        [cat(10, 0, 110, 100)],
        (300, 300, 3),
    )[0]

    expected_shift = round(10 * (1 - math.exp(-1)))
    # Spell out the stable rounded value instead of coupling the assertion to
    # implementation floating-point details.
    assert expected_shift == 6
    assert result.raw_detection.xyxy == (10, 0, 110, 100)
    assert result.smoothed_detection.xyxy == (6, 0, 106, 100)


def test_tracker_expires_old_track_and_allocates_new_id() -> None:
    subject = tracker(max_age_seconds=0.5)
    first = subject.update("cam", 0.0, [cat(0, 0, 20, 20)], (100, 100, 3))
    second = subject.update("cam", 1.0, [cat(0, 0, 20, 20)], (100, 100, 3))

    assert first[0].track_id == 1
    assert second[0].track_id == 2


def test_tracker_does_not_guess_during_ambiguous_overlap() -> None:
    subject = tracker(ambiguity_margin=0.05)
    subject.update(
        "cam",
        0.0,
        [cat(0, 0, 100, 100), cat(120, 0, 220, 100)],
        (400, 400, 3),
    )

    result = subject.update(
        "cam",
        0.05,
        [cat(60, 0, 160, 100), cat(60, 0, 160, 100)],
        (400, 400, 3),
    )

    assert [item.track_id for item in result] == [3, 4]
    assert all(item.smoothed_detection == item.raw_detection for item in result)

    separated = subject.update(
        "cam",
        0.10,
        [cat(0, 0, 100, 100), cat(120, 0, 220, 100)],
        (400, 400, 3),
    )
    # Pre-conflict IDs are permanently retired; they cannot resurrect with a
    # stale pose history after the animals separate or cross.
    assert not {item.track_id for item in separated} & {1, 2}


def test_tracker_clears_source_and_handles_timestamp_reset() -> None:
    subject = tracker()
    subject.update("cam", 2.0, [cat(0, 0, 20, 20)], (100, 100, 3))
    reset = subject.update("cam", 0.0, [cat(0, 0, 20, 20)], (100, 100, 3))
    subject.clear_source("cam")
    cleared = subject.update("cam", 0.0, [cat(0, 0, 20, 20)], (100, 100, 3))

    assert reset[0].track_id == 1
    assert cleared[0].track_id == 1


def test_disabled_tracker_is_an_aligned_noop() -> None:
    detection = cat(0, 0, 20, 20)

    result = CatTracker().update("cam", 0.0, [detection], (100, 100, 3))

    assert result[0].track_id is None
    assert result[0].raw_detection is detection
    assert result[0].smoothed_detection is detection


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"match_iou": -0.1}, "tracking.match_iou"),
        ({"max_center_distance_ratio": 1.1}, "tracking.max_center"),
        ({"ambiguity_margin": -0.1}, "tracking.ambiguity_margin"),
        ({"max_age_seconds": 0.0}, "tracking.max_age_seconds"),
        ({"bbox_smoothing_seconds": float("nan")}, "tracking.bbox_smoothing"),
        ({"max_tracks_per_source": True}, "tracking.max_tracks"),
    ],
)
def test_tracking_config_rejects_invalid_values(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrackingConfig(**kwargs)
