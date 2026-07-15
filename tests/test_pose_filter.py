"""Quality gates and track-aware pose filtering tests."""

from __future__ import annotations

import pytest

from eating_behavior.models import Detection
from eating_behavior.pose import AP10K_KEYPOINT_NAMES, PoseKeypoint, PosePrediction
from eating_behavior.pose_filter import PoseFilterConfig, TrackPoseFilter


BASE_POINTS = (
    (45.0, 20.0),
    (55.0, 20.0),
    (50.0, 28.0),
    (50.0, 40.0),
    (50.0, 75.0),
    (40.0, 42.0),
    (32.0, 55.0),
    (25.0, 70.0),
    (60.0, 42.0),
    (68.0, 55.0),
    (75.0, 70.0),
    (42.0, 72.0),
    (38.0, 82.0),
    (35.0, 93.0),
    (58.0, 72.0),
    (62.0, 82.0),
    (65.0, 93.0),
)


def cat(box: tuple[int, int, int, int] = (20, 20, 120, 120)) -> Detection:
    return Detection(0, "cat", 0.9, box)


def prediction(
    *,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    confidence: float = 0.9,
    overrides: dict[int, tuple[float, float, float]] | None = None,
    box: tuple[int, int, int, int] = (20, 20, 120, 120),
) -> PosePrediction:
    overrides = overrides or {}
    points = []
    for index, (x, y) in enumerate(BASE_POINTS):
        x, y, score = overrides.get(
            index,
            (x + 20.0 + shift_x, y + 20.0 + shift_y, confidence),
        )
        points.append(
            PoseKeypoint(index, AP10K_KEYPOINT_NAMES[index], x, y, score)
        )
    return PosePrediction(box, tuple(points), confidence)


def subject(**overrides) -> TrackPoseFilter:
    return TrackPoseFilter(PoseFilterConfig(enabled=True, **overrides))


def test_filter_accepts_connected_high_quality_pose() -> None:
    result = subject().update(
        "cam",
        0.0,
        1,
        cat(),
        prediction(),
        (200, 200, 3),
    )

    assert result.status == "accepted"
    assert result.prediction is not None
    assert result.prediction.visible_keypoint_count(0.5) == 17


def test_filter_rejects_isolated_high_confidence_point() -> None:
    low = {
        index: (x + 20.0, y + 20.0, 0.1)
        for index, (x, y) in enumerate(BASE_POINTS)
    }
    low[2] = (70.0, 48.0, 0.99)

    result = subject().update(
        "cam",
        0.0,
        1,
        cat(),
        prediction(overrides=low),
        (200, 200, 3),
    )

    assert result.prediction is None
    assert result.rejection_code == "insufficient_reliable_keypoints"


def test_filter_rejects_collapsed_high_confidence_skeleton() -> None:
    collapsed = {
        index: (70.0, 70.0, 0.99)
        for index in range(len(AP10K_KEYPOINT_NAMES))
    }

    result = subject().update(
        "cam",
        0.0,
        1,
        cat(),
        prediction(overrides=collapsed),
        (200, 200, 3),
    )

    assert result.prediction is None
    assert result.rejection_code == "degenerate_pose_geometry"


def test_new_track_does_not_sum_disconnected_limb_fragments() -> None:
    fragmented = {
        index: (x + 20.0, y + 20.0, 0.1)
        for index, (x, y) in enumerate(BASE_POINTS)
    }
    for index in (5, 6, 11, 12, 13):
        x, y = BASE_POINTS[index]
        fragmented[index] = (x + 20.0, y + 20.0, 0.9)

    result = subject().update(
        "cam",
        0.0,
        1,
        cat(),
        prediction(overrides=fragmented),
        (200, 200, 3),
    )

    assert result.prediction is None
    assert result.rejection_code == "insufficient_reliable_keypoints"


def test_filter_drops_point_outside_its_expanded_bbox() -> None:
    result = subject().update(
        "cam",
        0.0,
        1,
        cat(),
        prediction(overrides={7: (150.0, 90.0, 0.99)}),
        (200, 200, 3),
    )

    assert result.prediction is not None
    assert result.prediction.keypoints[7].confidence == 0.0
    assert result.dropped_keypoints >= 1


def test_one_euro_filter_reduces_small_frame_to_frame_jitter() -> None:
    filter_ = subject(one_euro_beta=0.0)
    first = filter_.update("cam", 0.0, 1, cat(), prediction(), (200, 200, 3))
    second = filter_.update(
        "cam",
        0.05,
        1,
        cat(),
        prediction(shift_x=5.0),
        (200, 200, 3),
    )

    assert first.prediction is not None
    assert second.prediction is not None
    previous_x = first.prediction.keypoints[3].x
    filtered_x = second.prediction.keypoints[3].x
    raw_x = prediction(shift_x=5.0).keypoints[3].x
    assert previous_x < filtered_x < raw_x


def test_bbox_translation_does_not_trigger_false_keypoint_motion() -> None:
    filter_ = subject(one_euro_beta=0.0)
    first = filter_.update("cam", 0.0, 1, cat(), prediction(), (250, 250, 3))
    moved_box = (30, 20, 130, 120)
    second = filter_.update(
        "cam",
        0.05,
        1,
        cat(moved_box),
        prediction(shift_x=10.0, box=moved_box),
        (250, 250, 3),
    )

    assert first.prediction is not None
    assert second.prediction is not None
    assert second.prediction.keypoints[3].x == pytest.approx(
        first.prediction.keypoints[3].x + 10.0
    )


def test_bbox_quality_floor_never_warps_a_held_pose() -> None:
    filter_ = subject()
    accepted = filter_.update("cam", 0.0, 1, cat(), prediction(), (200, 200, 3))
    assert accepted.prediction is not None

    # This is a gradual same-centre shrink, below the keep-area floor but not
    # large enough to rely on the independent bbox-shock reset.
    below_floor = cat((32, 32, 107, 107))
    result = filter_.update("cam", 0.05, 1, below_floor, None, (200, 200, 3))

    assert result.prediction is None
    assert result.rejection_code == "pose_bbox_below_quality_floor"


def test_bbox_size_shock_resets_even_without_low_iou() -> None:
    filter_ = subject()
    accepted = filter_.update("cam", 0.0, 1, cat(), prediction(), (300, 300, 3))
    assert accepted.prediction is not None

    # Width changes by >2x while IoU remains above the old 0.30 reset gate.
    shocked = cat((20, 20, 230, 120))
    result = filter_.update("cam", 0.05, 1, shocked, None, (300, 300, 3))

    assert result.prediction is None
    assert result.rejection_code == "missing_pose"


def test_bbox_relative_hold_never_emits_visible_off_frame_points() -> None:
    filter_ = subject()
    edge_pose = prediction(overrides={7: (0.0, 90.0, 0.9)})
    accepted = filter_.update("cam", 0.0, 1, cat(), edge_pose, (200, 200, 3))
    assert accepted.prediction is not None
    assert accepted.prediction.keypoints[7].confidence > 0.0

    moved = filter_.update(
        "cam",
        0.05,
        1,
        cat((0, 20, 100, 120)),
        None,
        (200, 200, 3),
    )

    assert moved.prediction is not None
    paw = moved.prediction.keypoints[7]
    assert paw.x == 0.0
    assert paw.confidence == 0.0
    assert all(
        0.0 <= point.x <= 199.0 and 0.0 <= point.y <= 199.0
        for point in moved.prediction.keypoints
    )


def test_hard_motion_gate_holds_even_high_confidence_jump() -> None:
    filter_ = subject()
    first = filter_.update("cam", 0.0, 1, cat(), prediction(), (250, 250, 3))
    jumped = filter_.update(
        "cam",
        0.05,
        1,
        cat(),
        prediction(shift_x=40.0),
        (250, 250, 3),
    )

    assert first.prediction is not None
    assert jumped.prediction is not None
    assert jumped.status == "held"
    assert jumped.prediction.keypoints[3].x == first.prediction.keypoints[3].x


def test_motion_gate_uses_each_points_last_valid_timestamp() -> None:
    filter_ = subject()
    first = filter_.update("cam", 0.0, 1, cat(), prediction(), (200, 200, 3))
    assert first.prediction is not None

    low_neck = {3: (70.0, 60.0, 0.1)}
    filter_.update(
        "cam",
        0.05,
        1,
        cat(),
        prediction(overrides=low_neck),
        (200, 200, 3),
    )
    filter_.update(
        "cam",
        0.10,
        1,
        cat(),
        prediction(overrides=low_neck),
        (200, 200, 3),
    )
    returned = filter_.update(
        "cam",
        0.15,
        1,
        cat(),
        prediction(overrides={3: (100.0, 60.0, 0.9)}),
        (200, 200, 3),
    )

    assert returned.prediction is not None
    assert returned.status == "accepted"
    assert returned.prediction.keypoints[3].x > first.prediction.keypoints[3].x


def test_hold_expires_instead_of_rendering_stale_pose_forever() -> None:
    filter_ = subject()
    accepted = filter_.update("cam", 0.0, 1, cat(), prediction(), (200, 200, 3))
    assert accepted.prediction is not None

    results = [
        filter_.update("cam", index * 0.05, 1, cat(), None, (200, 200, 3))
        for index in range(1, 5)
    ]

    assert results[0].status == "held"
    assert results[-1].prediction is None


def test_bridge_stops_output_before_reset_discards_filter_state() -> None:
    filter_ = subject(one_euro_beta=0.0, core_hold_frames=10)
    first = filter_.update("cam", 0.0, 1, cat(), prediction(), (200, 200, 3))
    assert first.prediction is not None

    bridged = filter_.update("cam", 0.10, 1, cat(), None, (200, 200, 3))
    beyond_bridge = filter_.update("cam", 0.20, 1, cat(), None, (200, 200, 3))
    recovered = filter_.update(
        "cam",
        0.25,
        1,
        cat(),
        prediction(shift_x=5.0),
        (200, 200, 3),
    )

    assert bridged.status == "held"
    assert beyond_bridge.prediction is None
    assert recovered.prediction is not None
    assert first.prediction.keypoints[3].x < recovered.prediction.keypoints[3].x
    assert recovered.prediction.keypoints[3].x < prediction(shift_x=5.0).keypoints[3].x

    reset_filter = subject(one_euro_beta=0.0, core_hold_frames=10)
    reset_filter.update("cam", 0.0, 1, cat(), prediction(), (200, 200, 3))
    reset = reset_filter.update(
        "cam",
        0.31,
        1,
        cat(),
        prediction(shift_x=5.0),
        (200, 200, 3),
    )
    assert reset.prediction is not None
    assert reset.prediction.keypoints[3].x == prediction(shift_x=5.0).keypoints[3].x


def test_repeated_missing_frames_cannot_keep_weak_entry_state_alive() -> None:
    filter_ = subject(core_hold_frames=10, joint_hold_frames=10, paw_hold_frames=10)
    accepted = filter_.update("cam", 0.0, 1, cat(), prediction(), (200, 200, 3))
    assert accepted.prediction is not None

    for index in range(1, 21):
        filter_.update("cam", index * 0.05, 1, cat(), None, (200, 200, 3))
    low = {
        index: (x + 20.0, y + 20.0, 0.1)
        for index, (x, y) in enumerate(BASE_POINTS)
    }
    for index in (0, 1, 2):
        x, y = BASE_POINTS[index]
        low[index] = (x + 20.0, y + 20.0, 0.9)
    result = filter_.update(
        "cam",
        1.05,
        1,
        cat(),
        prediction(overrides=low),
        (200, 200, 3),
    )

    assert result.prediction is None
    assert result.rejection_code == "insufficient_reliable_keypoints"


def test_one_bilateral_pair_cannot_swap_an_entire_limb_chain() -> None:
    filter_ = subject(one_euro_beta=0.0)
    first = filter_.update("cam", 0.0, 1, cat(), prediction(), (200, 200, 3))
    assert first.prediction is not None

    one_pair = {
        5: (80.0, 62.0, 0.9),
        8: (60.0, 62.0, 0.9),
        6: (52.0, 75.0, 0.1),
        7: (45.0, 90.0, 0.1),
        9: (88.0, 75.0, 0.1),
        10: (95.0, 90.0, 0.1),
    }
    result = filter_.update(
        "cam",
        0.05,
        1,
        cat(),
        prediction(overrides=one_pair),
        (200, 200, 3),
    )

    assert result.prediction is not None
    assert result.prediction.keypoints[5].x > first.prediction.keypoints[5].x
    assert result.prediction.keypoints[8].x < first.prediction.keypoints[8].x


def test_two_bilateral_pairs_can_correct_a_consistent_limb_swap() -> None:
    filter_ = subject(one_euro_beta=0.0)
    first = filter_.update("cam", 0.0, 1, cat(), prediction(), (200, 200, 3))
    assert first.prediction is not None

    two_pairs = {
        5: (80.0, 62.0, 0.9),
        8: (60.0, 62.0, 0.9),
        6: (88.0, 75.0, 0.9),
        9: (52.0, 75.0, 0.9),
        7: (45.0, 90.0, 0.1),
        10: (95.0, 90.0, 0.1),
    }
    result = filter_.update(
        "cam",
        0.05,
        1,
        cat(),
        prediction(overrides=two_pairs),
        (200, 200, 3),
    )

    assert result.prediction is not None
    for index in (5, 6, 8, 9):
        assert result.prediction.keypoints[index].x == pytest.approx(
            first.prediction.keypoints[index].x
        )


def test_filter_state_isolated_and_cleared_by_source() -> None:
    filter_ = subject()
    filter_.update("cam-a", 0.0, 1, cat(), prediction(), (200, 200, 3))

    other = filter_.update(
        "cam-b",
        0.05,
        1,
        cat(),
        prediction(shift_x=5.0),
        (200, 200, 3),
    )
    filter_.clear_source("cam-a")
    cleared = filter_.update(
        "cam-a",
        0.0,
        1,
        cat(),
        prediction(shift_x=5.0),
        (200, 200, 3),
    )

    assert other.prediction.keypoints[3].x == prediction(shift_x=5.0).keypoints[3].x
    assert cleared.prediction.keypoints[3].x == prediction(shift_x=5.0).keypoints[3].x


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"min_bbox_area_keep": 9000}, "min_bbox_area_keep"),
        ({"point_keep_confidence": 0.6}, "point_keep_confidence"),
        ({"max_bridge_gap_seconds": 0.4}, "max_bridge_gap_seconds"),
        ({"one_euro_beta": -0.1}, "one_euro_beta"),
        ({"core_hold_frames": True}, "core_hold_frames"),
        ({"point_enter_confidence": True}, "point_enter_confidence"),
        ({"point_enter_confidence": "0.5"}, "point_enter_confidence"),
        ({"max_bridge_gap_seconds": True}, "max_bridge_gap_seconds"),
        ({"one_euro_beta": True}, "one_euro_beta"),
        ({"enter_min_points": 18}, "enter_min_points"),
        ({"enter_min_edges": 18}, "enter_min_edges"),
        ({"enter_min_anchors": 7}, "enter_min_anchors"),
        ({"keep_min_points": 6}, "keep_min_points"),
        ({"keep_min_edges": 4}, "keep_min_edges"),
    ],
)
def test_pose_filter_config_rejects_invalid_values(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PoseFilterConfig(**kwargs)
