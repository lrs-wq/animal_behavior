from types import SimpleNamespace

import numpy as np
import pytest

import eating_behavior.visualization as visualization_module
from eating_behavior.identity import IdentityDecision
from eating_behavior.models import CoatPrediction, Detection
from eating_behavior.pose import PoseKeypoint, PosePrediction
from tools.run_batch_test import (
    PoseStabilityDiagnostics,
    accumulate_pose_frame,
    batch_pose_diagnostics,
    cat_pose_label,
    render_frame,
)


def make_pose(
    x: float,
    confidence: float = 0.9,
    bbox: tuple[int, int, int, int] = (0, 0, 100, 100),
) -> PosePrediction:
    point = PoseKeypoint(0, "left_eye", x, float(bbox[1] + 20), confidence)
    return PosePrediction(bbox, (point,), confidence)


def empty_result() -> dict:
    return {
        "pose_frames": 0,
        "pose_instances": 0,
        "pose_visible_keypoints": 0,
        "raw_pose_frames": 0,
        "raw_pose_instances": 0,
        "raw_pose_visible_keypoints": 0,
        "filtered_pose_frames": 0,
        "filtered_pose_instances": 0,
        "filtered_pose_visible_keypoints": 0,
        "pose_status_counts": {},
        "pose_rejection_codes": {},
        "posture_observation_counts": {},
        "posture_state_counts": {},
        "pose_held_keypoints": 0,
        "pose_dropped_keypoints": 0,
        "tracked_cat_instances": 0,
        "untracked_cat_instances": 0,
        "track_ids": [],
    }


def test_stability_compares_bbox_local_raw_and_filtered_steps() -> None:
    diagnostics = PoseStabilityDiagnostics(0.3)
    diagnostics.observe(
        source_id="camera",
        track_id=7,
        raw_pose=make_pose(10),
        filtered_pose=make_pose(10),
    )
    diagnostics.observe(
        source_id="camera",
        track_id=7,
        raw_pose=make_pose(40),
        filtered_pose=make_pose(15),
    )

    summary = diagnostics.summary()
    assert summary["coordinate_space"] == "bbox_local_xy"
    assert summary["raw"]["step_count"] == 1
    assert summary["raw"]["p95_local_step"] == pytest.approx(0.3)
    assert summary["raw"]["large_step_count"] == 1
    assert summary["filtered"]["p95_local_step"] == pytest.approx(0.05)
    assert summary["filtered"]["large_step_count"] == 0


def test_missing_pose_breaks_stability_continuity() -> None:
    diagnostics = PoseStabilityDiagnostics(0.3)
    diagnostics.observe(
        source_id="camera",
        track_id=2,
        raw_pose=make_pose(10),
        filtered_pose=make_pose(10),
    )
    diagnostics.observe(
        source_id="camera",
        track_id=2,
        raw_pose=None,
        filtered_pose=None,
    )
    diagnostics.observe(
        source_id="camera",
        track_id=2,
        raw_pose=make_pose(90),
        filtered_pose=make_pose(90),
    )

    summary = diagnostics.summary()
    assert summary["raw"]["step_count"] == 0
    assert summary["raw"]["p95_local_step"] is None
    assert summary["filtered"]["step_count"] == 0


def test_stability_does_not_count_whole_bbox_translation_as_jitter() -> None:
    diagnostics = PoseStabilityDiagnostics(0.3)
    diagnostics.observe(
        source_id="camera",
        track_id=2,
        raw_pose=make_pose(10),
        filtered_pose=make_pose(10),
    )
    translated = make_pose(60, bbox=(50, 50, 150, 150))
    diagnostics.observe(
        source_id="camera",
        track_id=2,
        raw_pose=translated,
        filtered_pose=translated,
    )

    summary = diagnostics.summary()
    assert summary["raw"]["p95_local_step"] == 0.0
    assert summary["filtered"]["p95_local_step"] == 0.0


def test_accumulate_pose_frame_preserves_legacy_filtered_counters() -> None:
    detection = Detection(0, "cat", 0.9, (0, 0, 100, 100))
    cat = SimpleNamespace(
        detection=detection,
        track_id=12,
        raw_pose=make_pose(30),
        pose=make_pose(20),
        pose_status="held",
        pose_rejection_code="geometry",
        pose_held_keypoints=2,
        pose_dropped_keypoints=3,
        observed_posture="standing",
        posture="crouching",
    )
    result = empty_result()
    diagnostics = PoseStabilityDiagnostics(0.3)
    track_ids: set[int] = set()

    labels = accumulate_pose_frame(
        result,
        (cat,),
        source_id="camera",
        visibility_threshold=0.3,
        stability=diagnostics,
        track_ids=track_ids,
    )

    assert result["raw_pose_frames"] == 1
    assert result["raw_pose_instances"] == 1
    assert result["filtered_pose_frames"] == 1
    assert result["filtered_pose_instances"] == 1
    assert result["pose_frames"] == result["filtered_pose_frames"]
    assert result["pose_instances"] == result["filtered_pose_instances"]
    assert result["pose_visible_keypoints"] == 1
    assert result["pose_status_counts"] == {"held": 1}
    assert result["pose_rejection_codes"] == {"geometry": 1}
    assert result["pose_held_keypoints"] == 2
    assert result["pose_dropped_keypoints"] == 3
    assert result["posture_observation_counts"] == {"standing": 1}
    assert result["posture_state_counts"] == {"crouching": 1}
    assert result["tracked_cat_instances"] == 1
    assert track_ids == {12}
    assert labels[detection] == "posture=crouching"


def test_old_cat_analysis_shape_falls_back_to_filtered_pose() -> None:
    detection = Detection(0, "cat", 0.9, (0, 0, 100, 100))
    old_cat = SimpleNamespace(detection=detection, pose=make_pose(10), track_id=None)
    result = empty_result()

    label = cat_pose_label(old_cat, 0.3)
    accumulate_pose_frame(
        result,
        (old_cat,),
        source_id="camera",
        visibility_threshold=0.3,
        stability=PoseStabilityDiagnostics(0.3),
    )

    assert label == "posture=unknown"
    assert result["raw_pose_instances"] == 1
    assert result["filtered_pose_instances"] == 1
    assert result["untracked_cat_instances"] == 1


def test_render_frame_only_labels_classification_and_posture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cat = Detection(0, "cat", 0.87, (20, 40, 180, 140))
    feeding = Detection(1, "feeding_head", 0.82, (60, 60, 90, 90))
    bowl = Detection(2, "bowl", 0.71, (200, 40, 280, 120))
    coat = CoatPrediction(0, "blue_cat", 0.93, "orange_cat", 0.04)
    identity = IdentityDecision(
        "mimi",
        "direct_by_single_same_coat",
        0.99,
        "test",
        ("mimi",),
    )
    rendered_text: list[str] = []
    monkeypatch.setattr(
        visualization_module,
        "draw_text",
        lambda _frame, text, _origin, _color: rendered_text.append(text),
    )

    render_frame(
        np.zeros((160, 320, 3), dtype=np.uint8),
        (cat, feeding, bowl),
        ((cat, coat, identity),),
        (),
        source_id="camera",
        frame_index=1,
        timestamp_seconds=0.0,
        eating_state="idle",
        cat_classes={"cat"},
        feeding_classes={"feeding_head"},
        association_max_distance_ratio=1.25,
        association_ambiguity_margin=0.1,
        pose_keypoint_confidence=0.3,
        cat_pose_labels={cat: "posture=crouching"},
    )

    assert rendered_text == [
        "coat=blue_cat posture=crouching",
        "feeding_head",
        "bowl",
    ]


def test_batch_pose_diagnostics_merges_per_video_counters() -> None:
    first = empty_result()
    second = empty_result()
    first.update(
        raw_pose_instances=2,
        filtered_pose_instances=1,
        pose_status_counts={"accepted": 1, "rejected": 1},
        pose_rejection_codes={"geometry": 1},
        posture_observation_counts={"standing": 1},
        posture_state_counts={"crouching": 1},
        track_ids=[1, 2],
    )
    second.update(
        raw_pose_instances=3,
        filtered_pose_instances=2,
        pose_status_counts={"accepted": 2},
        pose_rejection_codes={},
        posture_observation_counts={"standing": 2},
        posture_state_counts={"standing": 2},
        track_ids=[1],
    )

    summary = batch_pose_diagnostics(
        (first, second), PoseStabilityDiagnostics(0.3)
    )

    assert summary["raw_pose_instances"] == 5
    assert summary["filtered_pose_instances"] == 3
    assert summary["unique_source_track_count"] == 3
    assert summary["pose_status_counts"] == {"accepted": 3, "rejected": 1}
    assert summary["pose_rejection_codes"] == {"geometry": 1}
    assert summary["posture_observation_counts"] == {"standing": 3}
    assert summary["posture_state_counts"] == {"crouching": 1, "standing": 2}
