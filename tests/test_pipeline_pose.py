"""Pipeline integration tests for aligned RTMPose results."""

from __future__ import annotations

import numpy as np
import pytest

from eating_behavior.events import AnnouncementRecorder
from eating_behavior.identity import SourceIdentityService
from eating_behavior.models import CoatPrediction, Detection
from eating_behavior.pipeline import PipelineConfig, RealtimePipeline
from eating_behavior.pose import (
    AP10K_KEYPOINT_NAMES,
    PoseInferenceError,
    PoseKeypoint,
    PosePrediction,
)
from eating_behavior.pose_filter import PoseFilterResult
from eating_behavior.tracking import CatTracker, TrackingConfig


class TwoCatDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        del frame
        return [
            Detection(0, "cat", 0.9, (0, 0, 20, 20)),
            Detection(0, "cat", 0.8, (20, 20, 40, 40)),
        ]


class FixedCoatClassifier:
    def classify(self, crop: np.ndarray) -> CoatPrediction:
        del crop
        return CoatPrediction(0, "blue_cat", 0.9, "other", 0.1)


def prediction(box: tuple[int, int, int, int]) -> PosePrediction:
    points = tuple(
        PoseKeypoint(index, name, float(index), float(index), 0.9)
        for index, name in enumerate(AP10K_KEYPOINT_NAMES)
    )
    return PosePrediction(box, points, 0.9)


class RecordingPoseEstimator:
    def __init__(self) -> None:
        self.calls: list[tuple[Detection, ...]] = []

    def predict(self, frame, cat_detections):
        del frame
        self.calls.append(tuple(cat_detections))
        return tuple(prediction(item.xyxy) for item in cat_detections)


class FailingPoseEstimator:
    def predict(self, frame, cat_detections):
        del frame, cat_detections
        raise PoseInferenceError("pose_runtime_failure")


def make_pipeline(
    tmp_path,
    pose_estimator,
    *,
    detector=None,
    cat_tracker=None,
    pose_filter=None,
) -> RealtimePipeline:
    return RealtimePipeline(
        detector=detector or TwoCatDetector(),
        coat_classifier=FixedCoatClassifier(),
        identity_service=SourceIdentityService.from_config(
            {
                "cats": [
                    {"name": "blue", "coat_class": "blue_cat"},
                    {"name": "other-blue", "coat_class": "blue_cat"},
                ]
            }
        ),
        recorder=AnnouncementRecorder(tmp_path),
        config=PipelineConfig(
            cat_classes={"cat"},
            feeding_classes={"feeding_head"},
            coat_min_area=0,
            identity_history_seconds=5.0,
            pass_event_interval_seconds=5.0,
            eating_event_interval_seconds=2.0,
            bowl_monitor_interval_seconds=30.0,
        ),
        pose_estimator=pose_estimator,
        cat_tracker=cat_tracker,
        pose_filter=pose_filter,
    )


def test_pipeline_runs_pose_once_and_returns_reusable_frame_analysis(tmp_path) -> None:
    pose = RecordingPoseEstimator()
    pipeline = make_pipeline(tmp_path, pose)

    analysis = pipeline.process_frame(
        "cam01",
        0.0,
        np.zeros((40, 40, 3), dtype=np.uint8),
    )

    assert len(pose.calls) == 1
    assert len(pose.calls[0]) == 2
    assert len(analysis.detections) == 2
    assert len(analysis.cats) == 2
    assert len(analysis.cat_decisions) == 2
    assert len(analysis.poses) == 2
    assert analysis.cats[1].pose.cat_xyxy == (20, 20, 40, 40)
    assert analysis.pose_error_code == ""


def test_pose_frame_error_does_not_change_existing_state_chain(
    tmp_path, caplog
) -> None:
    pipeline = make_pipeline(tmp_path, FailingPoseEstimator())

    analysis = pipeline.process_frame(
        "cam01",
        0.0,
        np.zeros((40, 40, 3), dtype=np.uint8),
    )

    assert analysis.pose_error_code == "pose_runtime_failure"
    assert analysis.poses == ()
    assert analysis.eating_state == "idle"
    assert "Pose inference degraded" in caplog.text
    # Existing throttling collapses the two unresolved same-coat observations.
    assert (
        len(
            [
                event
                for event in pipeline.recorder._events
                if event.event_type == "cat_passed"
            ]
        )
        == 1
    )


def test_pipeline_rejects_unaligned_pose_adapter(tmp_path) -> None:
    class BrokenPoseEstimator:
        def predict(self, frame, cat_detections):
            del frame, cat_detections
            return (None,)

    pipeline = make_pipeline(tmp_path, BrokenPoseEstimator())
    with pytest.raises(RuntimeError, match="result count"):
        pipeline.process_frame(
            "cam01",
            0.0,
            np.zeros((40, 40, 3), dtype=np.uint8),
        )


def test_pipeline_uses_smoothed_pose_box_but_keeps_raw_event_box(tmp_path) -> None:
    class MovingDetector:
        def __init__(self) -> None:
            self.index = 0

        def detect(self, frame):
            del frame
            boxes = [(0, 0, 20, 20), (4, 0, 24, 20), (4, 0, 24, 20)]
            box = boxes[self.index]
            self.index += 1
            return [Detection(0, "cat", 0.9, box)]

    pose = RecordingPoseEstimator()
    pipeline = make_pipeline(
        tmp_path,
        pose,
        detector=MovingDetector(),
        cat_tracker=CatTracker(
            TrackingConfig(enabled=True, bbox_smoothing_seconds=0.1)
        ),
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    first = pipeline.process_frame("cam01", 0.0, frame)
    second = pipeline.process_frame("cam01", 0.1, frame)

    assert first.cats[0].track_id == second.cats[0].track_id == 1
    assert second.cats[0].detection.xyxy == (4, 0, 24, 20)
    assert second.cats[0].pose_detection.xyxy == (3, 0, 23, 20)
    assert second.cats[0].pose.cat_xyxy == (3, 0, 23, 20)
    assert pose.calls[1][0].xyxy == (3, 0, 23, 20)

    pipeline.finish_source("cam01", 0.1, "eof")
    new_session = pipeline.process_frame("cam01", 0.0, frame)
    assert new_session.cats[0].track_id == 1
    assert new_session.cats[0].pose_detection.xyxy == (4, 0, 24, 20)


def test_pipeline_keeps_raw_pose_and_exposes_filter_diagnostics(tmp_path) -> None:
    class RecordingFilter:
        def __init__(self) -> None:
            self.calls = []
            self.cleared = []

        def update(
            self,
            source,
            timestamp_seconds,
            track_id,
            detection,
            raw_pose,
            frame_shape,
        ):
            self.calls.append(
                (
                    source,
                    timestamp_seconds,
                    track_id,
                    detection,
                    raw_pose,
                    frame_shape,
                )
            )
            return PoseFilterResult(
                prediction=None,
                status="rejected",
                rejection_code="test_gate",
                held_keypoints=2,
                dropped_keypoints=15,
            )

        def clear_source(self, source):
            self.cleared.append(source)

    pose_filter = RecordingFilter()
    pipeline = make_pipeline(
        tmp_path,
        RecordingPoseEstimator(),
        pose_filter=pose_filter,
    )

    analysis = pipeline.process_frame(
        "cam01",
        0.0,
        np.zeros((40, 40, 3), dtype=np.uint8),
    )

    assert len(pose_filter.calls) == 2
    assert analysis.poses == ()
    assert analysis.cats[0].raw_pose is not None
    assert analysis.cats[0].pose is None
    assert analysis.cats[0].pose_status == "rejected"
    assert analysis.cats[0].pose_rejection_code == "test_gate"
    assert analysis.cats[0].pose_held_keypoints == 2
    assert analysis.cats[0].pose_dropped_keypoints == 15

    pipeline.finish_source("cam01", 0.0, "eof")
    assert pose_filter.cleared == ["cam01"]


def test_pose_filter_failure_does_not_block_identity_or_eating_state(tmp_path) -> None:
    class FailingFilter:
        def __init__(self) -> None:
            self.cleared = []

        def update(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("filter bug")

        def clear_source(self, source):
            self.cleared.append(source)

    pose_filter = FailingFilter()
    pipeline = make_pipeline(
        tmp_path,
        RecordingPoseEstimator(),
        pose_filter=pose_filter,
    )

    analysis = pipeline.process_frame(
        "cam01",
        0.0,
        np.zeros((40, 40, 3), dtype=np.uint8),
    )

    assert analysis.pose_error_code == "pose_filter_failure"
    assert analysis.eating_state == "idle"
    assert "cam01" in pipeline.eating_machines
    assert len(analysis.cats) == 2
    assert all(cat.raw_pose is not None for cat in analysis.cats)
    assert all(cat.pose is None for cat in analysis.cats)
    assert all(cat.pose_rejection_code == "pose_filter_failure" for cat in analysis.cats)
    assert pose_filter.cleared == ["cam01"]
    assert any(event.event_type == "cat_passed" for event in pipeline.recorder._events)


def test_pose_tracker_failure_falls_back_to_raw_boxes(tmp_path) -> None:
    class FailingTracker:
        def __init__(self) -> None:
            self.cleared = []

        def update(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("tracker bug")

        def clear_source(self, source):
            self.cleared.append(source)

    class PassthroughFilter:
        def __init__(self) -> None:
            self.cleared = []

        def update(
            self,
            source,
            timestamp_seconds,
            track_id,
            detection,
            raw_pose,
            frame_shape,
        ):
            del source, timestamp_seconds, track_id, detection, frame_shape
            return PoseFilterResult(raw_pose, "raw")

        def clear_source(self, source):
            self.cleared.append(source)

    tracker = FailingTracker()
    pose_filter = PassthroughFilter()
    pose = RecordingPoseEstimator()
    pipeline = make_pipeline(
        tmp_path,
        pose,
        cat_tracker=tracker,
        pose_filter=pose_filter,
    )

    analysis = pipeline.process_frame(
        "cam01",
        0.0,
        np.zeros((40, 40, 3), dtype=np.uint8),
    )

    assert analysis.pose_error_code == "pose_tracking_failure"
    assert analysis.eating_state == "idle"
    assert "cam01" in pipeline.eating_machines
    assert [item.xyxy for item in pose.calls[0]] == [(0, 0, 20, 20), (20, 20, 40, 40)]
    assert all(cat.track_id is None for cat in analysis.cats)
    assert all(cat.pose_detection == cat.detection for cat in analysis.cats)
    assert tracker.cleared == ["cam01"]
    assert pose_filter.cleared == ["cam01"]


def test_combined_tracker_and_pose_failures_are_counted_independently(tmp_path) -> None:
    class FailingTracker:
        def update(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("tracker bug")

        def clear_source(self, source):
            del source

    pipeline = make_pipeline(
        tmp_path,
        FailingPoseEstimator(),
        cat_tracker=FailingTracker(),
    )

    analysis = pipeline.process_frame(
        "cam01",
        0.0,
        np.zeros((40, 40, 3), dtype=np.uint8),
    )

    assert analysis.pose_error_code == "pose_tracking_failure"
    assert pipeline.pose_error_counts[("cam01", "pose_tracking_failure")] == 1
    assert pipeline.pose_error_counts[("cam01", "pose_runtime_failure")] == 1
