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


def make_pipeline(tmp_path, pose_estimator) -> RealtimePipeline:
    return RealtimePipeline(
        detector=TwoCatDetector(),
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
