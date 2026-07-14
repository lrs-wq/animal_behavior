"""Regression tests for coat-confidence gating and safe history reuse."""

import numpy as np
import pytest

from eating_behavior.events import AnnouncementRecorder
from eating_behavior.identity import SourceIdentityService
from eating_behavior.models import CoatPrediction, Detection
from eating_behavior.pipeline import PipelineConfig, RealtimePipeline


class OneCatDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [Detection(0, "cat", 0.9, (0, 0, 20, 20))]


class FixedCoatClassifier:
    def __init__(self, prediction: CoatPrediction) -> None:
        self.prediction = prediction

    def classify(self, crop: np.ndarray) -> CoatPrediction:
        return self.prediction


class CatThenUnassociatedFeedingDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        if self.calls == 1:
            return [Detection(0, "cat", 0.9, (0, 0, 20, 20))]
        return [Detection(1, "feeding_head", 0.9, (5, 5, 10, 10))]


def make_pipeline(tmp_path, detector, prediction: CoatPrediction) -> RealtimePipeline:
    return RealtimePipeline(
        detector=detector,
        coat_classifier=FixedCoatClassifier(prediction),
        identity_service=SourceIdentityService.from_config(
            {"cats": [{"name": "blue", "coat_class": "blue_cat"}]}
        ),
        recorder=AnnouncementRecorder(tmp_path),
        config=PipelineConfig(
            cat_classes={"cat"},
            feeding_classes={"feeding_head"},
            coat_min_area=1,
            identity_history_seconds=8.0,
            pass_event_interval_seconds=5.0,
            eating_event_interval_seconds=2.0,
            bowl_monitor_interval_seconds=30.0,
            coat_min_confidence=0.55,
            coat_min_margin=0.05,
        ),
    )


@pytest.mark.parametrize(
    ("prediction", "expected_method"),
    [
        (CoatPrediction(0, "blue_cat", 0.01, "other", 0.0), "low_coat_confidence"),
        (
            CoatPrediction(0, "blue_cat", 0.60, "other", 0.58),
            "ambiguous_coat_prediction",
        ),
        (
            CoatPrediction(0, "blue_cat", 1.50, "other", -0.20),
            "invalid_coat_prediction",
        ),
    ],
)
def test_untrusted_coat_prediction_cannot_publish_or_seed_identity(
    tmp_path, prediction: CoatPrediction, expected_method: str
) -> None:
    pipeline = make_pipeline(tmp_path, OneCatDetector(), prediction)

    pipeline.process_frame("cam01", 0.0, np.zeros((30, 30, 3), dtype=np.uint8))

    passed = next(event for event in pipeline.recorder._events if event.event_type == "cat_passed")
    assert passed.cat_name == ""
    assert passed.details["identity_method"] == expected_method
    assert pipeline.identity_timeline.observations == []


def test_accepted_coat_confidence_is_preserved_as_identity_confidence(tmp_path) -> None:
    prediction = CoatPrediction(0, "blue_cat", 0.80, "other", 0.10)
    pipeline = make_pipeline(tmp_path, OneCatDetector(), prediction)

    pipeline.process_frame("cam01", 0.0, np.zeros((30, 30, 3), dtype=np.uint8))

    passed = next(event for event in pipeline.recorder._events if event.event_type == "cat_passed")
    assert passed.cat_name == "blue"
    assert passed.confidence == pytest.approx(0.80)
    assert pipeline.identity_timeline.observations[0].confidence == pytest.approx(0.80)


def test_unassociated_feeding_does_not_inherit_arbitrary_history(tmp_path) -> None:
    prediction = CoatPrediction(0, "blue_cat", 0.90, "other", 0.10)
    pipeline = make_pipeline(tmp_path, CatThenUnassociatedFeedingDetector(), prediction)
    frame = np.zeros((30, 30, 3), dtype=np.uint8)

    pipeline.process_frame("cam01", 0.0, frame)
    pipeline.process_frame("cam01", 1.0, frame)
    pipeline.process_frame("cam01", 3.0, frame)
    pipeline.finish_source("cam01", 3.0, "eof")

    eating = next(event for event in pipeline.recorder._events if event.event_type == "cat_eating")
    assert eating.cat_name == ""
    assert eating.confidence == 0.0
    assert eating.details["accepted_identity_observation_count"] == 0
    assert eating.details["identity_rejected_reason"] == "no_accepted_identity"
