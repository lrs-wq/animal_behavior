import numpy as np

from eating_behavior.events import AnnouncementRecorder
from eating_behavior.identity import SourceIdentityService
from eating_behavior.models import CoatPrediction, Detection
from eating_behavior.pipeline import PipelineConfig, RealtimePipeline


class FakeDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [Detection(class_id=0, class_name="cat", confidence=0.9, xyxy=(0, 0, 20, 20))]


class FakeCoatClassifier:
    def classify(self, crop: np.ndarray) -> CoatPrediction:
        return CoatPrediction(
            class_id=0,
            class_name="blue_cat",
            confidence=0.95,
            second_class_name="unknown",
            second_confidence=0.0,
        )


def make_pipeline(tmp_path) -> RealtimePipeline:
    identity_service = SourceIdentityService.from_config(
        {
            "households": [
                {
                    "household_id": "customer_a_home",
                    "source_ids": ["cam_customer_a_01"],
                    "cats": [{"name": "blue", "coat_class": "blue_cat"}],
                },
                {
                    "household_id": "customer_b_home",
                    "source_ids": ["cam_customer_b_01"],
                    "cats": [{"name": "blue", "coat_class": "blue_cat"}],
                },
            ]
        }
    )
    return RealtimePipeline(
        detector=FakeDetector(),
        coat_classifier=FakeCoatClassifier(),
        identity_service=identity_service,
        recorder=AnnouncementRecorder(tmp_path),
        config=PipelineConfig(
            cat_classes={"cat"},
            feeding_classes={"feeding_head"},
            coat_min_area=1,
            identity_history_seconds=8.0,
            pass_event_interval_seconds=5.0,
            eating_event_interval_seconds=2.0,
            bowl_monitor_interval_seconds=30.0,
        ),
    )


def test_pass_event_throttling_is_isolated_by_source(tmp_path) -> None:
    pipeline = make_pipeline(tmp_path)
    frame = np.zeros((30, 30, 3), dtype=np.uint8)

    pipeline.process_frame("cam_customer_a_01", 0.0, frame)
    pipeline.process_frame("cam_customer_b_01", 1.0, frame)

    pass_events = [event for event in pipeline.recorder._events if event.event_type == "cat_passed"]
    assert len(pass_events) == 2
    assert {event.source for event in pass_events} == {"cam_customer_a_01", "cam_customer_b_01"}
    assert {event.household_id for event in pass_events} == {"customer_a_home", "customer_b_home"}
    assert {event.details["household_id"] for event in pass_events} == {"customer_a_home", "customer_b_home"}
