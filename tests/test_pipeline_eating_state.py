"""End-to-end frame pipeline tests for completed eating events."""

import numpy as np
import pytest

from eating_behavior.eating_state import EatingStateConfig
from eating_behavior.events import AnnouncementRecorder
from eating_behavior.identity import SourceIdentityService
from eating_behavior.models import CoatPrediction, Detection
from eating_behavior.pipeline import PipelineConfig, RealtimePipeline


class SequenceDetector:
    def __init__(self, feeding_frames: list[bool]) -> None:
        self.feeding_frames = iter(feeding_frames)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if not next(self.feeding_frames):
            return []
        return [
            Detection(0, "cat", 0.95, (0, 0, 30, 30)),
            Detection(1, "feeding_head", 0.90, (5, 5, 15, 15)),
        ]


class BlueCoatClassifier:
    def classify(self, crop: np.ndarray) -> CoatPrediction:
        return CoatPrediction(0, "blue_cat", 0.90, "other", 0.10)


def make_pipeline(tmp_path, feeding_frames: list[bool]) -> RealtimePipeline:
    return RealtimePipeline(
        detector=SequenceDetector(feeding_frames),
        coat_classifier=BlueCoatClassifier(),
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
            eating_state=EatingStateConfig(
                enter_seconds=2.0,
                exit_seconds=1.0,
                merge_gap_seconds=2.0,
                min_event_seconds=2.0,
                event_confirm_ratio=0.60,
                min_identity_observations=2,
            ),
        ),
    )


def test_pipeline_emits_one_completed_event_with_interval_and_vote_summary(tmp_path) -> None:
    pipeline = make_pipeline(tmp_path, [True, True, False, False])
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    for timestamp in (0.0, 2.0, 3.0, 5.0):
        pipeline.process_frame("cam01", timestamp, frame)

    eating_events = [
        event for event in pipeline.recorder._events if event.event_type == "cat_eating"
    ]
    assert len(eating_events) == 1
    eating = eating_events[0]
    assert eating.cat_name == "blue"
    assert eating.coat_class == "blue_cat"
    assert eating.household_id == "default"
    assert eating.details["schema_version"] == 2
    assert eating.details["start_seconds"] == 0.0
    assert eating.details["end_seconds"] == 3.0
    assert eating.details["duration_seconds"] == 3.0
    assert eating.details["observation_count"] == 2
    assert eating.details["accepted_identity_observation_count"] == 2
    assert eating.details["event_identity_confidence"] == 1.0
    assert eating.details["truncated"] is False
    assert eating.details["end_reason"] == "normal"


def test_short_feeding_candidate_is_discarded(tmp_path) -> None:
    pipeline = make_pipeline(tmp_path, [True, False])
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    pipeline.process_frame("cam01", 0.0, frame)
    pipeline.process_frame("cam01", 1.0, frame)
    pipeline.finish_source("cam01", 1.0, "eof")

    assert not any(
        event.event_type == "cat_eating" for event in pipeline.recorder._events
    )


def test_eof_finalizes_confirmed_event_as_truncated(tmp_path) -> None:
    pipeline = make_pipeline(tmp_path, [True, True])
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    pipeline.process_frame("cam01", 0.0, frame)
    pipeline.process_frame("cam01", 2.0, frame)
    pipeline.finish_source("cam01", 2.0, "eof")

    eating = next(
        event for event in pipeline.recorder._events if event.event_type == "cat_eating"
    )
    assert eating.details["start_seconds"] == 0.0
    assert eating.details["end_seconds"] == 2.0
    assert eating.details["truncated"] is True
    assert eating.details["end_reason"] == "eof"


def test_finish_clears_media_time_caches_even_when_event_sink_fails(tmp_path) -> None:
    pipeline = make_pipeline(tmp_path, [True, True])
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    pipeline.process_frame("cam01", 0.0, frame)
    pipeline.process_frame("cam01", 2.0, frame)

    class FailingRecorder:
        def record(self, event) -> None:
            raise RuntimeError("sink unavailable")

    pipeline.recorder = FailingRecorder()

    with pytest.raises(RuntimeError, match="sink unavailable"):
        pipeline.finish_source("cam01", 2.0, "error")

    assert "cam01" not in pipeline.eating_machines
    assert not any(
        observation.source == "cam01"
        for observation in pipeline.identity_timeline.observations
    )
    assert not any(
        len(key) >= 2 and key[1] == "cam01"
        for key in pipeline.throttler.last_event_time
    )


def test_event_ids_do_not_collide_across_pipeline_restarts(tmp_path) -> None:
    event_ids: list[str] = []
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    for index in range(2):
        pipeline = make_pipeline(tmp_path / str(index), [True, True])
        pipeline.process_frame("cam01", 0.0, frame)
        pipeline.process_frame("cam01", 2.0, frame)
        pipeline.finish_source("cam01", 2.0, "eof")
        eating = next(
            event
            for event in pipeline.recorder._events
            if event.event_type == "cat_eating"
        )
        event_ids.append(eating.details["event_id"])

    assert event_ids[0] != event_ids[1]
