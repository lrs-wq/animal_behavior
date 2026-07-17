"""End-to-end pipeline tests for any-cat presence reminders."""

import numpy as np
import pytest

from eating_behavior.events import AnnouncementRecorder
from eating_behavior.identity import SourceIdentityService
from eating_behavior.models import CoatPrediction, Detection
from eating_behavior.pipeline import PipelineConfig, RealtimePipeline


CAT_BLUE = Detection(0, "cat", 0.95, (0, 0, 20, 20))
CAT_TABBY = Detection(0, "cat", 0.90, (20, 0, 40, 20))


class SequenceDetector:
    def __init__(self, frames: list[list[Detection]]) -> None:
        self.frames = iter(frames)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return next(self.frames)


class SequenceCoatClassifier:
    def __init__(self, predictions: list[CoatPrediction] | None = None) -> None:
        self.predictions = iter(
            predictions or [CoatPrediction(0, "blue_cat", 0.90, "other", 0.10)] * 100
        )

    def classify(self, crop: np.ndarray) -> CoatPrediction:
        return next(self.predictions)


def build_pipeline(
    tmp_path,
    detection_frames: list[list[Detection]],
    *,
    coat_predictions: list[CoatPrediction] | None = None,
    interval: float = 10.0,
    absence: float = 2.0,
) -> RealtimePipeline:
    return RealtimePipeline(
        detector=SequenceDetector(detection_frames),
        coat_classifier=SequenceCoatClassifier(coat_predictions),
        identity_service=SourceIdentityService.from_config(
            {
                "cats": [
                    {"name": "blue", "coat_class": "blue_cat"},
                    {"name": "tabby", "coat_class": "tabby_cat"},
                ]
            }
        ),
        recorder=AnnouncementRecorder(tmp_path),
        config=PipelineConfig(
            cat_classes={"cat"},
            feeding_classes={"feeding_head"},
            coat_min_area=1,
            identity_history_seconds=8.0,
            pass_event_interval_seconds=interval,
            eating_event_interval_seconds=2.0,
            bowl_monitor_interval_seconds=30.0,
            pass_absence_confirm_seconds=absence,
        ),
    )


def pass_events(pipeline: RealtimePipeline):
    return [
        event for event in pipeline.recorder._events if event.event_type == "cat_passed"
    ]


def process(pipeline: RealtimePipeline, *timestamps: float) -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    for timestamp in timestamps:
        pipeline.process_frame("cam01", timestamp, frame)


def test_first_presence_is_immediate_and_continuous_presence_repeats(tmp_path) -> None:
    pipeline = build_pipeline(tmp_path, detection_frames=[[CAT_BLUE]] * 3)

    process(pipeline, 0.0, 9.999, 10.0)

    events = pass_events(pipeline)
    assert [event.timestamp_seconds for event in events] == [0.0, 10.0]
    assert [event.details["announcement_reason"] for event in events] == [
        "initial_presence",
        "reminder_interval_elapsed",
    ]
    assert [event.details["reminder_sequence"] for event in events] == [1, 2]
    assert [event.details["presence_sequence"] for event in events] == [1, 1]


def test_early_reappearance_is_reminded_at_original_deadline_if_still_visible(
    tmp_path,
) -> None:
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE], [], [], [CAT_BLUE], [CAT_BLUE]],
    )

    process(pipeline, 0.0, 1.0, 3.0, 9.0, 10.0)

    events = pass_events(pipeline)
    assert [event.timestamp_seconds for event in events] == [0.0, 10.0]
    assert events[1].details["presence_sequence"] == 2
    assert events[1].details["presence_started_seconds"] == 9.0


def test_absence_past_deadline_does_not_remind_until_cat_reappears(tmp_path) -> None:
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE], [], [], [], [CAT_BLUE]],
    )

    process(pipeline, 0.0, 1.0, 3.0, 10.0, 15.0)

    assert [event.timestamp_seconds for event in pass_events(pipeline)] == [0.0, 15.0]


def test_multiple_cats_and_identity_changes_share_one_source_reminder(tmp_path) -> None:
    blue = CoatPrediction(0, "blue_cat", 0.90, "other", 0.10)
    tabby = CoatPrediction(1, "tabby_cat", 0.85, "other", 0.10)
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE, CAT_TABBY]] * 3,
        coat_predictions=[blue, tabby, tabby, blue, blue, tabby],
    )

    process(pipeline, 0.0, 1.0, 10.0)

    events = pass_events(pipeline)
    assert len(events) == 2
    assert all(event.cat_name == "" for event in events)
    assert all(event.details["visible_cat_count"] == 2 for event in events)
    assert all(
        event.details["visible_cat_names"] == ["blue", "tabby"] for event in events
    )


def test_no_cat_frame_never_emits_a_presence_reminder(tmp_path) -> None:
    pipeline = build_pipeline(tmp_path, [[]])

    process(pipeline, 0.0)

    assert pass_events(pipeline) == []


def test_finish_source_clears_media_time_presence_session(tmp_path) -> None:
    pipeline = build_pipeline(tmp_path, [[CAT_BLUE], [CAT_BLUE]])
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    pipeline.process_frame("cam01", 0.0, frame)
    pipeline.finish_source("cam01", 0.0, "eof")
    assert "cam01" not in pipeline.presence_machines

    pipeline.process_frame("cam01", 0.0, frame)
    assert [event.timestamp_seconds for event in pass_events(pipeline)] == [0.0, 0.0]


def test_failed_event_sink_does_not_consume_presence_reminder(tmp_path) -> None:
    pipeline = build_pipeline(tmp_path, [[CAT_BLUE], [CAT_BLUE]])
    persisted_recorder = pipeline.recorder

    class FailOnceRecorder:
        def __init__(self) -> None:
            self.attempts = 0

        def record(self, event) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("sink unavailable")
            persisted_recorder.record(event)

    pipeline.recorder = FailOnceRecorder()
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="sink unavailable"):
        pipeline.process_frame("cam01", 0.0, frame)
    pipeline.process_frame("cam01", 0.1, frame)

    events = [
        event
        for event in persisted_recorder._events
        if event.event_type == "cat_passed"
    ]
    assert len(events) == 1
    assert events[0].timestamp_seconds == 0.1
    assert events[0].details["reminder_sequence"] == 1


def test_household_specific_reminder_interval_is_source_isolated(tmp_path) -> None:
    pipeline = RealtimePipeline(
        detector=SequenceDetector([[CAT_BLUE]] * 6),
        coat_classifier=SequenceCoatClassifier(),
        identity_service=SourceIdentityService.from_config(
            {
                "households": [
                    {
                        "household_id": "home_a",
                        "source_ids": ["cam_a"],
                        "cats": [{"name": "blue_a", "coat_class": "blue_cat"}],
                    },
                    {
                        "household_id": "home_b",
                        "source_ids": ["cam_b"],
                        "cats": [{"name": "blue_b", "coat_class": "blue_cat"}],
                    },
                ]
            }
        ),
        recorder=AnnouncementRecorder(tmp_path),
        config=PipelineConfig(
            cat_classes={"cat"},
            feeding_classes={"feeding_head"},
            coat_min_area=1,
            identity_history_seconds=8.0,
            pass_event_interval_seconds=10.0,
            eating_event_interval_seconds=2.0,
            bowl_monitor_interval_seconds=30.0,
            pass_event_intervals_by_household={"home_a": 5.0},
        ),
    )
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    for source, timestamp in (
        ("cam_a", 0.0),
        ("cam_b", 0.0),
        ("cam_a", 5.0),
        ("cam_b", 5.0),
        ("cam_a", 10.0),
        ("cam_b", 10.0),
    ):
        pipeline.process_frame(source, timestamp, frame)

    events_by_source = {
        source: [event for event in pass_events(pipeline) if event.source == source]
        for source in ("cam_a", "cam_b")
    }
    assert [event.timestamp_seconds for event in events_by_source["cam_a"]] == [
        0.0,
        5.0,
        10.0,
    ]
    assert [event.timestamp_seconds for event in events_by_source["cam_b"]] == [
        0.0,
        10.0,
    ]
    assert {
        event.details["reminder_interval_seconds"]
        for event in events_by_source["cam_a"]
    } == {5.0}
    assert {
        event.details["reminder_interval_seconds"]
        for event in events_by_source["cam_b"]
    } == {10.0}
