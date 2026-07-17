"""End-to-end pipeline tests for durable per-cat posture events."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from eating_behavior.events import AnnouncementRecorder
from eating_behavior.identity import SourceIdentityService
from eating_behavior.models import CoatPrediction, Detection
from eating_behavior.pipeline import PipelineConfig, RealtimePipeline
from eating_behavior.pose import PosePrediction
from eating_behavior.pose_filter import PoseFilterResult
from eating_behavior.posture import CatPosture, PostureConfig, PostureObservation
from eating_behavior.tracking import TrackedDetection


CAT_BLUE = Detection(0, "cat", 0.95, (0, 0, 40, 40))
CAT_TABBY = Detection(0, "cat", 0.93, (60, 0, 100, 40))
FEEDING_BLUE = Detection(1, "feeding_head", 0.91, (10, 10, 20, 20))
BLUE_COAT = CoatPrediction(0, "blue_cat", 0.92, "other", 0.05)
TABBY_COAT = CoatPrediction(1, "tabby_cat", 0.90, "other", 0.05)
UNKNOWN_COAT = CoatPrediction(2, "mystery_cat", 0.90, "other", 0.05)
FRAME = np.zeros((120, 120, 3), dtype=np.uint8)


class SequenceDetector:
    def __init__(self, frames: Sequence[Sequence[Detection]]) -> None:
        self._frames = iter(frames)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        del frame
        return list(next(self._frames))


class SequenceCoatClassifier:
    def __init__(self, predictions: Sequence[CoatPrediction]) -> None:
        self._predictions = iter(predictions)

    def classify(self, crop: np.ndarray) -> CoatPrediction:
        del crop
        return next(self._predictions)


class BoxTracker:
    """Assign deterministic track ids while preserving detector alignment."""

    def __init__(self) -> None:
        self.cleared: list[str] = []

    def update(
        self,
        source: str,
        timestamp_seconds: float,
        detections: Sequence[Detection],
        frame_shape: tuple[int, ...],
    ) -> tuple[TrackedDetection, ...]:
        del source, timestamp_seconds, frame_shape
        return tuple(
            TrackedDetection(
                1 if detection.xyxy == CAT_BLUE.xyxy else 2,
                detection,
                detection,
            )
            for detection in detections
        )

    def clear_source(self, source: str) -> None:
        self.cleared.append(source)


class IncrementingTracker:
    """Simulate a long stream whose tracker continuously allocates new ids."""

    def __init__(self) -> None:
        self.next_track_id = 1

    def update(
        self,
        source: str,
        timestamp_seconds: float,
        detections: Sequence[Detection],
        frame_shape: tuple[int, ...],
    ) -> tuple[TrackedDetection, ...]:
        del source, timestamp_seconds, frame_shape
        tracked: list[TrackedDetection] = []
        for detection in detections:
            tracked.append(
                TrackedDetection(
                    self.next_track_id,
                    detection,
                    detection,
                )
            )
            self.next_track_id += 1
        return tuple(tracked)

    def clear_source(self, source: str) -> None:
        del source


class ScriptedTracker:
    def __init__(self, track_ids: Sequence[int]) -> None:
        self._track_ids = iter(track_ids)

    def update(
        self,
        source: str,
        timestamp_seconds: float,
        detections: Sequence[Detection],
        frame_shape: tuple[int, ...],
    ) -> tuple[TrackedDetection, ...]:
        del source, timestamp_seconds, frame_shape
        return tuple(
            TrackedDetection(next(self._track_ids), detection, detection)
            for detection in detections
        )

    def clear_source(self, source: str) -> None:
        del source


class AlignedPoseEstimator:
    def predict(
        self,
        frame: np.ndarray,
        cat_detections: Sequence[Detection],
    ) -> tuple[PosePrediction, ...]:
        del frame
        return tuple(
            PosePrediction(detection.xyxy, (), 0.9) for detection in cat_detections
        )


class SequencePoseFilter:
    def __init__(self, statuses: Sequence[str] = ()) -> None:
        self._statuses = tuple(statuses)
        self._index = 0
        self.cleared: list[str] = []

    def update(
        self,
        source: str,
        timestamp_seconds: float,
        track_id: int | None,
        detection: Detection,
        prediction: PosePrediction | None,
        frame_shape: tuple[int, ...],
    ) -> PoseFilterResult:
        del source, timestamp_seconds, track_id, detection, frame_shape
        status = (
            self._statuses[self._index] if self._index < len(self._statuses) else "raw"
        )
        self._index += 1
        return PoseFilterResult(
            prediction if status != "rejected" else None,
            status,
            rejection_code="test_gate" if status in {"held", "rejected"} else "",
            held_keypoints=1 if status == "held" else 0,
        )

    def clear_source(self, source: str) -> None:
        self.cleared.append(source)


class SequencePostureClassifier:
    """Return controlled observations while honoring pose-quality status."""

    def __init__(self, postures: Sequence[CatPosture | None]) -> None:
        self._postures = iter(postures)
        self.calls: list[tuple[str, float]] = []

    def classify(
        self,
        prediction: PosePrediction | None,
        pose_status: str,
        feeding_confidence: float = 0.0,
    ) -> PostureObservation | None:
        del prediction
        self.calls.append((pose_status, feeding_confidence))
        posture = next(self._postures)
        if posture is None or pose_status not in {"accepted", "raw"}:
            return None
        return PostureObservation(
            posture,
            0.9,
            "pose_geometry",
            (("test_feature", 1.0),),
        )


class FeedingAwarePostureClassifier:
    def __init__(self) -> None:
        self.feeding_confidences: list[float] = []

    def classify(
        self,
        prediction: PosePrediction | None,
        pose_status: str,
        feeding_confidence: float = 0.0,
    ) -> PostureObservation | None:
        del prediction
        self.feeding_confidences.append(feeding_confidence)
        if pose_status not in {"accepted", "raw"}:
            return None
        eating = feeding_confidence > 0.0
        return PostureObservation(
            CatPosture.EATING if eating else CatPosture.STANDING,
            0.9,
            "feeding_head_association" if eating else "pose_geometry",
        )


def build_pipeline(
    tmp_path,
    detection_frames: Sequence[Sequence[Detection]],
    coat_predictions: Sequence[CoatPrediction],
    posture_classifier,
    *,
    identity_config: dict | None = None,
    pose_statuses: Sequence[str] = (),
    cat_tracker=None,
    max_posture_subjects_per_source: int = 64,
) -> RealtimePipeline:
    return RealtimePipeline(
        detector=SequenceDetector(detection_frames),
        coat_classifier=SequenceCoatClassifier(coat_predictions),
        identity_service=SourceIdentityService.from_config(
            identity_config
            or {
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
            pass_event_interval_seconds=1800.0,
            eating_event_interval_seconds=2.0,
            bowl_monitor_interval_seconds=30.0,
            pose_enabled=True,
            posture=PostureConfig(
                enabled=True,
                transition_confirm_seconds=0.0,
                min_confirmed_observations=1,
                max_anonymous_subjects_per_source=(max_posture_subjects_per_source),
            ),
        ),
        pose_estimator=AlignedPoseEstimator(),
        cat_tracker=cat_tracker or BoxTracker(),
        pose_filter=SequencePoseFilter(pose_statuses),
        posture_classifier=posture_classifier,
    )


def posture_events(recorder: AnnouncementRecorder) -> list:
    return [
        event for event in recorder._events if event.event_type == "cat_posture_changed"
    ]


def test_initial_posture_emits_once_and_each_change_updates_state(tmp_path) -> None:
    classifier = SequencePostureClassifier(
        [
            CatPosture.STANDING,
            CatPosture.STANDING,
            CatPosture.LYING,
            CatPosture.CROUCHING,
        ]
    )
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE]] * 4,
        [BLUE_COAT] * 4,
        classifier,
    )

    analyses = [
        pipeline.process_frame("cam01", timestamp, FRAME)
        for timestamp in (0.0, 0.1, 0.2, 0.3)
    ]

    events = posture_events(pipeline.recorder)
    assert [event.timestamp_seconds for event in events] == [0.0, 0.2, 0.3]
    assert [event.details["posture"] for event in events] == [
        "standing",
        "lying",
        "crouching",
    ]
    assert [event.details["previous_posture"] for event in events] == [
        "",
        "standing",
        "lying",
    ]
    assert [event.details["change_reason"] for event in events] == [
        "initial_posture",
        "posture_changed",
        "posture_changed",
    ]
    assert [analysis.cats[0].posture for analysis in analyses] == [
        "standing",
        "standing",
        "lying",
        "crouching",
    ]
    assert analyses[-1].posture_states[0].posture == "crouching"
    assert analyses[-1].posture_states[0].transition_sequence == 3


def test_missing_cat_retains_posture_and_same_posture_reappearance_is_silent(
    tmp_path,
) -> None:
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE], [], [CAT_BLUE]],
        [BLUE_COAT, BLUE_COAT],
        SequencePostureClassifier([CatPosture.STANDING, CatPosture.STANDING]),
    )

    first = pipeline.process_frame("cam01", 0.0, FRAME)
    missing = pipeline.process_frame("cam01", 0.1, FRAME)
    reappeared = pipeline.process_frame("cam01", 0.2, FRAME)

    assert first.posture_states[0].visible is True
    assert missing.cats == ()
    assert len(missing.posture_states) == 1
    assert missing.posture_states[0].posture == "standing"
    assert missing.posture_states[0].visible is False
    assert reappeared.cats[0].posture == "standing"
    assert reappeared.posture_states[0].visible is True
    assert len(posture_events(pipeline.recorder)) == 1


def test_held_and_rejected_pose_do_not_overwrite_committed_posture(tmp_path) -> None:
    classifier = SequencePostureClassifier(
        [CatPosture.STANDING, CatPosture.LYING, CatPosture.CROUCHING]
    )
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE]] * 3,
        [BLUE_COAT] * 3,
        classifier,
        pose_statuses=["raw", "held", "rejected"],
    )

    initial = pipeline.process_frame("cam01", 0.0, FRAME)
    held = pipeline.process_frame("cam01", 0.1, FRAME)
    rejected = pipeline.process_frame("cam01", 0.2, FRAME)

    assert initial.cats[0].posture == "standing"
    assert held.cats[0].observed_posture == ""
    assert held.cats[0].posture == "standing"
    assert rejected.cats[0].observed_posture == ""
    assert rejected.cats[0].posture == "standing"
    assert rejected.posture_states[0].posture == "standing"
    assert [call[0] for call in classifier.calls] == ["raw", "held", "rejected"]
    assert len(posture_events(pipeline.recorder)) == 1


def test_feeding_head_marks_only_its_associated_cat_as_eating(tmp_path) -> None:
    classifier = FeedingAwarePostureClassifier()
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE, CAT_TABBY, FEEDING_BLUE]],
        [BLUE_COAT, TABBY_COAT],
        classifier,
    )

    analysis = pipeline.process_frame("cam01", 0.0, FRAME)

    assert [cat.identity.cat_name for cat in analysis.cats] == ["blue", "tabby"]
    assert [cat.posture for cat in analysis.cats] == ["eating", "standing"]
    assert classifier.feeding_confidences == [FEEDING_BLUE.confidence, 0.0]
    events = posture_events(pipeline.recorder)
    assert [event.cat_name for event in events] == ["blue", "tabby"]
    assert [event.details["posture"] for event in events] == [
        "eating",
        "standing",
    ]
    assert events[0].details["feeding_confidence"] == FEEDING_BLUE.confidence
    assert events[1].details["feeding_confidence"] == 0.0


def test_failed_posture_event_sink_does_not_ack_and_retries_exact_transition(
    tmp_path,
) -> None:
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE], [CAT_BLUE]],
        [BLUE_COAT, BLUE_COAT],
        SequencePostureClassifier([CatPosture.STANDING, CatPosture.STANDING]),
    )
    persisted_recorder = pipeline.recorder

    class FailFirstPostureRecorder:
        def __init__(self) -> None:
            self.failed = False
            self.attempted_events = []

        def record(self, event) -> None:
            if event.event_type == "cat_posture_changed":
                self.attempted_events.append(event)
                if not self.failed:
                    self.failed = True
                    raise RuntimeError("posture sink unavailable")
            persisted_recorder.record(event)

    failing = FailFirstPostureRecorder()
    pipeline.recorder = failing

    failed = pipeline.process_frame("cam01", 0.0, FRAME)

    runtime = next(iter(pipeline.posture_subjects.values()))
    assert failed.cats[0].observed_posture == "standing"
    assert failed.cats[0].posture == ""
    assert failed.posture_states == ()
    assert runtime.machine.current_posture is None
    assert runtime.machine.pending_transition is not None
    assert runtime.machine.pending_transition.timestamp_seconds == 0.0
    assert pipeline.posture_event_error_counts == {"cam01": 1}

    retried = pipeline.process_frame("cam01", 0.1, FRAME)

    events = posture_events(persisted_recorder)
    assert len(events) == 1
    assert events[0].timestamp_seconds == 0.0
    assert events[0].details["transition_sequence"] == 1
    assert retried.cats[0].posture == "standing"
    assert runtime.machine.pending_transition is None
    assert runtime.machine.current_posture is CatPosture.STANDING
    assert len(failing.attempted_events) == 2
    assert failing.attempted_events[0] is failing.attempted_events[1]
    assert failing.attempted_events[0].record_id == events[0].record_id


def test_failed_transition_retries_while_cat_is_missing(tmp_path) -> None:
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE], []],
        [BLUE_COAT],
        SequencePostureClassifier([CatPosture.STANDING]),
    )
    persisted_recorder = pipeline.recorder

    class FailPostureOnce:
        failed = False

        def record(self, event) -> None:
            if event.event_type == "cat_posture_changed" and not self.failed:
                self.failed = True
                raise RuntimeError("posture sink unavailable")
            persisted_recorder.record(event)

    pipeline.recorder = FailPostureOnce()

    first = pipeline.process_frame("cam01", 0.0, FRAME)
    missing = pipeline.process_frame("cam01", 0.1, FRAME)

    assert first.cats[0].posture == ""
    assert missing.cats == ()
    assert len(missing.posture_states) == 1
    assert missing.posture_states[0].posture == "standing"
    assert missing.posture_states[0].visible is False
    assert len(posture_events(persisted_recorder)) == 1


def test_pending_named_transition_survives_reconnect_and_retries(tmp_path) -> None:
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE], [CAT_BLUE]],
        [BLUE_COAT, BLUE_COAT],
        SequencePostureClassifier([CatPosture.STANDING, CatPosture.STANDING]),
    )
    persisted_recorder = pipeline.recorder

    class RecoveringPostureRecorder:
        available = False

        def record(self, event) -> None:
            if event.event_type == "cat_posture_changed" and not self.available:
                raise RuntimeError("posture sink unavailable")
            persisted_recorder.record(event)

    recovering = RecoveringPostureRecorder()
    pipeline.recorder = recovering

    first = pipeline.process_frame("cam01", 0.0, FRAME)
    runtime = next(iter(pipeline.posture_subjects.values()))
    pending = runtime.machine.pending_transition
    assert first.cats[0].posture == ""
    assert pending is not None

    pipeline.finish_source("cam01", 0.0, "disconnect")
    assert runtime.machine.pending_transition == pending

    recovering.available = True
    retried = pipeline.process_frame("cam01", 0.0, FRAME)

    events = posture_events(persisted_recorder)
    assert len(events) == 1
    assert events[0].timestamp_seconds == 0.0
    assert events[0].details["transition_sequence"] == 1
    assert retried.cats[0].posture == "standing"
    assert runtime.machine.pending_transition is None


def test_pending_anonymous_transition_retries_after_session_cleanup(tmp_path) -> None:
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE], []],
        [UNKNOWN_COAT],
        SequencePostureClassifier([CatPosture.STANDING]),
        identity_config={"cats": [{"name": "blue", "coat_class": "blue_cat"}]},
    )
    persisted_recorder = pipeline.recorder

    class RecoveringPostureRecorder:
        available = False

        def record(self, event) -> None:
            if event.event_type == "cat_posture_changed" and not self.available:
                raise RuntimeError("posture sink unavailable")
            persisted_recorder.record(event)

    recovering = RecoveringPostureRecorder()
    pipeline.recorder = recovering

    first = pipeline.process_frame("cam01", 0.0, FRAME)
    assert first.cats[0].identity.cat_name == "unknown"
    assert first.cats[0].posture == ""

    pipeline.finish_source("cam01", 0.0, "disconnect")
    assert pipeline.posture_subjects == {}
    assert len(pipeline.orphaned_posture_deliveries) == 1

    recovering.available = True
    missing = pipeline.process_frame("cam01", 0.0, FRAME)

    assert missing.posture_states == ()
    assert pipeline.orphaned_posture_deliveries == {}
    events = posture_events(persisted_recorder)
    assert len(events) == 1
    assert events[0].details["subject_kind"] == "track"
    assert events[0].details["posture"] == "standing"


def test_pending_anonymous_transition_survives_subject_lru_eviction(tmp_path) -> None:
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE], [CAT_BLUE], []],
        [UNKNOWN_COAT, UNKNOWN_COAT],
        SequencePostureClassifier([CatPosture.STANDING, CatPosture.CROUCHING]),
        identity_config={"cats": [{"name": "blue", "coat_class": "blue_cat"}]},
        cat_tracker=ScriptedTracker([1, 2]),
        max_posture_subjects_per_source=1,
    )
    persisted_recorder = pipeline.recorder

    class FailPostureOnce:
        failed = False

        def record(self, event) -> None:
            if event.event_type == "cat_posture_changed" and not self.failed:
                self.failed = True
                raise RuntimeError("posture sink unavailable")
            persisted_recorder.record(event)

    pipeline.recorder = FailPostureOnce()

    pipeline.process_frame("cam01", 0.0, FRAME)
    pipeline.process_frame("cam01", 0.1, FRAME)
    assert len(pipeline.orphaned_posture_deliveries) == 1

    pipeline.process_frame("cam01", 0.2, FRAME)

    assert pipeline.orphaned_posture_deliveries == {}
    events = posture_events(persisted_recorder)
    assert sorted(
        (event.timestamp_seconds, event.details["posture"]) for event in events
    ) == [(0.0, "standing"), (0.1, "crouching")]


def test_orphan_outbox_eviction_releases_recorder_pending_slot(tmp_path) -> None:
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE], [CAT_BLUE]],
        [UNKNOWN_COAT, UNKNOWN_COAT],
        SequencePostureClassifier([CatPosture.STANDING, CatPosture.CROUCHING]),
        identity_config={"cats": [{"name": "blue", "coat_class": "blue_cat"}]},
        cat_tracker=ScriptedTracker([1, 2]),
        max_posture_subjects_per_source=1,
    )
    recorder = pipeline.recorder
    original_append_csv = recorder._append_csv

    def fail_only_posture_csv(csv_row) -> None:
        if dict(csv_row)["event_type"] == "cat_posture_changed":
            raise OSError("posture csv unavailable")
        original_append_csv(csv_row)

    recorder._append_csv = fail_only_posture_csv

    pipeline.process_frame("cam01", 0.0, FRAME)
    pipeline.finish_source("cam01", 0.0, "disconnect")
    assert len(recorder._pending_deliveries) == 1

    pipeline.process_frame("cam01", 0.0, FRAME)
    assert len(recorder._pending_deliveries) == 2
    pipeline.finish_source("cam01", 0.0, "disconnect")

    assert len(pipeline.orphaned_posture_deliveries) == 1
    remaining_record_id = next(iter(pipeline.orphaned_posture_deliveries))[1]
    assert set(recorder._pending_deliveries) == {remaining_record_id}


def test_finish_source_keeps_named_state_and_clears_anonymous_track(tmp_path) -> None:
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE, CAT_TABBY], []],
        [BLUE_COAT, UNKNOWN_COAT],
        SequencePostureClassifier([CatPosture.STANDING, CatPosture.LYING]),
        identity_config={"cats": [{"name": "blue", "coat_class": "blue_cat"}]},
    )

    first = pipeline.process_frame("cam01", 0.0, FRAME)
    assert {state.subject_kind for state in first.posture_states} == {
        "identity",
        "track",
    }

    pipeline.finish_source("cam01", 0.0, "eof")
    after_finish = pipeline.process_frame("cam01", 0.0, FRAME)

    assert len(after_finish.posture_states) == 1
    retained = after_finish.posture_states[0]
    assert retained.subject_kind == "identity"
    assert retained.cat_name == "blue"
    assert retained.posture == "standing"
    assert retained.visible is False
    assert retained.last_seen_seconds is None
    assert retained.track_id is None
    assert all(
        runtime.subject_kind == "identity"
        for runtime in pipeline.posture_subjects.values()
    )


def test_anonymous_track_migrates_to_identity_without_duplicate_event(tmp_path) -> None:
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE], [CAT_BLUE]],
        [UNKNOWN_COAT, BLUE_COAT],
        SequencePostureClassifier([CatPosture.STANDING, CatPosture.STANDING]),
        identity_config={"cats": [{"name": "blue", "coat_class": "blue_cat"}]},
    )

    anonymous = pipeline.process_frame("cam01", 0.0, FRAME)
    identified = pipeline.process_frame("cam01", 0.1, FRAME)

    assert anonymous.cats[0].identity.cat_name == "unknown"
    assert identified.cats[0].identity.cat_name == "blue"
    assert (
        anonymous.cats[0].posture_subject_key != identified.cats[0].posture_subject_key
    )
    assert identified.cats[0].posture == "standing"
    assert len(identified.posture_states) == 1
    assert identified.posture_states[0].subject_kind == "identity"
    assert identified.posture_states[0].cat_name == "blue"
    assert identified.posture_states[0].transition_sequence == 1
    events = posture_events(pipeline.recorder)
    assert len(events) == 1
    assert events[0].cat_name == ""
    assert events[0].details["subject_kind"] == "track"


def test_posture_state_is_isolated_between_sources(tmp_path) -> None:
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE], [CAT_BLUE]],
        [BLUE_COAT, BLUE_COAT],
        SequencePostureClassifier([CatPosture.STANDING, CatPosture.LYING]),
    )

    first = pipeline.process_frame("cam_a", 0.0, FRAME)
    second = pipeline.process_frame("cam_b", 0.0, FRAME)

    assert first.posture_states[0].posture == "standing"
    assert second.posture_states[0].posture == "lying"
    assert first.posture_states[0].subject_key != second.posture_states[0].subject_key
    events = posture_events(pipeline.recorder)
    assert [(event.source, event.details["posture"]) for event in events] == [
        ("cam_a", "standing"),
        ("cam_b", "lying"),
    ]


def test_identified_track_alias_history_is_bounded_per_source(tmp_path) -> None:
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE]] * 4,
        [BLUE_COAT] * 4,
        SequencePostureClassifier([CatPosture.STANDING] * 4),
        cat_tracker=IncrementingTracker(),
        max_posture_subjects_per_source=2,
    )

    for timestamp in (0.0, 0.1, 0.2, 0.3):
        latest = pipeline.process_frame("cam01", timestamp, FRAME)

    aliases = [key for key in pipeline.posture_track_aliases if key[0] == "cam01"]
    assert len(aliases) == 2
    assert {key[2] for key in aliases} == {3, 4}
    assert len(pipeline.posture_subjects) == 1
    assert latest.posture_states[0].cat_name == "blue"
    assert latest.posture_states[0].posture == "standing"
    assert len(posture_events(pipeline.recorder)) == 1


def test_active_alias_hit_refreshes_lru_during_temporary_identity_rejection(
    tmp_path,
) -> None:
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE]] * 4,
        [BLUE_COAT, TABBY_COAT, UNKNOWN_COAT, UNKNOWN_COAT],
        SequencePostureClassifier([CatPosture.STANDING] * 4),
        cat_tracker=ScriptedTracker([1, 2, 1, 3]),
        max_posture_subjects_per_source=2,
    )

    analyses = [
        pipeline.process_frame("cam01", timestamp, FRAME)
        for timestamp in (0.0, 0.1, 0.2, 0.3)
    ]

    assert analyses[2].cats[0].identity.cat_name == "unknown"
    assert analyses[2].cats[0].posture == "standing"
    aliases = [key for key in pipeline.posture_track_aliases if key[0] == "cam01"]
    assert {key[2] for key in aliases} == {1, 3}


def test_confirmed_identity_flip_never_migrates_another_cats_posture(tmp_path) -> None:
    pipeline = build_pipeline(
        tmp_path,
        [[CAT_BLUE], [CAT_BLUE]],
        [BLUE_COAT, TABBY_COAT],
        SequencePostureClassifier([CatPosture.STANDING, CatPosture.STANDING]),
    )

    first = pipeline.process_frame("cam01", 0.0, FRAME)
    second = pipeline.process_frame("cam01", 0.1, FRAME)

    assert first.cats[0].identity.cat_name == "blue"
    assert second.cats[0].identity.cat_name == "tabby"
    assert {
        (state.cat_name, state.posture, state.visible)
        for state in second.posture_states
    } == {
        ("blue", "standing", False),
        ("tabby", "standing", True),
    }
    events = posture_events(pipeline.recorder)
    assert [event.cat_name for event in events] == ["blue", "tabby"]
