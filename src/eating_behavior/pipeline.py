"""Realtime inference pipeline.

This module wires together detection, coat classification, source_id-based
identity routing, event throttling, and event recording. The hot path uses only
in-memory configuration.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .association import FeedingAssociation, associate_feeding_to_cat
from .config import config_bool, config_float, config_int, config_mapping
from .eating_state import (
    CompletedEatingEvent,
    EatingFrameObservation,
    EatingStateConfig,
    EatingStateMachine,
    IdentityVote,
)
from .events import AnnouncementEvent, AnnouncementRecorder, EventThrottler
from .identity import IdentityDecision, SourceIdentityService
from .media import (
    SourceSpec,
    configured_source_specs,
    is_stream_source,
    iter_frames,
    resolve_source_specs,
    source_id_list,
)
from .models import CatDetector, CoatClassifier, CoatPrediction, Detection
from .pose import (
    DisabledPoseEstimator,
    PoseEstimator,
    PoseInferenceError,
    PosePrediction,
    RTMPoseEstimator,
    draw_pose_predictions,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Runtime pipeline settings."""

    cat_classes: set[str]
    feeding_classes: set[str]
    coat_min_area: int
    identity_history_seconds: float
    pass_event_interval_seconds: float
    eating_event_interval_seconds: float
    bowl_monitor_interval_seconds: float
    worker_source_ids: tuple[str, ...] = ()
    require_worker_assignments: bool = False
    allow_debug_source: bool = True
    coat_min_confidence: float = 0.55
    coat_min_margin: float = 0.05
    feeding_association_max_distance_ratio: float = 0.5
    feeding_association_ambiguity_margin: float = 0.1
    eating_state: EatingStateConfig = field(default_factory=EatingStateConfig)
    emit_bowl_placeholder: bool = False
    pose_enabled: bool = False
    pose_keypoint_confidence: float = 0.3

    def __post_init__(self) -> None:
        """Reject unsafe or nonsensical hot-path configuration values."""
        if self.require_worker_assignments and self.allow_debug_source:
            raise ValueError(
                "worker.allow_debug_source must be false when worker.require_assignments is enabled"
            )
        if not isinstance(self.pose_enabled, bool):
            raise ValueError("pose.enabled must be a boolean")
        if self.coat_min_area < 0:
            raise ValueError("coat.min_area must be >= 0")
        for field_name, value in (
            ("coat.confidence", self.coat_min_confidence),
            ("coat.min_margin", self.coat_min_margin),
            ("pose.keypoint_confidence", self.pose_keypoint_confidence),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        for field_name, value in (
            (
                "association.max_distance_ratio",
                self.feeding_association_max_distance_ratio,
            ),
            ("association.ambiguity_margin", self.feeding_association_ambiguity_margin),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be a finite value >= 0")
        for field_name, value in (
            ("events.identity_history_seconds", self.identity_history_seconds),
            ("events.pass_event_interval_seconds", self.pass_event_interval_seconds),
            ("events.eating_event_interval_seconds", self.eating_event_interval_seconds),
            ("events.bowl_monitor_interval_seconds", self.bowl_monitor_interval_seconds),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be a finite value >= 0")


@dataclass(frozen=True)
class CatFrameAnalysis:
    """All per-cat results produced from one detector box."""

    detection: Detection
    coat: CoatPrediction
    identity: IdentityDecision
    pose: PosePrediction | None


@dataclass(frozen=True)
class FrameAnalysis:
    """Structured output for rendering, testing, and future downstream tasks."""

    detections: tuple[Detection, ...]
    cats: tuple[CatFrameAnalysis, ...]
    eating_state: str
    pose_error_code: str = ""

    @property
    def cat_decisions(
        self,
    ) -> tuple[tuple[Detection, CoatPrediction, IdentityDecision], ...]:
        """Expose the legacy association tuple shape without rerunning models."""
        return tuple(
            (cat.detection, cat.coat, cat.identity)
            for cat in self.cats
        )

    @property
    def poses(self) -> tuple[PosePrediction, ...]:
        """Return only successfully inferred poses."""
        return tuple(cat.pose for cat in self.cats if cat.pose is not None)


@dataclass(frozen=True)
class IdentityObservation:
    """Confirmed identity observation stored in the short timeline."""

    source: str
    timestamp_seconds: float
    cat_name: str
    coat_class: str
    confidence: float
    method: str
    household_id: str


class IdentityTimeline:
    """Short-lived identity cache per source_id."""

    def __init__(self) -> None:
        self.observations: list[IdentityObservation] = []

    def add(
        self,
        source: str,
        timestamp_seconds: float,
        coat: CoatPrediction,
        identity: IdentityDecision,
    ) -> None:
        """Store only identities that are safe to expose as public cat names."""
        cat_name = event_cat_name(identity)
        if not cat_name:
            return
        self.observations.append(
            IdentityObservation(
                source=source,
                timestamp_seconds=timestamp_seconds,
                cat_name=cat_name,
                coat_class=coat.class_name,
                confidence=float(identity.confidence or coat.confidence),
                method=identity.method,
                household_id=identity.household_id,
            )
        )

    def prune(self, source: str, timestamp_seconds: float, window_seconds: float) -> None:
        """Remove old observations for the current source_id."""
        self.observations = [
            observation
            for observation in self.observations
            if observation.source != source or timestamp_seconds - observation.timestamp_seconds <= window_seconds
        ]

    def infer(
        self,
        source: str,
        timestamp_seconds: float,
        window_seconds: float,
        coat_class: str | None = None,
    ) -> IdentityObservation | None:
        """Infer identity from recent observations on the same source_id."""
        candidates = [
            observation
            for observation in self.observations
            if observation.source == source and 0.0 <= timestamp_seconds - observation.timestamp_seconds <= window_seconds
        ]
        if coat_class and coat_class != "unknown":
            # Reuse history only when the coat_class still matches.
            candidates = [observation for observation in candidates if observation.coat_class == coat_class]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda observation: (
                # Prefer newer and more confident observations.
                observation.confidence / (1.0 + timestamp_seconds - observation.timestamp_seconds),
                observation.timestamp_seconds,
            ),
        )

    def clear_source(self, source: str) -> None:
        """Drop observations before a new media-time session starts."""
        self.observations = [
            observation
            for observation in self.observations
            if observation.source != source
        ]


def normalize_class_set(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> set[str]:
    """Normalize configured class names/IDs to lowercase strings."""
    return {str(value).lower() for value in (values or [])}


def bounded_confidence(value: float) -> float:
    """Return a finite probability suitable for state and event aggregation."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(confidence):
        return 0.0
    return max(0.0, min(1.0, confidence))


def crop_frame(frame: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a frame by a detection box."""
    x1, y1, x2, y2 = xyxy
    return frame[y1:y2, x1:x2]


def is_named_class(detection: Detection, names: set[str]) -> bool:
    """Return whether a detection matches configured class names or IDs."""
    return detection.class_name.lower() in names or str(detection.class_id) in names


def coat_prediction_rejection(
    coat: CoatPrediction,
    *,
    min_confidence: float,
    min_margin: float,
) -> tuple[str, str] | None:
    """Return a stable rejection method/reason, or None for accepted evidence."""
    if coat.class_name.strip().lower() in {"", "unknown", "none", "null"}:
        return "invalid_coat_prediction", "coat_classifier_returned_unknown"
    if (
        not math.isfinite(coat.confidence)
        or not math.isfinite(coat.second_confidence)
        or not 0.0 <= coat.confidence <= 1.0
        or not 0.0 <= coat.second_confidence <= 1.0
    ):
        return "invalid_coat_prediction", "coat_classifier_returned_invalid_confidence"
    if coat.confidence < min_confidence:
        return "low_coat_confidence", "coat_confidence_below_threshold"
    if coat.confidence - coat.second_confidence < min_margin:
        return "ambiguous_coat_prediction", "coat_top1_top2_margin_below_threshold"
    return None


def event_cat_name(identity: IdentityDecision | None) -> str:
    """Convert an internal identity decision to public event cat_name."""
    if identity is None:
        return ""
    cat_name = (identity.cat_name or "").strip()
    unknown_methods = {
        "needs_identity_confirmation",
        "unregistered_coat",
        "unregistered_source",
        "unassociated",
        "insufficient_coat_crop",
        "invalid_coat_prediction",
        "low_coat_confidence",
        "ambiguous_coat_prediction",
    }
    unknown_names = {"unknown", "unidentified", "none", "null"}
    if not cat_name or identity.method in unknown_methods or cat_name.lower() in unknown_names:
        return ""
    return cat_name


class RealtimePipeline:
    """Orchestrates realtime inference and event generation."""

    def __init__(
        self,
        detector: CatDetector,
        coat_classifier: CoatClassifier,
        identity_service: SourceIdentityService,
        recorder: AnnouncementRecorder,
        config: PipelineConfig,
        source_specs: list[SourceSpec] | tuple[SourceSpec, ...] | None = None,
        show: bool = False,
        pose_estimator: PoseEstimator | None = None,
    ) -> None:
        self.detector = detector
        self.coat_classifier = coat_classifier
        self.identity_service = identity_service
        self.recorder = recorder
        self.config = config
        self.source_specs = tuple(source_specs or ())
        self.show = show
        self.pose_estimator = pose_estimator or DisabledPoseEstimator()
        self.throttler = EventThrottler()
        self.identity_timeline = IdentityTimeline()
        self.eating_machines: dict[str, EatingStateMachine] = {}
        self.source_session_sequences: dict[str, int] = {}
        self.pose_error_counts: dict[tuple[str, str], int] = {}
        self.run_session_id = uuid.uuid4().hex

    @classmethod
    def from_configs(
        cls,
        models_config: dict[str, Any],
        pipeline_config: dict[str, Any],
        output_dir: str | Path,
        device: str | None = None,
        show: bool = False,
    ) -> "RealtimePipeline":
        """Construct the pipeline from model and runtime configuration."""
        detection_cfg = config_mapping(
            pipeline_config.get("detection", {}),
            field_name="detection",
        )
        coat_cfg = config_mapping(
            pipeline_config.get("coat", {}),
            field_name="coat",
        )
        pose_cfg = config_mapping(
            pipeline_config.get("pose", {}),
            field_name="pose",
        )
        association_cfg = config_mapping(
            pipeline_config.get("association", {}),
            field_name="association",
        )
        identity_cfg = config_mapping(
            pipeline_config.get("identity", {}),
            field_name="identity",
        )
        events_cfg = config_mapping(
            pipeline_config.get("events", {}),
            field_name="events",
        )
        output_cfg = config_mapping(
            pipeline_config.get("output", {}),
            field_name="output",
        )
        worker_cfg = config_mapping(
            pipeline_config.get("worker", {}),
            field_name="worker",
        )
        sources_cfg = pipeline_config.get("sources", pipeline_config.get("camera_sources", []))
        model_security_cfg = config_mapping(
            models_config.get("security", {}),
            field_name="models.security",
        )
        pose_model_cfg = config_mapping(
            models_config.get("pose", {}),
            field_name="models.pose",
        )
        require_model_sha256 = config_bool(
            model_security_cfg.get("require_sha256", True),
            field_name="models.security.require_sha256",
        )
        allow_pickle_weights = config_bool(
            model_security_cfg.get("allow_pickle_weights", False),
            field_name="models.security.allow_pickle_weights",
        )
        allowed_model_roots = model_security_cfg.get("allowed_roots")
        detection_sha256 = models_config.get("detection_model_sha256")
        coat_sha256 = models_config.get("coat_classifier_model_sha256")
        if require_model_sha256 and (not detection_sha256 or not coat_sha256):
            raise ValueError(
                "Both detection_model_sha256 and coat_classifier_model_sha256 are required"
            )
        pose_enabled = config_bool(
            pose_cfg.get("enabled", False),
            field_name="pose.enabled",
        )
        pose_config_path = pose_model_cfg.get("config")
        pose_checkpoint_path = pose_model_cfg.get("checkpoint")
        pose_config_sha256 = pose_model_cfg.get("config_sha256")
        pose_checkpoint_sha256 = pose_model_cfg.get("checkpoint_sha256")
        if pose_enabled and (not pose_config_path or not pose_checkpoint_path):
            raise ValueError(
                "models.pose.config and models.pose.checkpoint are required when pose is enabled"
            )
        if pose_enabled and (not pose_config_sha256 or not pose_checkpoint_sha256):
            raise ValueError(
                "models.pose.config_sha256 and models.pose.checkpoint_sha256 are required "
                "when pose is enabled"
            )
        if pose_enabled and not allowed_model_roots:
            raise ValueError(
                "models.security.allowed_roots is required when pose is enabled"
            )
        detection_confidence = config_float(
            detection_cfg.get("confidence", 0.25),
            field_name="detection.confidence",
        )
        detection_iou = config_float(
            detection_cfg.get("iou", 0.70),
            field_name="detection.iou",
        )
        detection_imgsz = config_int(
            detection_cfg.get("imgsz", 640),
            field_name="detection.imgsz",
        )
        coat_imgsz = config_int(
            coat_cfg.get("imgsz", 128),
            field_name="coat.imgsz",
        )
        pose_keypoint_confidence = config_float(
            pose_cfg.get("keypoint_confidence", 0.3),
            field_name="pose.keypoint_confidence",
        )
        pose_min_cat_area = config_int(
            pose_cfg.get("min_cat_area", 0),
            field_name="pose.min_cat_area",
        )
        pose_max_instances = config_int(
            pose_cfg.get("max_instances_per_frame", 8),
            field_name="pose.max_instances_per_frame",
        )
        for field_name, value in (
            ("detection.confidence", detection_confidence),
            ("detection.iou", detection_iou),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if detection_imgsz <= 0:
            raise ValueError("detection.imgsz must be > 0")
        if coat_imgsz <= 0:
            raise ValueError("coat.imgsz must be > 0")
        if not 0.0 <= pose_keypoint_confidence <= 1.0:
            raise ValueError("pose.keypoint_confidence must be between 0 and 1")
        if pose_min_cat_area < 0:
            raise ValueError("pose.min_cat_area must be >= 0")
        if pose_max_instances <= 0:
            raise ValueError("pose.max_instances_per_frame must be > 0")

        detector = CatDetector(
            models_config["detection_model"],
            confidence=detection_confidence,
            iou=detection_iou,
            imgsz=detection_imgsz,
            device=device,
            allowed_roots=allowed_model_roots,
            expected_sha256=str(detection_sha256) if detection_sha256 else None,
            allow_pickle_weights=allow_pickle_weights,
        )
        coat_classifier = CoatClassifier(
            models_config["coat_classifier_model"],
            imgsz=coat_imgsz,
            device=device,
            allowed_roots=allowed_model_roots,
            expected_sha256=str(coat_sha256) if coat_sha256 else None,
            allow_pickle_weights=allow_pickle_weights,
        )
        pose_estimator: PoseEstimator = DisabledPoseEstimator()
        if pose_enabled:
            pose_estimator = RTMPoseEstimator(
                str(pose_config_path),
                str(pose_checkpoint_path),
                device=device,
                min_cat_area=pose_min_cat_area,
                max_instances_per_frame=pose_max_instances,
                allowed_roots=allowed_model_roots,
                expected_config_sha256=(
                    str(pose_config_sha256) if pose_config_sha256 else None
                ),
                expected_checkpoint_sha256=(
                    str(pose_checkpoint_sha256) if pose_checkpoint_sha256 else None
                ),
                allow_pickle_weights=allow_pickle_weights,
            )
        identity_service = SourceIdentityService.from_config(identity_cfg)
        # Worker-local source snapshot: stable source_id plus real video URI.
        source_specs = configured_source_specs(sources_cfg)
        identity_service.validate_configured_sources(
            tuple(spec.source_id for spec in source_specs)
        )
        if config_bool(
            output_cfg.get("save_frames", False),
            field_name="output.save_frames",
        ):
            raise ValueError("output.save_frames is not implemented")
        return cls(
            detector=detector,
            coat_classifier=coat_classifier,
            identity_service=identity_service,
            recorder=AnnouncementRecorder(
                output_dir,
                save_jsonl=config_bool(
                    output_cfg.get("save_event_jsonl", True),
                    field_name="output.save_event_jsonl",
                ),
                save_csv=config_bool(
                    output_cfg.get("save_event_csv", True),
                    field_name="output.save_event_csv",
                ),
                max_recent_events=config_int(
                    output_cfg.get("max_recent_events", 1000),
                    field_name="output.max_recent_events",
                ),
            ),
            config=PipelineConfig(
                cat_classes=normalize_class_set(detection_cfg.get("cat_classes", ["cat"])),
                feeding_classes=normalize_class_set(detection_cfg.get("feeding_classes", ["feeding_head"])),
                coat_min_area=config_int(
                    coat_cfg.get("min_area", 128 * 128),
                    field_name="coat.min_area",
                ),
                identity_history_seconds=config_float(
                    events_cfg.get("identity_history_seconds", 8.0),
                    field_name="events.identity_history_seconds",
                ),
                pass_event_interval_seconds=config_float(
                    events_cfg.get("pass_event_interval_seconds", 5.0),
                    field_name="events.pass_event_interval_seconds",
                ),
                eating_event_interval_seconds=config_float(
                    events_cfg.get("eating_event_interval_seconds", 2.0),
                    field_name="events.eating_event_interval_seconds",
                ),
                bowl_monitor_interval_seconds=config_float(
                    events_cfg.get("bowl_monitor_interval_seconds", 30.0),
                    field_name="events.bowl_monitor_interval_seconds",
                ),
                worker_source_ids=source_id_list(worker_cfg.get("source_ids", worker_cfg.get("camera_ids"))),
                require_worker_assignments=config_bool(
                    worker_cfg.get("require_assignments", False),
                    field_name="worker.require_assignments",
                ),
                allow_debug_source=config_bool(
                    worker_cfg.get("allow_debug_source", True),
                    field_name="worker.allow_debug_source",
                ),
                coat_min_confidence=config_float(
                    coat_cfg.get("confidence", 0.55),
                    field_name="coat.confidence",
                ),
                coat_min_margin=config_float(
                    coat_cfg.get("min_margin", 0.05),
                    field_name="coat.min_margin",
                ),
                feeding_association_max_distance_ratio=config_float(
                    association_cfg.get("max_distance_ratio", 0.5),
                    field_name="association.max_distance_ratio",
                ),
                feeding_association_ambiguity_margin=config_float(
                    association_cfg.get("ambiguity_margin", 0.1),
                    field_name="association.ambiguity_margin",
                ),
                eating_state=EatingStateConfig(
                    enter_seconds=config_float(
                        events_cfg.get("enter_seconds", 2.0),
                        field_name="events.enter_seconds",
                    ),
                    exit_seconds=config_float(
                        events_cfg.get("exit_seconds", 5.0),
                        field_name="events.exit_seconds",
                    ),
                    merge_gap_seconds=config_float(
                        events_cfg.get("merge_gap_seconds", 10.0),
                        field_name="events.merge_gap_seconds",
                    ),
                    min_event_seconds=config_float(
                        events_cfg.get("min_event_seconds", 2.0),
                        field_name="events.min_event_seconds",
                    ),
                    event_confirm_ratio=config_float(
                        events_cfg.get("event_confirm_ratio", 0.60),
                        field_name="events.event_confirm_ratio",
                    ),
                    min_identity_observations=config_int(
                        events_cfg.get("min_identity_observations", 2),
                        field_name="events.min_identity_observations",
                    ),
                    max_identity_candidates=config_int(
                        events_cfg.get("max_identity_candidates", 32),
                        field_name="events.max_identity_candidates",
                    ),
                ),
                emit_bowl_placeholder=config_bool(
                    events_cfg.get("emit_bowl_placeholder", False),
                    field_name="events.emit_bowl_placeholder",
                ),
                pose_enabled=pose_enabled,
                pose_keypoint_confidence=pose_keypoint_confidence,
            ),
            source_specs=source_specs,
            show=show,
            pose_estimator=pose_estimator,
        )

    def configured_source_ids(self) -> tuple[str, ...]:
        """Return all source IDs present in the loaded source snapshot."""
        return tuple(spec.source_id for spec in self.source_specs)

    def select_source_specs(self, source_ids: tuple[str, ...] | list[str] | set[str] | None = None) -> list[SourceSpec]:
        """Select source specs from explicit IDs or worker_source_ids."""
        assigned_ids = self.config.worker_source_ids
        if self.config.require_worker_assignments and not assigned_ids:
            raise ValueError("worker.source_ids must not be empty when worker.require_assignments is enabled")
        requested_ids = source_id_list(source_ids) if source_ids is not None else ()
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError("Duplicate requested source_id")
        if requested_ids and assigned_ids:
            unauthorized = [source_id for source_id in requested_ids if source_id not in assigned_ids]
            if unauthorized:
                raise ValueError(f"Requested source_id is not assigned to this worker: {', '.join(unauthorized)}")
        selected_ids = requested_ids or assigned_ids
        if not selected_ids:
            return list(self.source_specs)
        specs_by_id = {spec.source_id: spec for spec in self.source_specs}
        missing = [source_id for source_id in selected_ids if source_id not in specs_by_id]
        if missing:
            raise ValueError(f"Configured source_id not found: {', '.join(missing)}")
        return [specs_by_id[source_id] for source_id in selected_ids]

    def _run_source_specs(
        self,
        source_specs: list[SourceSpec] | tuple[SourceSpec, ...],
        vid_stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        """Run a sequence of source specs.

        The current implementation runs sources sequentially; concurrency can be
        added here later without changing frame processing semantics.
        """
        stop_requested = False
        try:
            for source_spec in source_specs:
                last_timestamp_seconds: float | None = None
                end_reason = "disconnect" if is_stream_source(source_spec.uri) else "eof"
                try:
                    for packet in iter_frames(
                        source_spec,
                        vid_stride=vid_stride,
                        max_frames=max_frames,
                    ):
                        last_timestamp_seconds = packet.timestamp_seconds
                        analysis = self.process_frame(
                            packet.source,
                            packet.timestamp_seconds,
                            packet.frame,
                        )
                        if self.show:
                            display_frame = draw_pose_predictions(
                                packet.frame,
                                analysis.poses,
                                keypoint_threshold=self.config.pose_keypoint_confidence,
                            )
                            cv2.imshow("eating_behavior", display_frame)
                            key = cv2.waitKey(1) & 0xFF
                            if key in {ord("q"), 27}:
                                end_reason = "manual_stop"
                                stop_requested = True
                                break
                except KeyboardInterrupt:
                    end_reason = "manual_stop"
                    raise
                except Exception:
                    end_reason = "error"
                    raise
                finally:
                    if last_timestamp_seconds is not None:
                        self.finish_source(
                            source_spec.source_id,
                            last_timestamp_seconds,
                            end_reason,
                        )
                if stop_requested:
                    break
        finally:
            self.recorder.flush_csv()
            if self.show:
                cv2.destroyAllWindows()

    def run_source(
        self,
        source: str,
        source_id: str | None = None,
        vid_stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        """Debug entrypoint for one raw source, optionally tagged by source_id."""
        if not self.config.allow_debug_source:
            raise PermissionError("Raw --source input is disabled for this worker")
        source_specs = resolve_source_specs(source, source_id=source_id)
        if self.config.worker_source_ids:
            unauthorized = [
                spec.source_id
                for spec in source_specs
                if spec.source_id not in self.config.worker_source_ids
            ]
            if unauthorized:
                raise PermissionError(
                    "Raw source_id is not assigned to this worker: "
                    + ", ".join(unauthorized)
                )
        self._run_source_specs(
            source_specs,
            vid_stride=vid_stride,
            max_frames=max_frames,
        )

    def run_configured_sources(
        self,
        source_ids: tuple[str, ...] | list[str] | set[str] | None = None,
        vid_stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        """Worker entrypoint for configured source snapshots."""
        source_specs = self.select_source_specs(source_ids)
        if not source_specs:
            raise ValueError("No configured sources to run")
        self._run_source_specs(source_specs, vid_stride=vid_stride, max_frames=max_frames)

    def _eating_machine_for_source(self, source: str) -> EatingStateMachine:
        """Return the current source-session machine, creating it lazily."""
        machine = self.eating_machines.get(source)
        if machine is not None:
            return machine
        session_sequence = self.source_session_sequences.get(source, 0) + 1
        self.source_session_sequences[source] = session_sequence
        machine = EatingStateMachine(
            source,
            self.config.eating_state,
            session_id=f"{self.run_session_id}-{session_sequence}",
            household_id=self.identity_service.household_id_for_source(source),
        )
        self.eating_machines[source] = machine
        return machine

    def _feeding_observation(
        self,
        source: str,
        timestamp_seconds: float,
        feeding_detections: list[Detection],
        cat_decisions: list[tuple[Detection, CoatPrediction, IdentityDecision]],
    ) -> EatingFrameObservation:
        """Collapse a frame to one bounded source-level eating observation."""
        if not feeding_detections:
            return EatingFrameObservation(
                timestamp_seconds=timestamp_seconds,
                feeding_present=False,
            )

        feeding_confidence = max(
            bounded_confidence(feeding.confidence)
            for feeding in feeding_detections
        )
        associations: list[FeedingAssociation] = []
        for feeding in feeding_detections:
            association = associate_feeding_to_cat(
                feeding,
                cat_decisions,
                max_distance_ratio=self.config.feeding_association_max_distance_ratio,
                ambiguity_margin=self.config.feeding_association_ambiguity_margin,
            )
            if association is not None:
                associations.append(association)

        votes: tuple[IdentityVote, ...] = ()
        associated_boxes = {association.cat_detection.xyxy for association in associations}
        # The first implementation has one event stream per source. Only cast
        # an identity vote when every feeding target is associated and all of
        # them resolve to the same physical cat box.
        if (
            associations
            and len(associations) == len(feeding_detections)
            and len(associated_boxes) == 1
        ):
            association = max(
                associations,
                key=lambda item: item.feeding.confidence,
            )
            cat_name = event_cat_name(association.identity)
            identity_confidence = association.identity.confidence
            household_id = association.identity.household_id
            if not cat_name and coat_prediction_rejection(
                association.coat,
                min_confidence=self.config.coat_min_confidence,
                min_margin=self.config.coat_min_margin,
            ) is None:
                history = self.identity_timeline.infer(
                    source=source,
                    timestamp_seconds=timestamp_seconds,
                    window_seconds=self.config.identity_history_seconds,
                    coat_class=association.coat.class_name,
                )
                if history is not None:
                    cat_name = history.cat_name
                    identity_confidence = history.confidence
                    household_id = history.household_id
            if cat_name and household_id:
                votes = (
                    IdentityVote(
                        cat_name=cat_name,
                        identity_confidence=bounded_confidence(identity_confidence),
                        household_id=household_id,
                        coat_class=association.coat.class_name,
                    ),
                )

        return EatingFrameObservation(
            timestamp_seconds=timestamp_seconds,
            feeding_present=True,
            feeding_confidence=feeding_confidence,
            identity_votes=votes,
        )

    def _record_completed_eating_event(self, event: CompletedEatingEvent) -> None:
        """Persist one event-level state-machine result."""
        household_id = event.household_id or self.identity_service.household_id_for_source(
            event.source
        )
        self.recorder.record(
            AnnouncementEvent(
                event_type="cat_eating",
                source=event.source,
                timestamp_seconds=round(event.start_seconds, 3),
                cat_name=event.cat_name,
                coat_class=event.coat_class,
                confidence=(
                    event.event_identity_confidence if event.cat_name else 0.0
                ),
                message_key="cat_eating",
                household_id=household_id,
                details={
                    "schema_version": 2,
                    "event_id": event.event_id,
                    "session_id": event.session_id,
                    "event_sequence": event.event_sequence,
                    "household_id": household_id,
                    "start_seconds": round(event.start_seconds, 3),
                    "end_seconds": round(event.end_seconds, 3),
                    "duration_seconds": round(event.duration_seconds, 3),
                    "event_identity_confidence": round(
                        event.event_identity_confidence,
                        6,
                    ),
                    "event_identity_margin": round(event.event_identity_margin, 6),
                    "identity_rejected_reason": event.identity_rejected_reason,
                    "observation_count": event.observation_count,
                    "accepted_identity_observation_count": (
                        event.accepted_identity_observation_count
                    ),
                    "feeding_confidence_mean": round(
                        event.feeding_confidence_mean,
                        6,
                    ),
                    "feeding_confidence_max": round(
                        event.feeding_confidence_max,
                        6,
                    ),
                    "end_reason": event.end_reason,
                    "truncated": event.truncated,
                },
            )
        )

    def finish_source(
        self,
        source: str,
        last_timestamp_seconds: float,
        reason: str,
    ) -> None:
        """Finalize event state and clear media-time caches for one session."""
        machine = self.eating_machines.pop(source, None)
        try:
            if machine is not None:
                for event in machine.finish(last_timestamp_seconds, reason):
                    self._record_completed_eating_event(event)
        finally:
            self.identity_timeline.clear_source(source)
            self.throttler.clear_source(source)
            self.pose_error_counts = {
                key: count
                for key, count in self.pose_error_counts.items()
                if key[0] != source
            }

    def process_frame(
        self,
        source: str,
        timestamp_seconds: float,
        frame: np.ndarray,
    ) -> FrameAnalysis:
        """Process one frame, emit events, and return reusable inference results."""
        self.identity_timeline.prune(source, timestamp_seconds, self.config.identity_history_seconds)
        detections = self.detector.detect(frame)
        cat_detections = [det for det in detections if is_named_class(det, self.config.cat_classes)]
        feeding_detections = [det for det in detections if is_named_class(det, self.config.feeding_classes)]

        pose_error_code = ""
        try:
            pose_predictions = self.pose_estimator.predict(frame, cat_detections)
        except PoseInferenceError as exc:
            # Pose is currently an observational side channel.  A frame-level
            # failure must be visible to consumers without corrupting the
            # existing identity/eating state machine.
            pose_error_code = exc.code
            pose_predictions = tuple(None for _ in cat_detections)
            self._note_pose_error(source, pose_error_code)
        if len(pose_predictions) != len(cat_detections):
            raise RuntimeError("Pose estimator result count does not match cat detections")

        cat_decisions: list[tuple[Detection, CoatPrediction, IdentityDecision]] = []
        for cat_det in cat_detections:
            crop = crop_frame(frame, cat_det.xyxy)
            if crop.size == 0 or cat_det.area < self.config.coat_min_area:
                # Very small or invalid crops skip coat classification.
                coat = CoatPrediction(-1, "unknown", 0.0, "unknown", 0.0)
                identity = self.identity_service.reject(
                    source,
                    method="insufficient_coat_crop",
                    reason="cat_crop_below_minimum_area",
                )
            else:
                # source is a stable source_id. Identity routing first selects
                # the household, then resolves cat_name inside that household.
                coat = self.coat_classifier.classify(crop)
                rejection = coat_prediction_rejection(
                    coat,
                    min_confidence=self.config.coat_min_confidence,
                    min_margin=self.config.coat_min_margin,
                )
                if rejection is None:
                    identity = self.identity_service.decide(
                        source,
                        coat.class_name,
                        crop,
                        evidence_confidence=coat.confidence,
                    )
                else:
                    method, reason = rejection
                    identity = self.identity_service.reject(
                        source,
                        method=method,
                        reason=reason,
                    )
            cat_decisions.append((cat_det, coat, identity))

        eating_observation = self._feeding_observation(
            source,
            timestamp_seconds,
            feeding_detections,
            cat_decisions,
        )
        eating_machine = self._eating_machine_for_source(source)
        for completed_event in eating_machine.update(eating_observation):
            self._record_completed_eating_event(completed_event)

        for cat_det, coat, identity in cat_decisions:
            # Timeline stores only confirmed identities; unknowns stay out.
            self.identity_timeline.add(source, timestamp_seconds, coat, identity)
            self._record_pass_event(source, timestamp_seconds, cat_det, coat, identity)

        if self.config.emit_bowl_placeholder:
            self._record_bowl_monitor_placeholder(source, timestamp_seconds)

        return FrameAnalysis(
            detections=tuple(detections),
            cats=tuple(
                CatFrameAnalysis(
                    detection=detection,
                    coat=coat,
                    identity=identity,
                    pose=pose,
                )
                for (detection, coat, identity), pose in zip(
                    cat_decisions,
                    pose_predictions,
                )
            ),
            eating_state=eating_machine.state.value,
            pose_error_code=pose_error_code,
        )

    def _note_pose_error(self, source: str, error_code: str) -> None:
        """Emit a bounded operational warning for persistent pose degradation."""
        key = (source, error_code)
        count = self.pose_error_counts.get(key, 0) + 1
        self.pose_error_counts[key] = count
        if count in {1, 10} or count % 300 == 0:
            LOGGER.warning(
                "Pose inference degraded for source=%r code=%s count=%d",
                source,
                error_code,
                count,
            )

    def _record_pass_event(
        self,
        source: str,
        timestamp_seconds: float,
        detection: Detection,
        coat: CoatPrediction,
        identity: IdentityDecision,
    ) -> None:
        """Record a cat-passed event."""
        cat_name = event_cat_name(identity)
        key_name = cat_name or coat.class_name or "unidentified"
        if not self.throttler.allow(
            # Include source_id so customers with the same cat name do not
            # suppress each other's events.
            ("cat_passed", source, key_name),
            timestamp_seconds,
            self.config.pass_event_interval_seconds,
        ):
            return
        self.recorder.record(
            AnnouncementEvent(
                event_type="cat_passed",
                source=source,
                timestamp_seconds=round(timestamp_seconds, 3),
                cat_name=cat_name,
                coat_class=coat.class_name,
                confidence=float(identity.confidence or coat.confidence),
                message_key="cat_passed",
                household_id=identity.household_id,
                details={
                    "household_id": identity.household_id,
                    "identity_method": identity.method,
                    "identity_reason": identity.reason,
                    "identity_candidates": list(identity.candidates),
                    "coat_confidence": round(coat.confidence, 6),
                    "coat_second_class": coat.second_class_name,
                    "coat_second_confidence": round(coat.second_confidence, 6),
                    "coat_margin": round(coat.confidence - coat.second_confidence, 6),
                    "coat_min_confidence": self.config.coat_min_confidence,
                    "coat_min_margin": self.config.coat_min_margin,
                    "detection_confidence": round(detection.confidence, 6),
                    "xyxy": detection.xyxy,
                },
            )
        )

    def _record_bowl_monitor_placeholder(self, source: str, timestamp_seconds: float) -> None:
        """Record a placeholder food-status event until the food model lands."""
        household_id = self.identity_service.household_id_for_source(source)
        if not self.throttler.allow(
            ("bowl_food_status", source),
            timestamp_seconds,
            self.config.bowl_monitor_interval_seconds,
        ):
            return
        self.recorder.record(
            AnnouncementEvent(
                event_type="bowl_food_status",
                source=source,
                timestamp_seconds=round(timestamp_seconds, 3),
                cat_name="",
                coat_class="unknown",
                confidence=0.0,
                message_key="bowl_monitor_pending",
                household_id=household_id,
                details={
                    "status": "not_implemented",
                    "household_id": household_id,
                },
            )
        )
