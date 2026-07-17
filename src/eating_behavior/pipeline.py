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
from .pose_filter import PoseFilterConfig, PoseFilterResult, TrackPoseFilter
from .posture import (
    CatPostureClassifier,
    PostureConfig,
    PostureObservation,
    PostureStateMachine,
    PostureTransition,
)
from .presence_state import (
    CatPresenceConfig,
    CatPresenceReminder,
    CatPresenceStateMachine,
)
from .tracking import CatTracker, TrackedDetection, TrackingConfig


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
    pass_absence_confirm_seconds: float = 2.0
    pass_event_intervals_by_household: dict[str, float] = field(
        default_factory=dict
    )
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
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    pose_filter: PoseFilterConfig = field(default_factory=PoseFilterConfig)
    posture: PostureConfig = field(default_factory=PostureConfig)

    def __post_init__(self) -> None:
        """Reject unsafe or nonsensical hot-path configuration values."""
        if self.require_worker_assignments and self.allow_debug_source:
            raise ValueError(
                "worker.allow_debug_source must be false when worker.require_assignments is enabled"
            )
        if not isinstance(self.pose_enabled, bool):
            raise ValueError("pose.enabled must be a boolean")
        if not isinstance(self.posture, PostureConfig):
            raise ValueError("pose.posture must be a PostureConfig")
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
        if (
            isinstance(self.pass_event_interval_seconds, bool)
            or not isinstance(self.pass_event_interval_seconds, (int, float))
            or not math.isfinite(self.pass_event_interval_seconds)
            or self.pass_event_interval_seconds <= 0.0
        ):
            raise ValueError(
                "events.cat_presence.repeat_interval_seconds must be finite and > 0"
            )
        for field_name, value in (
            ("events.identity_history_seconds", self.identity_history_seconds),
            ("events.eating_event_interval_seconds", self.eating_event_interval_seconds),
            ("events.bowl_monitor_interval_seconds", self.bowl_monitor_interval_seconds),
            (
                "events.cat_presence.absence_confirm_seconds",
                self.pass_absence_confirm_seconds,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{field_name} must be a finite value >= 0")
        normalized_intervals: dict[str, float] = {}
        for household_id, interval in self.pass_event_intervals_by_household.items():
            normalized_id = str(household_id).strip()
            if not normalized_id:
                raise ValueError(
                    "events.cat_presence.household_repeat_interval_seconds "
                    "contains a blank household_id"
                )
            if (
                isinstance(interval, bool)
                or not isinstance(interval, (int, float))
                or not math.isfinite(interval)
                or interval <= 0.0
            ):
                raise ValueError(
                    "events.cat_presence.household_repeat_interval_seconds values "
                    "must be finite and > 0"
                )
            if normalized_id in normalized_intervals:
                raise ValueError(
                    "events.cat_presence.household_repeat_interval_seconds has "
                    f"duplicate household_id after normalization: {normalized_id}"
                )
            normalized_intervals[normalized_id] = float(interval)
        self.pass_event_intervals_by_household = normalized_intervals


@dataclass(frozen=True)
class CatFrameAnalysis:
    """All per-cat results produced from one detector box."""

    detection: Detection
    coat: CoatPrediction
    identity: IdentityDecision
    pose: PosePrediction | None
    track_id: int | None = None
    pose_detection: Detection | None = None
    raw_pose: PosePrediction | None = None
    pose_status: str = ""
    pose_rejection_code: str = ""
    pose_held_keypoints: int = 0
    pose_dropped_keypoints: int = 0
    observed_posture: str = ""
    posture: str = ""
    posture_label: str = ""
    posture_confidence: float = 0.0
    posture_subject_key: str = ""


@dataclass(frozen=True)
class PostureStateSnapshot:
    """Last committed posture for one cat subject, including absent cats."""

    source: str
    household_id: str
    subject_key: str
    subject_kind: str
    cat_name: str
    posture: str
    posture_label: str
    confidence: float
    visible: bool
    last_seen_seconds: float | None
    last_changed_seconds: float | None
    transition_sequence: int
    track_id: int | None = None


@dataclass(frozen=True)
class FrameAnalysis:
    """Structured output for rendering, testing, and future downstream tasks."""

    detections: tuple[Detection, ...]
    cats: tuple[CatFrameAnalysis, ...]
    eating_state: str
    pose_error_code: str = ""
    posture_states: tuple[PostureStateSnapshot, ...] = ()

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


@dataclass
class _PostureSubjectRuntime:
    """Mutable pipeline metadata around one pure posture state machine."""

    machine: PostureStateMachine
    subject_kind: str
    household_id: str
    cat_name: str
    visible: bool = False
    last_seen_seconds: float | None = None
    track_id: int | None = None
    pending_event: AnnouncementEvent | None = None


@dataclass(frozen=True)
class _PostureFrameState:
    """Aligned posture values returned for one current cat detection."""

    subject_key: str = ""
    observed: PostureObservation | None = None
    posture: str = ""
    posture_label: str = ""
    confidence: float = 0.0


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


def parse_presence_interval_overrides(value: Any) -> dict[str, float]:
    """Parse per-household reminder intervals without cross-tenant fallback."""
    mapping = config_mapping(
        value,
        field_name="events.cat_presence.household_repeat_interval_seconds",
    )
    parsed: dict[str, float] = {}
    for household_id, raw_interval in mapping.items():
        if not isinstance(household_id, str) or not household_id.strip():
            raise ValueError(
                "events.cat_presence.household_repeat_interval_seconds keys "
                "must be non-blank household IDs"
            )
        normalized_id = household_id.strip()
        if normalized_id in parsed:
            raise ValueError(
                "events.cat_presence.household_repeat_interval_seconds has "
                f"duplicate household_id after normalization: {normalized_id}"
            )
        parsed[normalized_id] = config_float(
            raw_interval,
            field_name=(
                "events.cat_presence.household_repeat_interval_seconds."
                f"{normalized_id}"
            ),
        )
    return parsed


def parse_posture_config(
    value: Any,
    *,
    pose_enabled: bool,
    pose_keypoint_confidence: float,
) -> PostureConfig:
    """Parse and validate posture settings before any model is constructed."""
    cfg = config_mapping(value, field_name="pose.posture")
    posture_enabled = config_bool(
        cfg.get("enabled", False),
        field_name="pose.posture.enabled",
    )
    return PostureConfig(
        enabled=pose_enabled and posture_enabled,
        min_keypoint_confidence=config_float(
            cfg.get(
                "min_keypoint_confidence",
                max(0.35, pose_keypoint_confidence),
            ),
            field_name="pose.posture.min_keypoint_confidence",
        ),
        min_feeding_confidence=config_float(
            cfg.get("min_feeding_confidence", 0.50),
            field_name="pose.posture.min_feeding_confidence",
        ),
        min_visible_keypoints=config_int(
            cfg.get("min_visible_keypoints", 5),
            field_name="pose.posture.min_visible_keypoints",
        ),
        min_body_length_ratio=config_float(
            cfg.get("min_body_length_ratio", 0.12),
            field_name="pose.posture.min_body_length_ratio",
        ),
        min_supporting_paws=config_int(
            cfg.get("min_supporting_paws", 2),
            field_name="pose.posture.min_supporting_paws",
        ),
        min_complete_leg_chains=config_int(
            cfg.get("min_complete_leg_chains", 1),
            field_name="pose.posture.min_complete_leg_chains",
        ),
        lying_max_clearance_ratio=config_float(
            cfg.get("lying_max_clearance_ratio", 0.10),
            field_name="pose.posture.lying_max_clearance_ratio",
        ),
        crouching_min_clearance_ratio=config_float(
            cfg.get("crouching_min_clearance_ratio", 0.10),
            field_name="pose.posture.crouching_min_clearance_ratio",
        ),
        crouching_max_clearance_ratio=config_float(
            cfg.get("crouching_max_clearance_ratio", 0.22),
            field_name="pose.posture.crouching_max_clearance_ratio",
        ),
        standing_min_front_drop_ratio=config_float(
            cfg.get("standing_min_front_drop_ratio", 0.18),
            field_name="pose.posture.standing_min_front_drop_ratio",
        ),
        standing_min_hind_drop_ratio=config_float(
            cfg.get("standing_min_hind_drop_ratio", 0.15),
            field_name="pose.posture.standing_min_hind_drop_ratio",
        ),
        standing_min_clearance_ratio=config_float(
            cfg.get("standing_min_clearance_ratio", 0.20),
            field_name="pose.posture.standing_min_clearance_ratio",
        ),
        standing_min_leg_straightness=config_float(
            cfg.get("standing_min_leg_straightness", 0.78),
            field_name="pose.posture.standing_min_leg_straightness",
        ),
        transition_confirm_seconds=config_float(
            cfg.get("transition_confirm_seconds", 0.50),
            field_name="pose.posture.transition_confirm_seconds",
        ),
        min_confirmed_observations=config_int(
            cfg.get("min_confirmed_observations", 3),
            field_name="pose.posture.min_confirmed_observations",
        ),
        max_anonymous_subjects_per_source=config_int(
            cfg.get("max_anonymous_subjects_per_source", 64),
            field_name="pose.posture.max_anonymous_subjects_per_source",
        ),
    )


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
        cat_tracker: CatTracker | None = None,
        pose_filter: TrackPoseFilter | None = None,
        posture_classifier: CatPostureClassifier | None = None,
    ) -> None:
        self.detector = detector
        self.coat_classifier = coat_classifier
        self.identity_service = identity_service
        self.recorder = recorder
        self.config = config
        self.source_specs = tuple(source_specs or ())
        self.show = show
        self.pose_estimator = pose_estimator or DisabledPoseEstimator()
        self.cat_tracker = cat_tracker or CatTracker(config.tracking)
        self.pose_filter = pose_filter or TrackPoseFilter(config.pose_filter)
        self.posture_classifier = posture_classifier or CatPostureClassifier(
            config.posture
        )
        self.throttler = EventThrottler()
        self.identity_timeline = IdentityTimeline()
        self.eating_machines: dict[str, EatingStateMachine] = {}
        self.presence_machines: dict[str, CatPresenceStateMachine] = {}
        self.posture_subjects: dict[
            tuple[str, str], _PostureSubjectRuntime
        ] = {}
        self.posture_track_aliases: dict[tuple[str, int, int], str] = {}
        self.orphaned_posture_deliveries: dict[
            tuple[str, str], _PostureSubjectRuntime
        ] = {}
        self.posture_event_error_counts: dict[str, int] = {}
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
        pose_quality_cfg = config_mapping(
            pose_cfg.get("quality", {}),
            field_name="pose.quality",
        )
        pose_smoothing_cfg = config_mapping(
            pose_cfg.get("smoothing", {}),
            field_name="pose.smoothing",
        )
        pose_posture_cfg = config_mapping(
            pose_cfg.get("posture", {}),
            field_name="pose.posture",
        )
        tracking_cfg = config_mapping(
            pipeline_config.get("tracking", {}),
            field_name="tracking",
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
        cat_presence_cfg = config_mapping(
            events_cfg.get("cat_presence", {}),
            field_name="events.cat_presence",
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
        detection_nms_iou = config_float(
            detection_cfg.get("nms_iou", detection_iou),
            field_name="detection.nms_iou",
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
        if not 0.0 <= pose_keypoint_confidence <= 1.0:
            raise ValueError("pose.keypoint_confidence must be between 0 and 1")
        posture_config = parse_posture_config(
            pose_posture_cfg,
            pose_enabled=pose_enabled,
            pose_keypoint_confidence=pose_keypoint_confidence,
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
            ("detection.nms_iou", detection_nms_iou),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if detection_imgsz <= 0:
            raise ValueError("detection.imgsz must be > 0")
        if coat_imgsz <= 0:
            raise ValueError("coat.imgsz must be > 0")
        if pose_min_cat_area < 0:
            raise ValueError("pose.min_cat_area must be >= 0")
        if pose_max_instances <= 0:
            raise ValueError("pose.max_instances_per_frame must be > 0")
        tracking_config = TrackingConfig(
            enabled=config_bool(
                tracking_cfg.get("enabled", False),
                field_name="tracking.enabled",
            ),
            match_iou=config_float(
                tracking_cfg.get("match_iou", 0.20),
                field_name="tracking.match_iou",
            ),
            max_center_distance_ratio=config_float(
                tracking_cfg.get("max_center_distance_ratio", 0.60),
                field_name="tracking.max_center_distance_ratio",
            ),
            ambiguity_margin=config_float(
                tracking_cfg.get("ambiguity_margin", 0.05),
                field_name="tracking.ambiguity_margin",
            ),
            max_age_seconds=config_float(
                tracking_cfg.get("max_age_seconds", 0.75),
                field_name="tracking.max_age_seconds",
            ),
            bbox_smoothing_seconds=config_float(
                tracking_cfg.get("bbox_smoothing_seconds", 0.12),
                field_name="tracking.bbox_smoothing_seconds",
            ),
            max_tracks_per_source=config_int(
                tracking_cfg.get("max_tracks_per_source", 32),
                field_name="tracking.max_tracks_per_source",
            ),
        )
        pose_filter_config = PoseFilterConfig(
            enabled=config_bool(
                pose_quality_cfg.get("enabled", False),
                field_name="pose.quality.enabled",
            ),
            min_bbox_area_enter=config_int(
                pose_quality_cfg.get("min_bbox_area_enter", 8192),
                field_name="pose.quality.min_bbox_area_enter",
            ),
            min_bbox_area_keep=config_int(
                pose_quality_cfg.get("min_bbox_area_keep", 6144),
                field_name="pose.quality.min_bbox_area_keep",
            ),
            min_bbox_short_side=config_int(
                pose_quality_cfg.get("min_bbox_short_side", 64),
                field_name="pose.quality.min_bbox_short_side",
            ),
            point_enter_confidence=config_float(
                pose_quality_cfg.get("point_enter_confidence", 0.50),
                field_name="pose.quality.point_enter_confidence",
            ),
            point_keep_confidence=config_float(
                pose_quality_cfg.get("point_keep_confidence", 0.30),
                field_name="pose.quality.point_keep_confidence",
            ),
            enter_min_points=config_int(
                pose_quality_cfg.get("enter_min_points", 5),
                field_name="pose.quality.enter_min_points",
            ),
            enter_min_edges=config_int(
                pose_quality_cfg.get("enter_min_edges", 3),
                field_name="pose.quality.enter_min_edges",
            ),
            enter_min_anchors=config_int(
                pose_quality_cfg.get("enter_min_anchors", 1),
                field_name="pose.quality.enter_min_anchors",
            ),
            keep_min_points=config_int(
                pose_quality_cfg.get("keep_min_points", 3),
                field_name="pose.quality.keep_min_points",
            ),
            keep_min_edges=config_int(
                pose_quality_cfg.get("keep_min_edges", 1),
                field_name="pose.quality.keep_min_edges",
            ),
            max_bridge_gap_seconds=config_float(
                pose_smoothing_cfg.get("max_bridge_gap_seconds", 0.15),
                field_name="pose.smoothing.max_bridge_gap_seconds",
            ),
            reset_gap_seconds=config_float(
                pose_smoothing_cfg.get("reset_gap_seconds", 0.30),
                field_name="pose.smoothing.reset_gap_seconds",
            ),
            one_euro_min_cutoff=config_float(
                pose_smoothing_cfg.get("one_euro_min_cutoff", 1.0),
                field_name="pose.smoothing.one_euro_min_cutoff",
            ),
            one_euro_beta=config_float(
                pose_smoothing_cfg.get("one_euro_beta", 0.5),
                field_name="pose.smoothing.one_euro_beta",
            ),
            one_euro_derivative_cutoff=config_float(
                pose_smoothing_cfg.get("one_euro_derivative_cutoff", 1.0),
                field_name="pose.smoothing.one_euro_derivative_cutoff",
            ),
            reference_dt_seconds=config_float(
                pose_smoothing_cfg.get("reference_dt_seconds", 0.05),
                field_name="pose.smoothing.reference_dt_seconds",
            ),
            core_hold_frames=config_int(
                pose_smoothing_cfg.get("core_hold_frames", 3),
                field_name="pose.smoothing.core_hold_frames",
            ),
            joint_hold_frames=config_int(
                pose_smoothing_cfg.get("joint_hold_frames", 2),
                field_name="pose.smoothing.joint_hold_frames",
            ),
            paw_hold_frames=config_int(
                pose_smoothing_cfg.get("paw_hold_frames", 2),
                field_name="pose.smoothing.paw_hold_frames",
            ),
        )

        detector = CatDetector(
            models_config["detection_model"],
            confidence=detection_confidence,
            iou=detection_iou,
            explicit_nms_iou=detection_nms_iou,
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
                    cat_presence_cfg.get(
                        "repeat_interval_seconds",
                        events_cfg.get("pass_event_interval_seconds", 30.0 * 60.0),
                    ),
                    field_name="events.cat_presence.repeat_interval_seconds",
                ),
                eating_event_interval_seconds=config_float(
                    events_cfg.get("eating_event_interval_seconds", 2.0),
                    field_name="events.eating_event_interval_seconds",
                ),
                bowl_monitor_interval_seconds=config_float(
                    events_cfg.get("bowl_monitor_interval_seconds", 30.0),
                    field_name="events.bowl_monitor_interval_seconds",
                ),
                pass_absence_confirm_seconds=config_float(
                    cat_presence_cfg.get(
                        "absence_confirm_seconds",
                        events_cfg.get("pass_event_absence_confirm_seconds", 2.0),
                    ),
                    field_name="events.cat_presence.absence_confirm_seconds",
                ),
                pass_event_intervals_by_household=(
                    parse_presence_interval_overrides(
                        cat_presence_cfg.get(
                            "household_repeat_interval_seconds",
                            {},
                        )
                    )
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
                tracking=tracking_config,
                pose_filter=pose_filter_config,
                posture=posture_config,
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

    def _presence_machine_for_source(self, source: str) -> CatPresenceStateMachine:
        """Return the source-local any-cat reminder state machine."""
        machine = self.presence_machines.get(source)
        if machine is None:
            household_id = self.identity_service.household_id_for_source(source)
            reminder_interval = self.config.pass_event_intervals_by_household.get(
                household_id,
                self.config.pass_event_interval_seconds,
            )
            machine = CatPresenceStateMachine(
                source,
                CatPresenceConfig(
                    reminder_interval_seconds=reminder_interval,
                    absence_confirm_seconds=(
                        self.config.pass_absence_confirm_seconds
                    ),
                ),
            )
            self.presence_machines[source] = machine
        return machine

    def _feeding_associations(
        self,
        feeding_detections: list[Detection],
        cat_decisions: list[tuple[Detection, CoatPrediction, IdentityDecision]],
    ) -> tuple[FeedingAssociation, ...]:
        """Associate each supported feeding-head box once for all consumers."""
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
        return tuple(associations)

    def _feeding_observation(
        self,
        source: str,
        timestamp_seconds: float,
        feeding_detections: list[Detection],
        cat_decisions: list[tuple[Detection, CoatPrediction, IdentityDecision]],
        associations: tuple[FeedingAssociation, ...],
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

    @staticmethod
    def _posture_key(kind: str, *parts: object) -> str:
        """Build an unambiguous internal posture subject key."""
        encoded = "|".join(
            f"{len(value)}:{value}"
            for part in parts
            for value in (str(part),)
        )
        return f"{kind}|{encoded}"

    def _evict_oldest_anonymous_posture(self, source: str) -> None:
        """Bound track churn without ever evicting confirmed identities."""
        candidates = [
            (key, runtime)
            for key, runtime in self.posture_subjects.items()
            if key[0] == source and runtime.subject_kind != "identity"
        ]
        limit = self.config.posture.max_anonymous_subjects_per_source
        if len(candidates) < limit:
            return
        registry_key, removed_runtime = min(
            candidates,
            key=lambda item: (
                item[1].visible,
                item[1].last_seen_seconds
                if item[1].last_seen_seconds is not None
                else -1.0,
                item[0][1],
            ),
        )
        removed_key = registry_key[1]
        self._stash_orphaned_posture_event(source, removed_runtime)
        self.posture_subjects.pop(registry_key, None)
        self.posture_track_aliases = {
            alias_key: subject_key
            for alias_key, subject_key in self.posture_track_aliases.items()
            if alias_key[0] != source or subject_key != removed_key
        }

    def _set_posture_track_alias(
        self,
        alias_key: tuple[str, int, int],
        subject_key: str,
    ) -> None:
        """Set one source-local LRU alias while bounding long-stream churn."""
        self.posture_track_aliases.pop(alias_key, None)
        self.posture_track_aliases[alias_key] = subject_key
        source = alias_key[0]
        limit = self.config.posture.max_anonymous_subjects_per_source
        source_aliases = [
            key for key in self.posture_track_aliases if key[0] == source
        ]
        while len(source_aliases) > limit:
            oldest = source_aliases.pop(0)
            self.posture_track_aliases.pop(oldest, None)

    def _posture_runtime_for_cat(
        self,
        source: str,
        track_id: int | None,
        cat_name: str,
        total_cat_count: int,
    ) -> _PostureSubjectRuntime | None:
        """Resolve an identity-first posture subject with safe track fallback."""
        household_id = self.identity_service.household_id_for_source(source)
        session_sequence = self.source_session_sequences.get(source, 0)
        alias_key = (
            (source, session_sequence, track_id)
            if track_id is not None
            else None
        )
        aliased_subject = (
            self.posture_track_aliases.get(alias_key)
            if alias_key is not None
            else None
        )

        if cat_name:
            subject_key = self._posture_key(
                "identity",
                source,
                household_id,
                cat_name,
            )
            if aliased_subject and aliased_subject != subject_key:
                old_registry_key = (source, aliased_subject)
                target_registry_key = (source, subject_key)
                old_runtime = self.posture_subjects.get(old_registry_key)
                target_runtime = self.posture_subjects.get(target_registry_key)
                # Only an anonymous Track may migrate into a confirmed cat.
                # A confirmed A -> confirmed B identity flip must never carry
                # A's semantic posture (or rewrite A's other Track aliases) to B.
                if old_runtime is not None and old_runtime.subject_kind != "identity":
                    self.posture_subjects.pop(old_registry_key, None)
                    if target_runtime is None:
                        old_runtime.machine.rekey(subject_key)
                        old_runtime.subject_kind = "identity"
                        old_runtime.household_id = household_id
                        old_runtime.cat_name = cat_name
                        self.posture_subjects[target_registry_key] = old_runtime
                    else:
                        self._stash_orphaned_posture_event(source, old_runtime)
                    self.posture_track_aliases = {
                        key: (
                            subject_key
                            if key[0] == source and value == aliased_subject
                            else value
                        )
                        for key, value in self.posture_track_aliases.items()
                    }
            if alias_key is not None:
                self._set_posture_track_alias(alias_key, subject_key)
            registry_key = (source, subject_key)
            runtime = self.posture_subjects.get(registry_key)
            if runtime is None:
                runtime = _PostureSubjectRuntime(
                    machine=PostureStateMachine(subject_key, self.config.posture),
                    subject_kind="identity",
                    household_id=household_id,
                    cat_name=cat_name,
                )
                self.posture_subjects[registry_key] = runtime
            else:
                runtime.subject_kind = "identity"
                runtime.household_id = household_id
                runtime.cat_name = cat_name
            return runtime

        if aliased_subject:
            runtime = self.posture_subjects.get((source, aliased_subject))
            if runtime is not None:
                if alias_key is not None:
                    self._set_posture_track_alias(alias_key, aliased_subject)
                return runtime
            if alias_key is not None:
                self.posture_track_aliases.pop(alias_key, None)
        if track_id is not None:
            subject_key = self._posture_key(
                "track",
                source,
                session_sequence,
                track_id,
            )
        elif total_cat_count == 1:
            subject_key = self._posture_key(
                "untracked",
                source,
                session_sequence,
            )
        else:
            # Multiple untracked, unidentified cats cannot be safely correlated.
            return None

        registry_key = (source, subject_key)
        runtime = self.posture_subjects.get(registry_key)
        if runtime is None:
            self._evict_oldest_anonymous_posture(source)
            runtime = _PostureSubjectRuntime(
                machine=PostureStateMachine(subject_key, self.config.posture),
                subject_kind="track" if track_id is not None else "untracked",
                household_id=household_id,
                cat_name="",
            )
            self.posture_subjects[registry_key] = runtime
        if alias_key is not None:
            self._set_posture_track_alias(alias_key, subject_key)
        return runtime

    def _posture_transition_event(
        self,
        source: str,
        detection: Detection,
        coat: CoatPrediction,
        identity: IdentityDecision,
        tracked: TrackedDetection,
        pose_result: PoseFilterResult,
        runtime: _PostureSubjectRuntime,
        transition: PostureTransition,
        feeding_confidence: float,
    ) -> AnnouncementEvent:
        """Freeze one confirmed posture change into a stable event envelope."""
        previous = (
            transition.previous_posture.value
            if transition.previous_posture is not None
            else ""
        )
        return AnnouncementEvent(
            event_type="cat_posture_changed",
            source=source,
            timestamp_seconds=round(transition.timestamp_seconds, 3),
            cat_name=runtime.cat_name,
            coat_class=coat.class_name,
            confidence=bounded_confidence(transition.confidence),
            message_key=f"cat_posture_{transition.posture.value}",
            household_id=runtime.household_id,
            details={
                "schema_version": 1,
                "previous_posture": previous,
                "posture": transition.posture.value,
                "posture_label": transition.chinese_label,
                "change_reason": transition.reason,
                "transition_sequence": transition.transition_sequence,
                "candidate_started_seconds": round(
                    transition.candidate_started_seconds,
                    3,
                ),
                "confirmed_observations": transition.confirmed_observations,
                "subject_key": transition.subject_key,
                "subject_kind": runtime.subject_kind,
                "state_retained_when_missing": True,
                "evidence_source": transition.evidence_source,
                "posture_features": dict(transition.features),
                "feeding_confidence": round(feeding_confidence, 6),
                "track_id": tracked.track_id,
                "pose_status": pose_result.status,
                "pose_rejection_code": pose_result.rejection_code,
                "pose_held_keypoints": pose_result.held_keypoints,
                "pose_dropped_keypoints": pose_result.dropped_keypoints,
                "identity_method": identity.method,
                "identity_reason": identity.reason,
                "identity_candidates": list(identity.candidates),
                "coat_confidence": round(coat.confidence, 6),
                "detection_confidence": round(detection.confidence, 6),
            },
        )

    def _deliver_pending_posture_event(
        self,
        source: str,
        runtime: _PostureSubjectRuntime,
    ) -> bool:
        """Try one frozen event and acknowledge only complete sink delivery."""
        transition = runtime.machine.pending_transition
        event = runtime.pending_event
        if transition is None and event is None:
            return True
        if transition is None or event is None:
            raise RuntimeError("posture transition and event envelope diverged")
        try:
            self.recorder.record(event)
        except Exception:
            self._note_posture_event_error(source)
            return False
        runtime.machine.acknowledge(transition)
        runtime.pending_event = None
        return True

    def _retry_orphaned_posture_events(self, source: str) -> None:
        """Retry session-local events after their anonymous Track disappeared."""
        for key, runtime in list(self.orphaned_posture_deliveries.items()):
            if key[0] != source:
                continue
            if not self._deliver_pending_posture_event(source, runtime):
                break
            self.orphaned_posture_deliveries.pop(key, None)

    def _stash_orphaned_posture_event(
        self,
        source: str,
        runtime: _PostureSubjectRuntime,
    ) -> None:
        """Keep a bounded in-memory delivery after an anonymous session ends."""
        event = runtime.pending_event
        if event is None or runtime.machine.pending_transition is None:
            return
        key = (source, event.record_id)
        self.orphaned_posture_deliveries.pop(key, None)
        self.orphaned_posture_deliveries[key] = runtime
        source_keys = [
            candidate
            for candidate in self.orphaned_posture_deliveries
            if candidate[0] == source
        ]
        limit = self.config.posture.max_anonymous_subjects_per_source
        while len(source_keys) > limit:
            dropped_key = source_keys.pop(0)
            dropped_runtime = self.orphaned_posture_deliveries.pop(
                dropped_key,
                None,
            )
            dropped_event = (
                dropped_runtime.pending_event
                if dropped_runtime is not None
                else None
            )
            discard_pending = getattr(self.recorder, "discard_pending", None)
            if dropped_event is not None and callable(discard_pending):
                try:
                    discard_pending(dropped_event.record_id)
                except Exception:
                    LOGGER.exception(
                        "Failed to release abandoned recorder delivery for "
                        "source=%r record_id=%s",
                        source,
                        dropped_event.record_id,
                    )
            LOGGER.error(
                "Dropping oldest orphaned posture delivery after bounded "
                "sink outage for source=%r record_id=%s",
                source,
                dropped_key[1],
            )

    def _posture_snapshots(self, source: str) -> tuple[PostureStateSnapshot, ...]:
        """Return committed source state even when its cats are currently absent."""
        snapshots: list[PostureStateSnapshot] = []
        for (runtime_source, subject_key), runtime in self.posture_subjects.items():
            posture = runtime.machine.current_posture
            if runtime_source != source or posture is None:
                continue
            snapshots.append(
                PostureStateSnapshot(
                    source=source,
                    household_id=runtime.household_id,
                    subject_key=subject_key,
                    subject_kind=runtime.subject_kind,
                    cat_name=runtime.cat_name,
                    posture=posture.value,
                    posture_label=posture.chinese_label,
                    confidence=bounded_confidence(
                        runtime.machine.current_confidence
                    ),
                    visible=runtime.visible,
                    last_seen_seconds=runtime.last_seen_seconds,
                    last_changed_seconds=runtime.machine.last_changed_seconds,
                    transition_sequence=runtime.machine.transition_sequence,
                    track_id=runtime.track_id,
                )
            )
        return tuple(
            sorted(
                snapshots,
                key=lambda item: (item.subject_kind, item.cat_name, item.subject_key),
            )
        )

    def _update_cat_postures(
        self,
        source: str,
        timestamp_seconds: float,
        cat_decisions: list[tuple[Detection, CoatPrediction, IdentityDecision]],
        tracked_cats: tuple[TrackedDetection, ...],
        pose_results: tuple[PoseFilterResult, ...],
        feeding_associations: tuple[FeedingAssociation, ...],
    ) -> tuple[tuple[_PostureFrameState, ...], tuple[PostureStateSnapshot, ...]]:
        """Classify visible cats, emit committed changes, and retain missing state."""
        if not (self.config.pose_enabled and self.config.posture.enabled):
            return (
                tuple(_PostureFrameState() for _ in cat_decisions),
                (),
            )

        self._retry_orphaned_posture_events(source)

        for (runtime_source, _), runtime in self.posture_subjects.items():
            if runtime_source == source:
                runtime.visible = False

        frame_values = [_PostureFrameState() for _ in cat_decisions]
        grouped: dict[str, list[tuple[Any, ...]]] = {}
        for index, ((detection, coat, identity), tracked, pose_result) in enumerate(
            zip(cat_decisions, tracked_cats, pose_results)
        ):
            cat_name = event_cat_name(identity)
            runtime = self._posture_runtime_for_cat(
                source,
                tracked.track_id,
                cat_name,
                len(cat_decisions),
            )
            feeding_confidence = max(
                (
                    bounded_confidence(association.feeding.confidence)
                    for association in feeding_associations
                    if association.cat_detection == detection
                ),
                default=0.0,
            )
            observation = self.posture_classifier.classify(
                pose_result.prediction,
                pose_result.status,
                feeding_confidence,
            )
            if runtime is None:
                frame_values[index] = _PostureFrameState(observed=observation)
                continue
            runtime.visible = True
            runtime.last_seen_seconds = timestamp_seconds
            runtime.track_id = tracked.track_id
            grouped.setdefault(runtime.machine.subject_key, []).append(
                (
                    index,
                    detection,
                    coat,
                    identity,
                    tracked,
                    pose_result,
                    feeding_confidence,
                    observation,
                    runtime,
                )
            )

        seen_subjects = set(grouped)
        for subject_key, items in grouped.items():
            runtime = items[0][8]
            # Duplicate detections resolving to one identity are ambiguous. Keep
            # the last state instead of letting one physical cat overwrite it.
            valid_observations = [item for item in items if item[7] is not None]
            selected = (
                max(valid_observations, key=lambda item: item[7].confidence)
                if len(items) == 1 and valid_observations
                else items[0]
            )
            observation = selected[7] if len(items) == 1 else None
            transition = runtime.machine.update(
                timestamp_seconds,
                observation,
                visible=True,
            )
            if transition is not None:
                if runtime.pending_event is None:
                    runtime.pending_event = self._posture_transition_event(
                        source,
                        selected[1],
                        selected[2],
                        selected[3],
                        selected[4],
                        selected[5],
                        runtime,
                        transition,
                        selected[6],
                    )
                self._deliver_pending_posture_event(source, runtime)

            current = runtime.machine.current_posture
            for item in items:
                item_observation = item[7]
                frame_values[item[0]] = _PostureFrameState(
                    subject_key=subject_key,
                    observed=item_observation,
                    posture=current.value if current is not None else "",
                    posture_label=(
                        current.chinese_label if current is not None else ""
                    ),
                    confidence=(
                        bounded_confidence(runtime.machine.current_confidence)
                        if current is not None
                        else 0.0
                    ),
                )

        for (runtime_source, subject_key), runtime in self.posture_subjects.items():
            if runtime_source != source or subject_key in seen_subjects:
                continue
            runtime.visible = False
            # Missing observations clear only unconfirmed jitter candidates;
            # the committed posture and any durable pending proposal remain.
            transition = runtime.machine.update(
                timestamp_seconds,
                None,
                visible=False,
            )
            if transition is not None:
                self._deliver_pending_posture_event(source, runtime)

        return tuple(frame_values), self._posture_snapshots(source)

    def _finish_posture_source(self, source: str) -> None:
        """Keep identified semantic state while clearing session-local aliases."""
        for registry_key, runtime in list(self.posture_subjects.items()):
            if registry_key[0] != source:
                continue
            if runtime.subject_kind == "identity":
                runtime.machine.reset_session(preserve_pending_transition=True)
                runtime.visible = False
                runtime.last_seen_seconds = None
                runtime.track_id = None
            else:
                self._stash_orphaned_posture_event(source, runtime)
                self.posture_subjects.pop(registry_key, None)
        self.posture_track_aliases = {
            key: value
            for key, value in self.posture_track_aliases.items()
            if key[0] != source
        }

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
            self.presence_machines.pop(source, None)
            self._finish_posture_source(source)
            self.identity_timeline.clear_source(source)
            self.throttler.clear_source(source)
            self.pose_error_counts = {
                key: count
                for key, count in self.pose_error_counts.items()
                if key[0] != source
            }
            self.posture_event_error_counts.pop(source, None)
            self.cat_tracker.clear_source(source)
            self.pose_filter.clear_source(source)

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
            tracked_cats = self.cat_tracker.update(
                source,
                timestamp_seconds,
                cat_detections,
                frame.shape,
            )
            if len(tracked_cats) != len(cat_detections):
                raise RuntimeError(
                    "Cat tracker result count does not match cat detections"
                )
        except Exception:
            # Tracking only supplies pose crops. Fall back to aligned raw boxes
            # so identity and eating state continue on their original evidence.
            pose_error_code = "pose_tracking_failure"
            self._note_pose_error(source, pose_error_code)
            try:
                self.cat_tracker.clear_source(source)
            except Exception:
                pass
            try:
                self.pose_filter.clear_source(source)
            except Exception:
                pass
            tracked_cats = tuple(
                TrackedDetection(None, detection, detection)
                for detection in cat_detections
            )
        pose_detections = [item.smoothed_detection for item in tracked_cats]

        try:
            raw_pose_predictions = self.pose_estimator.predict(frame, pose_detections)
        except PoseInferenceError as exc:
            # Pose is currently an observational side channel.  A frame-level
            # failure must be visible to consumers without corrupting the
            # existing identity/eating state machine.
            pose_error_code = pose_error_code or exc.code
            raw_pose_predictions = tuple(None for _ in pose_detections)
            self._note_pose_error(source, exc.code)
        if len(raw_pose_predictions) != len(cat_detections):
            raise RuntimeError("Pose estimator result count does not match cat detections")
        try:
            pose_results: tuple[PoseFilterResult, ...] = tuple(
                self.pose_filter.update(
                    source,
                    timestamp_seconds,
                    tracked.track_id,
                    tracked.smoothed_detection,
                    raw_pose,
                    frame.shape,
                )
                for tracked, raw_pose in zip(tracked_cats, raw_pose_predictions)
            )
        except Exception:
            # Tracking and pose are observational side channels. An internal
            # filtering failure must not prevent identity/eating evidence from
            # reaching their state machines, and uncertain filter state must
            # not leak into the next frame.
            pose_error_code = pose_error_code or "pose_filter_failure"
            self._note_pose_error(source, "pose_filter_failure")
            try:
                self.pose_filter.clear_source(source)
            except Exception:
                pass
            pose_results = tuple(
                PoseFilterResult(
                    None,
                    "rejected",
                    "pose_filter_failure",
                    dropped_keypoints=17,
                )
                for _ in raw_pose_predictions
            )

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

        feeding_associations = self._feeding_associations(
            feeding_detections,
            cat_decisions,
        )
        eating_observation = self._feeding_observation(
            source,
            timestamp_seconds,
            feeding_detections,
            cat_decisions,
            feeding_associations,
        )
        eating_machine = self._eating_machine_for_source(source)
        for completed_event in eating_machine.update(eating_observation):
            self._record_completed_eating_event(completed_event)

        for cat_det, coat, identity in cat_decisions:
            # Timeline stores only confirmed identities; unknowns stay out.
            self.identity_timeline.add(source, timestamp_seconds, coat, identity)
        posture_frame_states, posture_states = self._update_cat_postures(
            source,
            timestamp_seconds,
            cat_decisions,
            tracked_cats,
            pose_results,
            feeding_associations,
        )
        self._update_cat_presence(source, timestamp_seconds, cat_decisions)

        if self.config.emit_bowl_placeholder:
            self._record_bowl_monitor_placeholder(source, timestamp_seconds)

        return FrameAnalysis(
            detections=tuple(detections),
            cats=tuple(
                CatFrameAnalysis(
                    detection=detection,
                    coat=coat,
                    identity=identity,
                    pose=pose_result.prediction,
                    track_id=tracked.track_id,
                    pose_detection=tracked.smoothed_detection,
                    raw_pose=raw_pose,
                    pose_status=pose_result.status,
                    pose_rejection_code=pose_result.rejection_code,
                    pose_held_keypoints=pose_result.held_keypoints,
                    pose_dropped_keypoints=pose_result.dropped_keypoints,
                    observed_posture=(
                        posture_frame.observed.posture.value
                        if posture_frame.observed is not None
                        else ""
                    ),
                    posture=posture_frame.posture,
                    posture_label=posture_frame.posture_label,
                    posture_confidence=posture_frame.confidence,
                    posture_subject_key=posture_frame.subject_key,
                )
                for (
                    (detection, coat, identity),
                    tracked,
                    raw_pose,
                    pose_result,
                    posture_frame,
                ) in zip(
                    cat_decisions,
                    tracked_cats,
                    raw_pose_predictions,
                    pose_results,
                    posture_frame_states,
                )
            ),
            eating_state=eating_machine.state.value,
            pose_error_code=pose_error_code,
            posture_states=posture_states,
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

    def _note_posture_event_error(self, source: str) -> None:
        """Log bounded sink errors while keeping posture proposals retryable."""
        count = self.posture_event_error_counts.get(source, 0) + 1
        self.posture_event_error_counts[source] = count
        if count in {1, 10} or count % 300 == 0:
            LOGGER.warning(
                "Posture event persistence failed for source=%r count=%d; "
                "the transition remains pending",
                source,
                count,
                exc_info=True,
            )

    def _update_cat_presence(
        self,
        source: str,
        timestamp_seconds: float,
        cat_decisions: list[tuple[Detection, CoatPrediction, IdentityDecision]],
    ) -> None:
        """Update one any-cat presence stream and persist a due reminder."""
        machine = self._presence_machine_for_source(source)
        reminder = machine.update(timestamp_seconds, bool(cat_decisions))
        if reminder is None:
            return
        if not cat_decisions:
            raise RuntimeError("A cat-presence reminder requires a visible cat")
        detection, coat, identity = self._presence_representative(cat_decisions)
        self._record_pass_event(
            source,
            detection,
            coat,
            identity,
            reminder,
            cat_decisions,
        )
        # Only a successfully persisted event consumes the reminder interval.
        machine.acknowledge(reminder)

    @staticmethod
    def _presence_representative(
        cat_decisions: list[tuple[Detection, CoatPrediction, IdentityDecision]],
    ) -> tuple[Detection, CoatPrediction, IdentityDecision]:
        """Choose deterministic metadata for one source-level reminder."""
        return max(
            cat_decisions,
            key=lambda item: (
                bool(event_cat_name(item[2])),
                bounded_confidence(item[2].confidence),
                bounded_confidence(item[1].confidence),
                bounded_confidence(item[0].confidence),
                item[0].area,
            ),
        )

    def _record_pass_event(
        self,
        source: str,
        detection: Detection,
        coat: CoatPrediction,
        identity: IdentityDecision,
        reminder: CatPresenceReminder,
        cat_decisions: list[tuple[Detection, CoatPrediction, IdentityDecision]],
    ) -> None:
        """Record one source-level cat-presence reminder as ``cat_passed``."""
        visible_cat_names = sorted(
            {
                name
                for _, _, decision in cat_decisions
                if (name := event_cat_name(decision))
            }
        )[:32]
        cat_name = visible_cat_names[0] if len(visible_cat_names) == 1 else ""
        household_id = self.identity_service.household_id_for_source(source)
        self.recorder.record(
            AnnouncementEvent(
                event_type="cat_passed",
                source=source,
                timestamp_seconds=round(reminder.timestamp_seconds, 3),
                cat_name=cat_name,
                coat_class=coat.class_name,
                confidence=(
                    bounded_confidence(identity.confidence) if cat_name else 0.0
                ),
                message_key="cat_passed",
                household_id=household_id,
                details={
                    "schema_version": 2,
                    "semantics": "any_cat_presence_reminder",
                    "announcement_reason": reminder.reason,
                    "presence_sequence": reminder.presence_sequence,
                    "reminder_sequence": reminder.reminder_sequence,
                    "presence_started_seconds": round(
                        reminder.presence_started_seconds,
                        3,
                    ),
                    "presence_duration_seconds": round(
                        reminder.presence_duration_seconds,
                        3,
                    ),
                    "reminder_interval_seconds": (
                        reminder.reminder_interval_seconds
                    ),
                    "visible_cat_count": len(cat_decisions),
                    "visible_cat_names": visible_cat_names,
                    "household_id": household_id,
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
