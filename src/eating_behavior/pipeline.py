from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .events import AnnouncementEvent, AnnouncementRecorder, EventThrottler
from .identity import HouseholdRegistry, IdentityDecision, IdentityService
from .media import iter_frames, resolve_sources
from .models import CatDetector, CoatClassifier, CoatPrediction, Detection


@dataclass
class PipelineConfig:
    cat_classes: set[str]
    feeding_classes: set[str]
    coat_min_area: int
    identity_history_seconds: float
    pass_event_interval_seconds: float
    eating_event_interval_seconds: float
    bowl_monitor_interval_seconds: float


@dataclass(frozen=True)
class IdentityObservation:
    source: str
    timestamp_seconds: float
    cat_name: str
    coat_class: str
    confidence: float
    method: str


class IdentityTimeline:
    def __init__(self) -> None:
        self.observations: list[IdentityObservation] = []

    def add(
        self,
        source: str,
        timestamp_seconds: float,
        coat: CoatPrediction,
        identity: IdentityDecision,
    ) -> None:
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
            )
        )

    def prune(self, source: str, timestamp_seconds: float, window_seconds: float) -> None:
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
        candidates = [
            observation
            for observation in self.observations
            if observation.source == source and 0.0 <= timestamp_seconds - observation.timestamp_seconds <= window_seconds
        ]
        if coat_class and coat_class != "unknown":
            candidates = [observation for observation in candidates if observation.coat_class == coat_class]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda observation: (
                observation.confidence / (1.0 + timestamp_seconds - observation.timestamp_seconds),
                observation.timestamp_seconds,
            ),
        )


def normalize_class_set(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> set[str]:
    return {str(value).lower() for value in (values or [])}


def crop_frame(frame: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return frame[y1:y2, x1:x2]


def is_named_class(detection: Detection, names: set[str]) -> bool:
    return detection.class_name.lower() in names or str(detection.class_id) in names


def event_cat_name(identity: IdentityDecision | None) -> str:
    if identity is None:
        return ""
    cat_name = (identity.cat_name or "").strip()
    unknown_methods = {"needs_identity_confirmation", "unregistered_coat", "unassociated"}
    unknown_names = {"unknown", "unidentified", "none", "null"}
    if not cat_name or identity.method in unknown_methods or cat_name.lower() in unknown_names:
        return ""
    return cat_name


def associate_feeding_to_cat(
    feeding: Detection,
    cat_decisions: list[tuple[Detection, CoatPrediction, IdentityDecision]],
) -> tuple[CoatPrediction | None, IdentityDecision | None]:
    if not cat_decisions:
        return None, None
    fx1, fy1, fx2, fy2 = feeding.xyxy
    fc_x = (fx1 + fx2) / 2
    fc_y = (fy1 + fy2) / 2
    best: tuple[float, CoatPrediction, IdentityDecision] | None = None
    for cat_det, coat, identity in cat_decisions:
        x1, y1, x2, y2 = cat_det.xyxy
        inside = x1 <= fc_x <= x2 and y1 <= fc_y <= y2
        distance = 0.0 if inside else min(
            abs(fc_x - x1),
            abs(fc_x - x2),
            abs(fc_y - y1),
            abs(fc_y - y2),
        )
        score = -distance + (10000.0 if inside else 0.0)
        if best is None or score > best[0]:
            best = (score, coat, identity)
    if best is None:
        return None, None
    return best[1], best[2]


class RealtimePipeline:
    def __init__(
        self,
        detector: CatDetector,
        coat_classifier: CoatClassifier,
        identity_service: IdentityService,
        recorder: AnnouncementRecorder,
        config: PipelineConfig,
        show: bool = False,
    ) -> None:
        self.detector = detector
        self.coat_classifier = coat_classifier
        self.identity_service = identity_service
        self.recorder = recorder
        self.config = config
        self.show = show
        self.throttler = EventThrottler()
        self.identity_timeline = IdentityTimeline()

    @classmethod
    def from_configs(
        cls,
        models_config: dict[str, Any],
        pipeline_config: dict[str, Any],
        output_dir: str | Path,
        device: str | None = None,
        show: bool = False,
    ) -> "RealtimePipeline":
        detection_cfg = pipeline_config.get("detection", {})
        coat_cfg = pipeline_config.get("coat", {})
        identity_cfg = pipeline_config.get("identity", {})
        events_cfg = pipeline_config.get("events", {})

        detector = CatDetector(
            models_config["detection_model"],
            confidence=float(detection_cfg.get("confidence", 0.25)),
            iou=float(detection_cfg.get("iou", 0.70)),
            imgsz=int(detection_cfg.get("imgsz", 640)),
            device=device,
        )
        coat_classifier = CoatClassifier(
            models_config["coat_classifier_model"],
            imgsz=int(coat_cfg.get("imgsz", 128)),
            device=device,
        )
        registry = HouseholdRegistry(
            cats=list(identity_cfg.get("cats") or []),
            unknown_name=str(identity_cfg.get("unknown_name", "unknown")),
        )
        identity_service = IdentityService(
            registry=registry,
            direct_when_single_same_coat=bool(identity_cfg.get("direct_when_single_same_coat", True)),
        )
        return cls(
            detector=detector,
            coat_classifier=coat_classifier,
            identity_service=identity_service,
            recorder=AnnouncementRecorder(output_dir),
            config=PipelineConfig(
                cat_classes=normalize_class_set(detection_cfg.get("cat_classes", ["cat"])),
                feeding_classes=normalize_class_set(detection_cfg.get("feeding_classes", ["feeding_head"])),
                coat_min_area=int(coat_cfg.get("min_area", 128 * 128)),
                identity_history_seconds=float(events_cfg.get("identity_history_seconds", 8.0)),
                pass_event_interval_seconds=float(events_cfg.get("pass_event_interval_seconds", 5.0)),
                eating_event_interval_seconds=float(events_cfg.get("eating_event_interval_seconds", 2.0)),
                bowl_monitor_interval_seconds=float(events_cfg.get("bowl_monitor_interval_seconds", 30.0)),
            ),
            show=show,
        )

    def run_source(self, source: str, vid_stride: int = 1, max_frames: int | None = None) -> None:
        for resolved_source in resolve_sources(source):
            for packet in iter_frames(resolved_source, vid_stride=vid_stride, max_frames=max_frames):
                self.process_frame(packet.source, packet.timestamp_seconds, packet.frame)
                if self.show:
                    cv2.imshow("eating_behavior", packet.frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in {ord("q"), 27}:
                        break
        self.recorder.flush_csv()
        if self.show:
            cv2.destroyAllWindows()

    def process_frame(self, source: str, timestamp_seconds: float, frame: np.ndarray) -> None:
        self.identity_timeline.prune(source, timestamp_seconds, self.config.identity_history_seconds)
        detections = self.detector.detect(frame)
        cat_detections = [det for det in detections if is_named_class(det, self.config.cat_classes)]
        feeding_detections = [det for det in detections if is_named_class(det, self.config.feeding_classes)]

        cat_decisions: list[tuple[Detection, CoatPrediction, IdentityDecision]] = []
        for cat_det in cat_detections:
            crop = crop_frame(frame, cat_det.xyxy)
            if crop.size == 0 or cat_det.area < self.config.coat_min_area:
                coat = CoatPrediction(-1, "unknown", 0.0, "unknown", 0.0)
                identity = self.identity_service.decide("unknown", crop)
            else:
                coat = self.coat_classifier.classify(crop)
                identity = self.identity_service.decide(coat.class_name, crop)
            cat_decisions.append((cat_det, coat, identity))

        for feeding in feeding_detections:
            coat, identity = associate_feeding_to_cat(feeding, cat_decisions)
            history_identity = None
            if not event_cat_name(identity):
                history_identity = self.identity_timeline.infer(
                    source=source,
                    timestamp_seconds=timestamp_seconds,
                    window_seconds=self.config.identity_history_seconds,
                    coat_class=coat.class_name if coat is not None else None,
                )
            self._record_eating_event(source, timestamp_seconds, feeding, coat, identity, history_identity)

        for cat_det, coat, identity in cat_decisions:
            self.identity_timeline.add(source, timestamp_seconds, coat, identity)
            self._record_pass_event(source, timestamp_seconds, cat_det, coat, identity)

        self._record_bowl_monitor_placeholder(source, timestamp_seconds)

    def _record_pass_event(
        self,
        source: str,
        timestamp_seconds: float,
        detection: Detection,
        coat: CoatPrediction,
        identity: IdentityDecision,
    ) -> None:
        cat_name = event_cat_name(identity)
        key_name = cat_name or coat.class_name or "unidentified"
        if not self.throttler.allow(("cat_passed", key_name), timestamp_seconds, self.config.pass_event_interval_seconds):
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
                details={
                    "identity_method": identity.method,
                    "identity_reason": identity.reason,
                    "identity_candidates": list(identity.candidates),
                    "coat_confidence": round(coat.confidence, 6),
                    "detection_confidence": round(detection.confidence, 6),
                    "xyxy": detection.xyxy,
                },
            )
        )

    def _record_eating_event(
        self,
        source: str,
        timestamp_seconds: float,
        feeding: Detection,
        coat: CoatPrediction | None,
        identity: IdentityDecision | None,
        history_identity: IdentityObservation | None,
    ) -> None:
        current_cat_name = event_cat_name(identity)
        if current_cat_name:
            cat_name = current_cat_name
            identity_source = "current_frame"
            identity_confidence = float(identity.confidence) if identity is not None else 0.0
        elif history_identity is not None:
            cat_name = history_identity.cat_name
            identity_source = "recent_identity_timeline"
            identity_confidence = history_identity.confidence
        else:
            cat_name = ""
            identity_source = "unidentified"
            identity_confidence = 0.0
        coat_class = coat.class_name if coat is not None else "unknown"
        key_name = cat_name or coat_class or "unidentified"
        if not self.throttler.allow(("cat_eating", key_name), timestamp_seconds, self.config.eating_event_interval_seconds):
            return
        details: dict[str, Any] = {
            "identity_source": identity_source,
            "identity_method": identity.method if identity is not None else "unassociated",
            "identity_reason": identity.reason if identity is not None else "no_associated_cat",
            "identity_candidates": list(identity.candidates) if identity is not None else [],
            "identity_confidence": round(identity_confidence, 6),
            "feeding_confidence": round(feeding.confidence, 6),
            "xyxy": feeding.xyxy,
        }
        if history_identity is not None and identity_source == "recent_identity_timeline":
            details.update(
                {
                    "history_age_seconds": round(timestamp_seconds - history_identity.timestamp_seconds, 3),
                    "history_identity_method": history_identity.method,
                    "history_coat_class": history_identity.coat_class,
                    "identity_history_seconds": self.config.identity_history_seconds,
                }
            )
        self.recorder.record(
            AnnouncementEvent(
                event_type="cat_eating",
                source=source,
                timestamp_seconds=round(timestamp_seconds, 3),
                cat_name=cat_name,
                coat_class=coat_class,
                confidence=feeding.confidence,
                message_key="cat_eating",
                details=details,
            )
        )

    def _record_bowl_monitor_placeholder(self, source: str, timestamp_seconds: float) -> None:
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
                details={"status": "not_implemented"},
            )
        )
