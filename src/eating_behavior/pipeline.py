from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .events import AnnouncementEvent, AnnouncementRecorder, EventThrottler
from .identity import IdentityDecision, SourceIdentityService
from .media import SourceSpec, configured_source_specs, iter_frames, resolve_source_specs, source_id_list
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
    worker_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class IdentityObservation:
    source: str
    timestamp_seconds: float
    cat_name: str
    coat_class: str
    confidence: float
    method: str
    household_id: str


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
                household_id=identity.household_id,
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
    unknown_methods = {"needs_identity_confirmation", "unregistered_coat", "unregistered_source", "unassociated"}
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
        identity_service: SourceIdentityService,
        recorder: AnnouncementRecorder,
        config: PipelineConfig,
        source_specs: list[SourceSpec] | tuple[SourceSpec, ...] | None = None,
        show: bool = False,
    ) -> None:
        self.detector = detector
        self.coat_classifier = coat_classifier
        self.identity_service = identity_service
        self.recorder = recorder
        self.config = config
        self.source_specs = tuple(source_specs or ())
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
        worker_cfg = pipeline_config.get("worker", {})
        sources_cfg = pipeline_config.get("sources", pipeline_config.get("camera_sources", []))

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
        identity_service = SourceIdentityService.from_config(identity_cfg)
        source_specs = configured_source_specs(sources_cfg)
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
                worker_source_ids=source_id_list(worker_cfg.get("source_ids", worker_cfg.get("camera_ids"))),
            ),
            source_specs=source_specs,
            show=show,
        )

    def configured_source_ids(self) -> tuple[str, ...]:
        return tuple(spec.source_id for spec in self.source_specs)

    def select_source_specs(self, source_ids: tuple[str, ...] | list[str] | set[str] | None = None) -> list[SourceSpec]:
        selected_ids = source_id_list(source_ids) if source_ids is not None else self.config.worker_source_ids
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
        for source_spec in source_specs:
            for packet in iter_frames(source_spec, vid_stride=vid_stride, max_frames=max_frames):
                self.process_frame(packet.source, packet.timestamp_seconds, packet.frame)
                if self.show:
                    cv2.imshow("eating_behavior", packet.frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in {ord("q"), 27}:
                        break
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
        self._run_source_specs(
            resolve_source_specs(source, source_id=source_id),
            vid_stride=vid_stride,
            max_frames=max_frames,
        )

    def run_configured_sources(
        self,
        source_ids: tuple[str, ...] | list[str] | set[str] | None = None,
        vid_stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        source_specs = self.select_source_specs(source_ids)
        if not source_specs:
            raise ValueError("No configured sources to run")
        self._run_source_specs(source_specs, vid_stride=vid_stride, max_frames=max_frames)

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
                identity = self.identity_service.decide(source, "unknown", crop)
            else:
                coat = self.coat_classifier.classify(crop)
                identity = self.identity_service.decide(source, coat.class_name, crop)
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
        if not self.throttler.allow(
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
            household_id = identity.household_id if identity is not None else ""
        elif history_identity is not None:
            cat_name = history_identity.cat_name
            identity_source = "recent_identity_timeline"
            identity_confidence = history_identity.confidence
            household_id = history_identity.household_id
        else:
            cat_name = ""
            identity_source = "unidentified"
            identity_confidence = 0.0
            household_id = (
                identity.household_id
                if identity is not None
                else self.identity_service.household_id_for_source(source)
            )
        coat_class = coat.class_name if coat is not None else "unknown"
        key_name = cat_name or coat_class or "unidentified"
        if not self.throttler.allow(
            ("cat_eating", source, key_name),
            timestamp_seconds,
            self.config.eating_event_interval_seconds,
        ):
            return
        details: dict[str, Any] = {
            "household_id": household_id,
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
                household_id=household_id,
                details=details,
            )
        )

    def _record_bowl_monitor_placeholder(self, source: str, timestamp_seconds: float) -> None:
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
