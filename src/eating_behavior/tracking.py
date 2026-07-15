"""Source-scoped cat tracking and pose-crop stabilization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .models import Detection


FloatBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackingConfig:
    """Bounded online tracker configuration."""

    enabled: bool = False
    match_iou: float = 0.20
    max_center_distance_ratio: float = 0.60
    ambiguity_margin: float = 0.05
    max_age_seconds: float = 0.75
    bbox_smoothing_seconds: float = 0.12
    max_tracks_per_source: int = 32

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("tracking.enabled must be a boolean")
        for name, value in (
            ("tracking.match_iou", self.match_iou),
            (
                "tracking.max_center_distance_ratio",
                self.max_center_distance_ratio,
            ),
            ("tracking.ambiguity_margin", self.ambiguity_margin),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name, value in (
            ("tracking.max_age_seconds", self.max_age_seconds),
            ("tracking.bbox_smoothing_seconds", self.bbox_smoothing_seconds),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite value > 0")
        if (
            isinstance(self.max_tracks_per_source, bool)
            or not isinstance(self.max_tracks_per_source, int)
            or self.max_tracks_per_source <= 0
        ):
            raise ValueError("tracking.max_tracks_per_source must be > 0")


@dataclass(frozen=True)
class TrackedDetection:
    """One current detection plus its stable identity and pose crop."""

    track_id: int | None
    raw_detection: Detection
    smoothed_detection: Detection


@dataclass
class _Track:
    track_id: int
    last_timestamp: float
    raw_box: FloatBox
    smoothed_box: FloatBox
    velocity_x: float = 0.0
    velocity_y: float = 0.0


@dataclass
class _SourceTracks:
    next_track_id: int = 1
    last_timestamp: float | None = None
    tracks: dict[int, _Track] = field(default_factory=dict)


def _center_size(box: FloatBox) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, width, height


def _from_center_size(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> FloatBox:
    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


def _box_iou(first: FloatBox, second: FloatBox) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0,
        min(ay2, by2) - max(ay1, by1),
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _center_distance_ratio(first: FloatBox, second: FloatBox) -> float:
    first_x, first_y, first_width, first_height = _center_size(first)
    second_x, second_y, second_width, second_height = _center_size(second)
    diagonal = max(
        1.0,
        (
            math.hypot(first_width, first_height)
            + math.hypot(second_width, second_height)
        )
        / 2.0,
    )
    return math.hypot(second_x - first_x, second_y - first_y) / diagonal


class CatTracker:
    """Associate cat boxes within one source and smooth pose crops over time."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._sources: dict[str, _SourceTracks] = {}

    def update(
        self,
        source: str,
        timestamp_seconds: float,
        detections: Sequence[Detection],
        frame_shape: tuple[int, ...],
    ) -> tuple[TrackedDetection, ...]:
        """Return one aligned tracked result for each current detection."""
        if not self.config.enabled:
            return tuple(
                TrackedDetection(None, detection, detection)
                for detection in detections
            )
        if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0.0:
            raise ValueError("tracking timestamp must be finite and >= 0")
        if len(frame_shape) < 2 or frame_shape[0] <= 0 or frame_shape[1] <= 0:
            raise ValueError("tracking frame shape must contain positive dimensions")

        state = self._sources.setdefault(source, _SourceTracks())
        if (
            state.last_timestamp is not None
            and timestamp_seconds < state.last_timestamp
        ):
            # A source id can be reused by a later media session. Never carry
            # identities backwards across a reset timeline.
            state = _SourceTracks()
            self._sources[source] = state
        state.last_timestamp = timestamp_seconds
        self._expire_tracks(state, timestamp_seconds)

        raw_boxes = [tuple(float(value) for value in item.xyxy) for item in detections]
        matches = self._match(state, raw_boxes, timestamp_seconds)
        output: list[TrackedDetection | None] = [None] * len(detections)
        matched_tracks: set[int] = set()
        for detection_index, track_id in matches.items():
            track = state.tracks[track_id]
            smoothed_box = self._update_track(
                track,
                raw_boxes[detection_index],
                timestamp_seconds,
            )
            matched_tracks.add(track_id)
            output[detection_index] = TrackedDetection(
                track_id,
                detections[detection_index],
                self._to_detection(
                    detections[detection_index],
                    smoothed_box,
                    frame_shape,
                ),
            )

        for detection_index, detection in enumerate(detections):
            if output[detection_index] is not None:
                continue
            if not self._make_room(state, matched_tracks):
                output[detection_index] = TrackedDetection(
                    None,
                    detection,
                    detection,
                )
                continue
            track_id = state.next_track_id
            state.next_track_id += 1
            raw_box = raw_boxes[detection_index]
            state.tracks[track_id] = _Track(
                track_id=track_id,
                last_timestamp=timestamp_seconds,
                raw_box=raw_box,
                smoothed_box=raw_box,
            )
            matched_tracks.add(track_id)
            output[detection_index] = TrackedDetection(
                track_id,
                detection,
                detection,
            )

        return tuple(item for item in output if item is not None)

    def clear_source(self, source: str) -> None:
        """Forget all state belonging to one completed source session."""
        self._sources.pop(source, None)

    def _expire_tracks(
        self,
        state: _SourceTracks,
        timestamp_seconds: float,
    ) -> None:
        expired = [
            track_id
            for track_id, track in state.tracks.items()
            if timestamp_seconds - track.last_timestamp
            > self.config.max_age_seconds
        ]
        for track_id in expired:
            del state.tracks[track_id]

    def _predicted_box(self, track: _Track, timestamp_seconds: float) -> FloatBox:
        center_x, center_y, width, height = _center_size(track.smoothed_box)
        elapsed = max(
            0.0,
            min(
                timestamp_seconds - track.last_timestamp,
                self.config.max_age_seconds,
            ),
        )
        return _from_center_size(
            center_x + track.velocity_x * elapsed,
            center_y + track.velocity_y * elapsed,
            width,
            height,
        )

    def _match(
        self,
        state: _SourceTracks,
        raw_boxes: Sequence[FloatBox],
        timestamp_seconds: float,
    ) -> dict[int, int]:
        candidates: list[tuple[float, float, int, int]] = []
        for track_id, track in state.tracks.items():
            predicted = self._predicted_box(track, timestamp_seconds)
            for detection_index, raw_box in enumerate(raw_boxes):
                overlap = _box_iou(predicted, raw_box)
                distance = _center_distance_ratio(predicted, raw_box)
                if (
                    overlap >= self.config.match_iou
                    or distance <= self.config.max_center_distance_ratio
                ):
                    candidates.append(
                        (1.0 - overlap + 0.25 * distance, distance, track_id, detection_index)
                    )

        by_detection: dict[int, list[float]] = {}
        by_track: dict[int, list[float]] = {}
        for cost, _, track_id, detection_index in candidates:
            by_detection.setdefault(detection_index, []).append(cost)
            by_track.setdefault(track_id, []).append(cost)
        ambiguous_detections = {
            detection_index
            for detection_index, costs in by_detection.items()
            if len(costs) >= 2
            and sorted(costs)[1] - sorted(costs)[0]
            < self.config.ambiguity_margin
        }
        ambiguous_tracks = {
            track_id
            for track_id, costs in by_track.items()
            if len(costs) >= 2
            and sorted(costs)[1] - sorted(costs)[0]
            < self.config.ambiguity_margin
        }
        # Ambiguity is a hard history boundary. Merely skipping the current
        # assignment would allow an old ID (and its pose filter state) to
        # reappear after two cats cross, potentially attaching pre-crossing
        # keypoints to the other animal. Retire every track participating in a
        # near-tied conflict so later detections start a fresh track epoch.
        retired_tracks = ambiguous_tracks | {
            track_id
            for _, _, track_id, detection_index in candidates
            if detection_index in ambiguous_detections
        }
        for track_id in retired_tracks:
            state.tracks.pop(track_id, None)

        matches: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, _, track_id, detection_index in sorted(candidates):
            if (
                detection_index in ambiguous_detections
                or track_id in retired_tracks
            ):
                continue
            if track_id in used_tracks or detection_index in matches:
                continue
            matches[detection_index] = track_id
            used_tracks.add(track_id)
        return matches

    def _update_track(
        self,
        track: _Track,
        raw_box: FloatBox,
        timestamp_seconds: float,
    ) -> FloatBox:
        elapsed = timestamp_seconds - track.last_timestamp
        old_raw_x, old_raw_y, _, _ = _center_size(track.raw_box)
        raw_x, raw_y, raw_width, raw_height = _center_size(raw_box)
        smooth_x, smooth_y, smooth_width, smooth_height = _center_size(
            track.smoothed_box
        )
        if elapsed <= 0.0:
            alpha = 1.0
        else:
            alpha = 1.0 - math.exp(
                -elapsed / self.config.bbox_smoothing_seconds
            )
            measured_velocity_x = (raw_x - old_raw_x) / elapsed
            measured_velocity_y = (raw_y - old_raw_y) / elapsed
            velocity_alpha = min(1.0, alpha * 2.0)
            track.velocity_x += velocity_alpha * (
                measured_velocity_x - track.velocity_x
            )
            track.velocity_y += velocity_alpha * (
                measured_velocity_y - track.velocity_y
            )

        smoothed_box = _from_center_size(
            smooth_x + alpha * (raw_x - smooth_x),
            smooth_y + alpha * (raw_y - smooth_y),
            smooth_width + alpha * (raw_width - smooth_width),
            smooth_height + alpha * (raw_height - smooth_height),
        )
        track.last_timestamp = timestamp_seconds
        track.raw_box = raw_box
        track.smoothed_box = smoothed_box
        return smoothed_box

    def _make_room(
        self,
        state: _SourceTracks,
        protected_track_ids: set[int],
    ) -> bool:
        if len(state.tracks) < self.config.max_tracks_per_source:
            return True
        evictable = [
            track
            for track in state.tracks.values()
            if track.track_id not in protected_track_ids
        ]
        if not evictable:
            # Every live track was matched in this unusually crowded frame.
            # Leave excess detections untracked instead of breaking an active
            # identity or exceeding the source memory bound.
            return False
        oldest = min(evictable, key=lambda item: (item.last_timestamp, item.track_id))
        del state.tracks[oldest.track_id]
        return True

    @staticmethod
    def _to_detection(
        raw_detection: Detection,
        box: FloatBox,
        frame_shape: tuple[int, ...],
    ) -> Detection:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        return Detection(
            class_id=raw_detection.class_id,
            class_name=raw_detection.class_name,
            confidence=raw_detection.confidence,
            xyxy=(x1, y1, x2, y2),
        )
