"""Track-aware pose quality gates and temporal stabilization."""

from __future__ import annotations

import math
from numbers import Real
from dataclasses import dataclass, field

from .models import Detection
from .pose import (
    AP10K_KEYPOINT_NAMES,
    AP10K_SKELETON,
    PoseKeypoint,
    PosePrediction,
)


_PAW_INDICES = frozenset({7, 10, 13, 16})
_JOINT_INDICES = frozenset({6, 9, 12, 15})
_ANCHOR_INDICES = frozenset({3, 4, 5, 8, 11, 14})
_SYMMETRY_GROUPS: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (
    ((0,), (1,)),
    ((5, 6, 7), (8, 9, 10)),
    ((11, 12, 13), (14, 15, 16)),
)
_EDGE_MAX_RATIO = {
    (0, 1): 0.35,
    (0, 2): 0.30,
    (1, 2): 0.30,
    (2, 3): 0.55,
    (3, 4): 1.05,
    (3, 5): 0.50,
    (3, 8): 0.50,
    (4, 11): 0.50,
    (4, 14): 0.50,
    (5, 6): 0.55,
    (8, 9): 0.55,
    (11, 12): 0.55,
    (14, 15): 0.55,
    (6, 7): 0.60,
    (9, 10): 0.60,
    (12, 13): 0.60,
    (15, 16): 0.60,
}
_MIN_POSE_SPREAD_RATIO = 0.05
_TIME_EPSILON_SECONDS = 1e-9


@dataclass(frozen=True)
class PoseFilterConfig:
    """Conservative quality and smoothing settings for online cat pose."""

    enabled: bool = False
    min_bbox_area_enter: int = 8192
    min_bbox_area_keep: int = 6144
    min_bbox_short_side: int = 64
    point_enter_confidence: float = 0.50
    point_keep_confidence: float = 0.30
    enter_min_points: int = 5
    enter_min_edges: int = 3
    enter_min_anchors: int = 1
    keep_min_points: int = 3
    keep_min_edges: int = 1
    max_bridge_gap_seconds: float = 0.15
    reset_gap_seconds: float = 0.30
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.5
    one_euro_derivative_cutoff: float = 1.0
    reference_dt_seconds: float = 0.05
    core_hold_frames: int = 3
    joint_hold_frames: int = 2
    paw_hold_frames: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("pose.quality.enabled must be a boolean")
        self._validate_integer(
            "pose.quality.min_bbox_area_enter",
            self.min_bbox_area_enter,
        )
        self._validate_integer(
            "pose.quality.min_bbox_area_keep",
            self.min_bbox_area_keep,
        )
        self._validate_integer(
            "pose.quality.min_bbox_short_side",
            self.min_bbox_short_side,
        )
        self._validate_integer(
            "pose.quality.enter_min_points",
            self.enter_min_points,
            maximum=len(AP10K_KEYPOINT_NAMES),
        )
        self._validate_integer(
            "pose.quality.enter_min_edges",
            self.enter_min_edges,
            maximum=len(AP10K_SKELETON),
        )
        self._validate_integer(
            "pose.quality.enter_min_anchors",
            self.enter_min_anchors,
            maximum=len(_ANCHOR_INDICES),
        )
        self._validate_integer(
            "pose.quality.keep_min_points",
            self.keep_min_points,
            maximum=len(AP10K_KEYPOINT_NAMES),
        )
        self._validate_integer(
            "pose.quality.keep_min_edges",
            self.keep_min_edges,
            maximum=len(AP10K_SKELETON),
        )
        for name, value in (
            ("pose.smoothing.core_hold_frames", self.core_hold_frames),
            ("pose.smoothing.joint_hold_frames", self.joint_hold_frames),
            ("pose.smoothing.paw_hold_frames", self.paw_hold_frames),
        ):
            self._validate_integer(name, value)
        if self.min_bbox_area_keep > self.min_bbox_area_enter:
            raise ValueError(
                "pose.quality.min_bbox_area_keep must be <= min_bbox_area_enter"
            )
        for name, value in (
            ("pose.quality.point_enter_confidence", self.point_enter_confidence),
            ("pose.quality.point_keep_confidence", self.point_keep_confidence),
        ):
            self._validate_real(name, value, minimum=0.0, maximum=1.0)
        if self.point_keep_confidence > self.point_enter_confidence:
            raise ValueError(
                "pose.quality.point_keep_confidence must be <= point_enter_confidence"
            )
        if self.keep_min_points > self.enter_min_points:
            raise ValueError(
                "pose.quality.keep_min_points must be <= enter_min_points"
            )
        if self.keep_min_edges > self.enter_min_edges:
            raise ValueError(
                "pose.quality.keep_min_edges must be <= enter_min_edges"
            )
        for name, value in (
            ("pose.smoothing.max_bridge_gap_seconds", self.max_bridge_gap_seconds),
            ("pose.smoothing.reset_gap_seconds", self.reset_gap_seconds),
            ("pose.smoothing.one_euro_min_cutoff", self.one_euro_min_cutoff),
            ("pose.smoothing.one_euro_derivative_cutoff", self.one_euro_derivative_cutoff),
            ("pose.smoothing.reference_dt_seconds", self.reference_dt_seconds),
        ):
            self._validate_real(name, value, minimum=0.0, minimum_inclusive=False)
        if self.max_bridge_gap_seconds > self.reset_gap_seconds:
            raise ValueError(
                "pose.smoothing.max_bridge_gap_seconds must be <= reset_gap_seconds"
            )
        self._validate_real(
            "pose.smoothing.one_euro_beta",
            self.one_euro_beta,
            minimum=0.0,
        )

    @staticmethod
    def _validate_integer(
        name: str,
        value: object,
        *,
        maximum: int | None = None,
    ) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or (maximum is not None and value > maximum)
        ):
            suffix = f" and <= {maximum}" if maximum is not None else ""
            raise ValueError(f"{name} must be an integer >= 0{suffix}")

    @staticmethod
    def _validate_real(
        name: str,
        value: object,
        *,
        minimum: float,
        maximum: float | None = None,
        minimum_inclusive: bool = True,
    ) -> None:
        valid = (
            not isinstance(value, bool)
            and isinstance(value, Real)
            and math.isfinite(float(value))
        )
        if valid:
            valid = value >= minimum if minimum_inclusive else value > minimum
        if valid and maximum is not None:
            valid = value <= maximum
        if not valid:
            if maximum is not None:
                rule = f"between {minimum:g} and {maximum:g}"
            elif minimum_inclusive:
                rule = f">= {minimum:g}"
            else:
                rule = f"> {minimum:g}"
            raise ValueError(f"{name} must be a finite number {rule}")


@dataclass(frozen=True)
class PoseFilterResult:
    """Filtered output plus bounded operational diagnostics."""

    prediction: PosePrediction | None
    status: str
    rejection_code: str = ""
    held_keypoints: int = 0
    dropped_keypoints: int = 0


@dataclass
class _OneEuro:
    min_cutoff: float
    beta: float
    derivative_cutoff: float
    timestamp: float | None = None
    raw_value: float = 0.0
    filtered_value: float = 0.0
    filtered_derivative: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, elapsed: float) -> float:
        ratio = 2.0 * math.pi * cutoff * elapsed
        return ratio / (ratio + 1.0)

    def reset(self, value: float, timestamp: float) -> float:
        self.timestamp = timestamp
        self.raw_value = value
        self.filtered_value = value
        self.filtered_derivative = 0.0
        return value

    def update(self, value: float, timestamp: float, weight: float = 1.0) -> float:
        if self.timestamp is None or timestamp <= self.timestamp:
            return self.reset(value, timestamp)
        elapsed = timestamp - self.timestamp
        derivative = (value - self.raw_value) / elapsed
        derivative_alpha = self._alpha(self.derivative_cutoff, elapsed)
        self.filtered_derivative += derivative_alpha * (
            derivative - self.filtered_derivative
        )
        cutoff = self.min_cutoff + self.beta * abs(self.filtered_derivative)
        value_alpha = self._alpha(cutoff, elapsed) * max(0.0, min(1.0, weight))
        self.filtered_value += value_alpha * (value - self.filtered_value)
        self.timestamp = timestamp
        self.raw_value = value
        return self.filtered_value


@dataclass
class _PointState:
    x_filter: _OneEuro
    y_filter: _OneEuro
    confidence: float
    hold_count: int = 0
    pending_u: float | None = None
    pending_v: float | None = None
    pending_count: int = 0

    @property
    def u(self) -> float:
        return self.x_filter.filtered_value

    @property
    def v(self) -> float:
        return self.y_filter.filtered_value


@dataclass
class _PoseTrackState:
    last_frame_timestamp: float
    last_valid_timestamp: float
    bbox: tuple[int, int, int, int]
    frame_size: tuple[int, int]
    points: list[_PointState | None] = field(
        default_factory=lambda: [None] * len(AP10K_KEYPOINT_NAMES)
    )


def _bbox_geometry(
    detection: Detection,
) -> tuple[float, float, float, float, float]:
    x1, y1, x2, y2 = detection.xyxy
    width = max(1.0, float(x2 - x1))
    height = max(1.0, float(y2 - y1))
    return float(x1), float(y1), width, height, math.hypot(width, height)


def _bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0,
        min(ay2, by2) - max(ay1, by1),
    )
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_center_step_ratio(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_x = (first[0] + first[2]) / 2.0
    first_y = (first[1] + first[3]) / 2.0
    second_x = (second[0] + second[2]) / 2.0
    second_y = (second[1] + second[3]) / 2.0
    diagonal = max(
        1.0,
        (
            math.hypot(first[2] - first[0], first[3] - first[1])
            + math.hypot(second[2] - second[0], second[3] - second[1])
        )
        / 2.0,
    )
    return math.hypot(second_x - first_x, second_y - first_y) / diagonal


def _bbox_size_shock(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    """Detect scale/aspect shocks which centre-only motion cannot see."""
    first_width = max(1, first[2] - first[0])
    first_height = max(1, first[3] - first[1])
    second_width = max(1, second[2] - second[0])
    second_height = max(1, second[3] - second[1])
    width_ratio = max(first_width / second_width, second_width / first_width)
    height_ratio = max(first_height / second_height, second_height / first_height)
    first_area = first_width * first_height
    second_area = second_width * second_height
    area_ratio = max(first_area / second_area, second_area / first_area)
    return width_ratio > 2.0 or height_ratio > 2.0 or area_ratio > 3.0


def _point_margin(index: int) -> float:
    if index in _PAW_INDICES:
        return 0.20
    if index in _JOINT_INDICES:
        return 0.15
    return 0.10


def _hard_step_ratio(index: int) -> float:
    if index in _PAW_INDICES:
        return 0.35
    if index in _JOINT_INDICES:
        return 0.25
    return 0.20


def _hold_limit(config: PoseFilterConfig, index: int) -> int:
    if index in _PAW_INDICES:
        return config.paw_hold_frames
    if index in _JOINT_INDICES:
        return config.joint_hold_frames
    return config.core_hold_frames


class TrackPoseFilter:
    """Reject implausible points and stabilize accepted poses per cat track."""

    def __init__(self, config: PoseFilterConfig | None = None) -> None:
        self.config = config or PoseFilterConfig()
        self._states: dict[tuple[str, int], _PoseTrackState] = {}

    def update(
        self,
        source: str,
        timestamp_seconds: float,
        track_id: int | None,
        detection: Detection,
        prediction: PosePrediction | None,
        frame_shape: tuple[int, ...],
    ) -> PoseFilterResult:
        """Filter one aligned pose without changing detector/event evidence."""
        if not self.config.enabled:
            return PoseFilterResult(
                prediction,
                "raw" if prediction is not None else "missing",
            )
        if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0.0:
            raise ValueError("pose filter timestamp must be finite and >= 0")
        if len(frame_shape) < 2 or frame_shape[0] <= 0 or frame_shape[1] <= 0:
            raise ValueError("pose filter frame shape must contain positive dimensions")
        self.prune_source(source, timestamp_seconds)
        if track_id is None:
            return self._stateless_gate(detection, prediction, frame_shape)

        key = (source, track_id)
        state = self._states.get(key)
        if state is not None and self._must_reset(
            state,
            timestamp_seconds,
            detection,
            frame_shape,
        ):
            del self._states[key]
            state = None

        points, valid, rejection = self._quality_gate(
            detection,
            prediction,
            frame_shape,
            existing=state is not None,
            state=state,
        )
        if rejection:
            if state is None:
                return PoseFilterResult(
                    None,
                    "rejected",
                    rejection,
                    dropped_keypoints=len(AP10K_KEYPOINT_NAMES),
                )
            if rejection == "pose_bbox_below_quality_floor":
                del self._states[key]
                return PoseFilterResult(
                    None,
                    "rejected",
                    rejection,
                    dropped_keypoints=len(AP10K_KEYPOINT_NAMES),
                )
            return self._hold_track(
                state,
                detection,
                timestamp_seconds,
                rejection,
                (frame_shape[1], frame_shape[0]),
            )

        if state is None:
            state = _PoseTrackState(
                last_frame_timestamp=timestamp_seconds,
                last_valid_timestamp=timestamp_seconds,
                bbox=detection.xyxy,
                frame_size=(frame_shape[1], frame_shape[0]),
            )
            self._states[key] = state
        else:
            points = self._correct_symmetry(points, valid, state, detection)
            # Relabeling changes which bones the observations belong to. Re-run
            # the limits so a corrected chain cannot bypass geometry checks.
            self._apply_bone_limits(points, valid, detection)

        held = 0
        dropped = 0
        observed = 0
        output: list[PoseKeypoint] = []
        frame_size = (frame_shape[1], frame_shape[0])
        x1, y1, width, height, diagonal = _bbox_geometry(detection)
        for index, point in enumerate(points):
            point_state = state.points[index]
            accepted = valid[index]
            u = (point.x - x1) / width
            v = (point.y - y1) / height
            point_timestamp = (
                point_state.x_filter.timestamp if point_state is not None else None
            )
            point_elapsed = (
                max(0.0, timestamp_seconds - point_timestamp)
                if point_timestamp is not None
                else 0.0
            )
            if accepted and point_state is not None and point_elapsed > 0.0:
                previous_x = x1 + point_state.u * width
                previous_y = y1 + point_state.v * height
                step_ratio = math.hypot(
                    point.x - previous_x,
                    point.y - previous_y,
                ) / diagonal
                scale = max(
                    0.5,
                    min(
                        3.0,
                        point_elapsed / self.config.reference_dt_seconds,
                    ),
                )
                if step_ratio > _hard_step_ratio(index) * scale:
                    accepted = self._allow_reacquisition(
                        point_state,
                        u,
                        v,
                        point.confidence,
                        width,
                        height,
                        diagonal,
                    )
                    if accepted:
                        point_state.x_filter.reset(u, timestamp_seconds)
                        point_state.y_filter.reset(v, timestamp_seconds)

            if accepted:
                observed += 1
                if point_state is None:
                    point_state = self._new_point_state(
                        u,
                        v,
                        point.confidence,
                        timestamp_seconds,
                    )
                    state.points[index] = point_state
                else:
                    confidence_weight = max(
                        0.25,
                        min(
                            1.0,
                            (
                                point.confidence
                                - self.config.point_keep_confidence
                            )
                            / max(
                                1e-6,
                                self.config.point_enter_confidence
                                - self.config.point_keep_confidence,
                            ),
                        ),
                    )
                    point_state.x_filter.update(
                        u,
                        timestamp_seconds,
                        confidence_weight,
                    )
                    point_state.y_filter.update(
                        v,
                        timestamp_seconds,
                        confidence_weight,
                    )
                    point_state.confidence = (
                        0.7 * point_state.confidence
                        + 0.3 * point.confidence
                    )
                    point_state.hold_count = 0
                    point_state.pending_u = None
                    point_state.pending_v = None
                    point_state.pending_count = 0
                output.append(
                    self._state_keypoint(
                        index,
                        point_state,
                        detection,
                        frame_size,
                    )
                )
            elif point_state is not None and point_state.hold_count < _hold_limit(
                self.config,
                index,
            ):
                point_state.hold_count += 1
                point_state.confidence *= 0.65 if index in _PAW_INDICES else 0.75
                held += 1
                output.append(
                    self._state_keypoint(
                        index,
                        point_state,
                        detection,
                        frame_size,
                    )
                )
            else:
                state.points[index] = None
                dropped += 1
                output.append(
                    self._bounded_keypoint(
                        index,
                        point.x,
                        point.y,
                        0.0,
                        frame_size,
                    )
                )

        state.last_frame_timestamp = timestamp_seconds
        if observed:
            state.last_valid_timestamp = timestamp_seconds
        state.bbox = detection.xyxy
        state.frame_size = frame_size
        prediction_out = self._prediction(detection, output)
        visible = prediction_out.visible_keypoint_count(
            self.config.point_keep_confidence
        )
        if visible < self.config.keep_min_points:
            return PoseFilterResult(
                None,
                "rejected",
                "insufficient_filtered_points",
                held_keypoints=held,
                dropped_keypoints=dropped,
            )
        return PoseFilterResult(
            prediction_out,
            "held" if held else "accepted",
            held_keypoints=held,
            dropped_keypoints=dropped,
        )

    def prune_source(self, source: str, timestamp_seconds: float) -> None:
        """Bound pose state for long-running sources."""
        expired = [
            key
            for key, state in self._states.items()
            if key[0] == source
            and (
                timestamp_seconds < state.last_frame_timestamp
                or timestamp_seconds - state.last_valid_timestamp
                > self.config.reset_gap_seconds + _TIME_EPSILON_SECONDS
            )
        ]
        for key in expired:
            del self._states[key]

    def clear_source(self, source: str) -> None:
        """Forget every pose state belonging to a completed source."""
        for key in [key for key in self._states if key[0] == source]:
            del self._states[key]

    def _must_reset(
        self,
        state: _PoseTrackState,
        timestamp_seconds: float,
        detection: Detection,
        frame_shape: tuple[int, ...],
    ) -> bool:
        frame_gap = timestamp_seconds - state.last_frame_timestamp
        valid_gap = timestamp_seconds - state.last_valid_timestamp
        return (
            frame_gap < 0.0
            or valid_gap < 0.0
            or valid_gap
            > self.config.reset_gap_seconds + _TIME_EPSILON_SECONDS
            or state.frame_size != (frame_shape[1], frame_shape[0])
            or _bbox_size_shock(state.bbox, detection.xyxy)
            or (
                _bbox_iou(state.bbox, detection.xyxy) < 0.30
                and _bbox_center_step_ratio(state.bbox, detection.xyxy) > 0.25
            )
        )

    def _stateless_gate(
        self,
        detection: Detection,
        prediction: PosePrediction | None,
        frame_shape: tuple[int, ...],
    ) -> PoseFilterResult:
        points, valid, rejection = self._quality_gate(
            detection,
            prediction,
            frame_shape,
            existing=False,
            state=None,
        )
        if rejection:
            return PoseFilterResult(
                None,
                "rejected",
                rejection,
                dropped_keypoints=len(AP10K_KEYPOINT_NAMES),
            )
        output = [
            self._bounded_keypoint(
                index,
                point.x,
                point.y,
                point.confidence if valid[index] else 0.0,
                (frame_shape[1], frame_shape[0]),
            )
            for index, point in enumerate(points)
        ]
        return PoseFilterResult(
            self._prediction(detection, output),
            "accepted",
            dropped_keypoints=sum(not item for item in valid),
        )

    def _quality_gate(
        self,
        detection: Detection,
        prediction: PosePrediction | None,
        frame_shape: tuple[int, ...],
        *,
        existing: bool,
        state: _PoseTrackState | None,
    ) -> tuple[list[PoseKeypoint], list[bool], str]:
        area_threshold = (
            self.config.min_bbox_area_keep
            if existing
            else self.config.min_bbox_area_enter
        )
        width = detection.xyxy[2] - detection.xyxy[0]
        height = detection.xyxy[3] - detection.xyxy[1]
        if (
            detection.area < area_threshold
            or min(width, height) < self.config.min_bbox_short_side
        ):
            points = (
                list(prediction.keypoints)
                if prediction is not None
                and len(prediction.keypoints) == len(AP10K_KEYPOINT_NAMES)
                else self._empty_points()
            )
            return points, [False] * 17, "pose_bbox_below_quality_floor"
        if prediction is None:
            return self._empty_points(), [False] * 17, "missing_pose"
        points = list(prediction.keypoints)
        if len(points) != len(AP10K_KEYPOINT_NAMES):
            return self._empty_points(), [False] * 17, "invalid_keypoint_count"
        valid = [
            math.isfinite(point.confidence)
            and 0.0 <= point.confidence <= 1.0
            and point.confidence
            >= (
                self.config.point_keep_confidence
                if existing
                and state is not None
                and state.points[index] is not None
                else self.config.point_enter_confidence
            )
            and self._point_in_bounds(index, point, detection, frame_shape)
            for index, point in enumerate(points)
        ]
        self._apply_bone_limits(points, valid, detection)
        if not existing:
            self._drop_disconnected_components(valid)
        point_count = sum(valid)
        edge_count = sum(valid[first] and valid[second] for first, second in AP10K_SKELETON)
        anchor_count = sum(valid[index] for index in _ANCHOR_INDICES)
        minimum_points = (
            self.config.keep_min_points if existing else self.config.enter_min_points
        )
        minimum_edges = (
            self.config.keep_min_edges if existing else self.config.enter_min_edges
        )
        minimum_anchors = 0 if existing else self.config.enter_min_anchors
        frame_area = frame_shape[0] * frame_shape[1]
        touches_edge = (
            detection.xyxy[0] == 0
            or detection.xyxy[1] == 0
            or detection.xyxy[2] == frame_shape[1]
            or detection.xyxy[3] == frame_shape[0]
        )
        if not existing and touches_edge and detection.area >= 0.15 * frame_area:
            minimum_points = max(minimum_points, 7)
            minimum_edges = max(minimum_edges, 4)
            minimum_anchors = max(minimum_anchors, 2)
        if point_count < minimum_points:
            return points, valid, "insufficient_reliable_keypoints"
        if edge_count < minimum_edges:
            return points, valid, "insufficient_valid_skeleton_edges"
        if anchor_count < minimum_anchors:
            return points, valid, "missing_pose_anchor"
        if not self._has_minimum_spread(points, valid, detection):
            return points, valid, "degenerate_pose_geometry"
        return points, valid, ""

    @staticmethod
    def _empty_points() -> list[PoseKeypoint]:
        return [
            PoseKeypoint(index, name, 0.0, 0.0, 0.0)
            for index, name in enumerate(AP10K_KEYPOINT_NAMES)
        ]

    @staticmethod
    def _point_in_bounds(
        index: int,
        point: PoseKeypoint,
        detection: Detection,
        frame_shape: tuple[int, ...],
    ) -> bool:
        if (
            not math.isfinite(point.x)
            or not math.isfinite(point.y)
            or point.x < 0.0
            or point.y < 0.0
            or point.x > frame_shape[1] - 1
            or point.y > frame_shape[0] - 1
        ):
            return False
        x1, y1, width, height, _ = _bbox_geometry(detection)
        margin = _point_margin(index)
        return (
            x1 - margin * width <= point.x <= x1 + (1.0 + margin) * width
            and y1 - margin * height <= point.y <= y1 + (1.0 + margin) * height
        )

    @staticmethod
    def _apply_bone_limits(
        points: list[PoseKeypoint],
        valid: list[bool],
        detection: Detection,
    ) -> None:
        diagonal = _bbox_geometry(detection)[4]
        for first, second in AP10K_SKELETON:
            if not (valid[first] and valid[second]):
                continue
            length_ratio = math.hypot(
                points[first].x - points[second].x,
                points[first].y - points[second].y,
            ) / diagonal
            if length_ratio <= _EDGE_MAX_RATIO[(first, second)]:
                continue
            if points[first].confidence < points[second].confidence:
                valid[first] = False
            else:
                # Skeleton edges are declared proximal-to-distal except for the
                # small face triangle. On near ties, dropping the latter point
                # is the conservative choice for long limb hallucinations.
                valid[second] = False

    @staticmethod
    def _has_minimum_spread(
        points: list[PoseKeypoint],
        valid: list[bool],
        detection: Detection,
    ) -> bool:
        indices = [index for index, is_valid in enumerate(valid) if is_valid]
        diagonal = _bbox_geometry(detection)[4]
        maximum_distance = max(
            (
                math.hypot(
                    points[first].x - points[second].x,
                    points[first].y - points[second].y,
                )
                for offset, first in enumerate(indices)
                for second in indices[offset + 1 :]
            ),
            default=0.0,
        )
        return maximum_distance / diagonal >= _MIN_POSE_SPREAD_RATIO

    @staticmethod
    def _drop_disconnected_components(valid: list[bool]) -> None:
        adjacency: dict[int, set[int]] = {index: set() for index in range(17)}
        for first, second in AP10K_SKELETON:
            if valid[first] and valid[second]:
                adjacency[first].add(second)
                adjacency[second].add(first)
        components: list[set[int]] = []
        visited: set[int] = set()
        for start in range(17):
            if not valid[start] or start in visited:
                continue
            stack = [start]
            component: set[int] = set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                visited.add(current)
                stack.extend(adjacency[current] - component)
            if len(component) >= 2 and component & _ANCHOR_INDICES:
                components.append(component)
        # A new track must be supported by one coherent skeleton component.
        # Summing several isolated limb fragments can otherwise satisfy the
        # point/edge thresholds without establishing a plausible cat pose.
        retained = max(
            components,
            key=lambda component: (
                len(component),
                len(component & _ANCHOR_INDICES),
                -min(component),
            ),
            default=set(),
        )
        for index in range(17):
            valid[index] = index in retained

    def _correct_symmetry(
        self,
        points: list[PoseKeypoint],
        valid: list[bool],
        state: _PoseTrackState,
        detection: Detection,
    ) -> list[PoseKeypoint]:
        corrected = list(points)
        x1, y1, width, height, diagonal = _bbox_geometry(detection)
        for left_group, right_group in _SYMMETRY_GROUPS:
            identity_cost = 0.0
            swap_cost = 0.0
            pair_comparisons = 0
            for left, right in zip(left_group, right_group):
                left_state = state.points[left]
                right_state = state.points[right]
                if (
                    not valid[left]
                    or not valid[right]
                    or left_state is None
                    or right_state is None
                ):
                    continue
                left_previous = (x1 + left_state.u * width, y1 + left_state.v * height)
                right_previous = (x1 + right_state.u * width, y1 + right_state.v * height)
                identity_cost += math.hypot(
                    points[left].x - left_previous[0],
                    points[left].y - left_previous[1],
                ) + math.hypot(
                    points[right].x - right_previous[0],
                    points[right].y - right_previous[1],
                )
                swap_cost += math.hypot(
                    points[left].x - right_previous[0],
                    points[left].y - right_previous[1],
                ) + math.hypot(
                    points[right].x - left_previous[0],
                    points[right].y - left_previous[1],
                )
                pair_comparisons += 1
            # A whole chain is relabelled only with agreement from at least two
            # bilateral pairs. One shoulder/hip pair is too weak, and the
            # single eye pair must never relabel unrelated points by itself.
            if pair_comparisons < 2:
                continue
            comparisons = pair_comparisons * 2
            identity_average = identity_cost / comparisons / diagonal
            swap_average = swap_cost / comparisons / diagonal
            if (
                swap_average <= 0.5 * identity_average
                and identity_average - swap_average >= 0.08
            ):
                for left, right in zip(left_group, right_group):
                    corrected[left], corrected[right] = (
                        corrected[right],
                        corrected[left],
                    )
                    valid[left], valid[right] = valid[right], valid[left]
        return [
            PoseKeypoint(
                index,
                AP10K_KEYPOINT_NAMES[index],
                point.x,
                point.y,
                point.confidence,
            )
            for index, point in enumerate(corrected)
        ]

    @staticmethod
    def _allow_reacquisition(
        state: _PointState,
        u: float,
        v: float,
        confidence: float,
        width: float,
        height: float,
        diagonal: float,
    ) -> bool:
        if confidence < 0.60:
            state.pending_u = None
            state.pending_v = None
            state.pending_count = 0
            return False
        if state.pending_u is None or state.pending_v is None:
            state.pending_u = u
            state.pending_v = v
            state.pending_count = 1
            return False
        pending_distance = math.hypot(
            (u - state.pending_u) * width,
            (v - state.pending_v) * height,
        ) / diagonal
        if pending_distance <= 0.08:
            state.pending_count += 1
        else:
            state.pending_u = u
            state.pending_v = v
            state.pending_count = 1
        return state.pending_count >= 2

    def _new_point_state(
        self,
        u: float,
        v: float,
        confidence: float,
        timestamp_seconds: float,
    ) -> _PointState:
        x_filter = _OneEuro(
            self.config.one_euro_min_cutoff,
            self.config.one_euro_beta,
            self.config.one_euro_derivative_cutoff,
        )
        y_filter = _OneEuro(
            self.config.one_euro_min_cutoff,
            self.config.one_euro_beta,
            self.config.one_euro_derivative_cutoff,
        )
        x_filter.reset(u, timestamp_seconds)
        y_filter.reset(v, timestamp_seconds)
        return _PointState(x_filter, y_filter, confidence)

    def _hold_track(
        self,
        state: _PoseTrackState,
        detection: Detection,
        timestamp_seconds: float,
        rejection_code: str,
        frame_size: tuple[int, int],
    ) -> PoseFilterResult:
        bridge_gap = timestamp_seconds - state.last_valid_timestamp
        if (
            bridge_gap
            > self.config.max_bridge_gap_seconds + _TIME_EPSILON_SECONDS
        ):
            state.last_frame_timestamp = timestamp_seconds
            state.bbox = detection.xyxy
            state.frame_size = frame_size
            return PoseFilterResult(
                None,
                "rejected",
                rejection_code,
                dropped_keypoints=len(AP10K_KEYPOINT_NAMES),
            )
        output: list[PoseKeypoint] = []
        held = 0
        for index, point_state in enumerate(state.points):
            if point_state is None or point_state.hold_count >= _hold_limit(
                self.config,
                index,
            ):
                state.points[index] = None
                output.append(
                    PoseKeypoint(index, AP10K_KEYPOINT_NAMES[index], 0.0, 0.0, 0.0)
                )
                continue
            point_state.hold_count += 1
            point_state.confidence *= 0.65 if index in _PAW_INDICES else 0.75
            output.append(
                self._state_keypoint(
                    index,
                    point_state,
                    detection,
                    frame_size,
                )
            )
            held += 1
        state.last_frame_timestamp = timestamp_seconds
        state.bbox = detection.xyxy
        state.frame_size = frame_size
        prediction = self._prediction(detection, output)
        if (
            prediction.visible_keypoint_count(self.config.point_keep_confidence)
            < self.config.keep_min_points
        ):
            return PoseFilterResult(
                None,
                "rejected",
                rejection_code,
                held_keypoints=held,
                dropped_keypoints=len(AP10K_KEYPOINT_NAMES) - held,
            )
        return PoseFilterResult(
            prediction,
            "held",
            rejection_code,
            held_keypoints=held,
            dropped_keypoints=len(AP10K_KEYPOINT_NAMES) - held,
        )

    @staticmethod
    def _state_keypoint(
        index: int,
        state: _PointState,
        detection: Detection,
        frame_size: tuple[int, int],
    ) -> PoseKeypoint:
        x1, y1, width, height, _ = _bbox_geometry(detection)
        return TrackPoseFilter._bounded_keypoint(
            index,
            x1 + state.u * width,
            y1 + state.v * height,
            state.confidence,
            frame_size,
        )

    @staticmethod
    def _bounded_keypoint(
        index: int,
        x: float,
        y: float,
        confidence: float,
        frame_size: tuple[int, int],
    ) -> PoseKeypoint:
        width, height = frame_size
        coordinates_finite = math.isfinite(x) and math.isfinite(y)
        in_frame = (
            coordinates_finite
            and 0.0 <= x <= width - 1
            and 0.0 <= y <= height - 1
        )
        if not coordinates_finite:
            x = 0.0
            y = 0.0
        else:
            x = max(0.0, min(float(width - 1), x))
            y = max(0.0, min(float(height - 1), y))
        if not math.isfinite(confidence) or not in_frame:
            confidence = 0.0
        return PoseKeypoint(
            index,
            AP10K_KEYPOINT_NAMES[index],
            x,
            y,
            max(0.0, min(1.0, confidence)),
        )

    @staticmethod
    def _prediction(
        detection: Detection,
        points: list[PoseKeypoint],
    ) -> PosePrediction:
        mean_confidence = sum(point.confidence for point in points) / len(points)
        return PosePrediction(
            detection.xyxy,
            tuple(points),
            mean_confidence,
        )
