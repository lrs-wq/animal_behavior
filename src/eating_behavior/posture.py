"""Pure cat-posture classification and change-state tracking.

The classifier consumes only an AP-10K pose that has already passed the
project's pose quality filter.  It deliberately returns ``None`` for missing,
held, rejected, or geometrically unsupported observations: uncertainty must
not erase a cat's last confirmed posture.

``PostureStateMachine`` separates a confirmed posture from the transition that
is merely due for persistence.  Callers acknowledge a transition only after
the event sink succeeds, so a failed write can be retried without swallowing a
state change.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace
from enum import Enum

from .pose import AP10K_KEYPOINT_NAMES, PoseKeypoint, PosePrediction


class CatPosture(str, Enum):
    """Stable machine-facing posture values."""

    EATING = "eating"
    LYING = "lying"
    STANDING = "standing"
    CROUCHING = "crouching"

    @property
    def chinese_label(self) -> str:
        """Return the user-facing Simplified Chinese label."""
        return _CHINESE_LABELS[self]


_CHINESE_LABELS: dict[CatPosture, str] = {
    CatPosture.EATING: "吃饭",
    CatPosture.LYING: "趴卧",
    CatPosture.STANDING: "站立",
    CatPosture.CROUCHING: "蹲姿",
}

# Public read-only-by-convention copy for renderers that prefer a mapping.
CAT_POSTURE_CHINESE_LABELS = dict(_CHINESE_LABELS)


@dataclass(frozen=True)
class PostureConfig:
    """Quality, geometry, and temporal confirmation thresholds."""

    enabled: bool = False
    min_keypoint_confidence: float = 0.35
    min_feeding_confidence: float = 0.50
    min_visible_keypoints: int = 5
    min_body_length_ratio: float = 0.12
    min_supporting_paws: int = 2
    min_complete_leg_chains: int = 1
    lying_max_clearance_ratio: float = 0.10
    crouching_min_clearance_ratio: float = 0.10
    crouching_max_clearance_ratio: float = 0.22
    standing_min_front_drop_ratio: float = 0.18
    standing_min_hind_drop_ratio: float = 0.15
    standing_min_clearance_ratio: float = 0.20
    standing_min_leg_straightness: float = 0.78
    transition_confirm_seconds: float = 0.50
    min_confirmed_observations: int = 3
    max_anonymous_subjects_per_source: int = 64

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("posture.enabled must be a boolean")

        for name, value in (
            ("posture.min_keypoint_confidence", self.min_keypoint_confidence),
            ("posture.min_feeding_confidence", self.min_feeding_confidence),
            (
                "posture.standing_min_leg_straightness",
                self.standing_min_leg_straightness,
            ),
        ):
            self._validate_real(name, value, minimum=0.0, maximum=1.0)

        for name, value in (
            ("posture.min_body_length_ratio", self.min_body_length_ratio),
            (
                "posture.lying_max_clearance_ratio",
                self.lying_max_clearance_ratio,
            ),
            (
                "posture.crouching_min_clearance_ratio",
                self.crouching_min_clearance_ratio,
            ),
            (
                "posture.crouching_max_clearance_ratio",
                self.crouching_max_clearance_ratio,
            ),
            (
                "posture.standing_min_front_drop_ratio",
                self.standing_min_front_drop_ratio,
            ),
            (
                "posture.standing_min_hind_drop_ratio",
                self.standing_min_hind_drop_ratio,
            ),
            (
                "posture.standing_min_clearance_ratio",
                self.standing_min_clearance_ratio,
            ),
        ):
            self._validate_real(name, value, minimum=0.0, maximum=2.0)

        self._validate_real(
            "posture.transition_confirm_seconds",
            self.transition_confirm_seconds,
            minimum=0.0,
        )
        self._validate_integer(
            "posture.min_visible_keypoints",
            self.min_visible_keypoints,
            minimum=1,
            maximum=len(AP10K_KEYPOINT_NAMES),
        )
        self._validate_integer(
            "posture.min_supporting_paws",
            self.min_supporting_paws,
            minimum=2,
            maximum=4,
        )
        self._validate_integer(
            "posture.min_complete_leg_chains",
            self.min_complete_leg_chains,
            minimum=1,
            maximum=4,
        )
        self._validate_integer(
            "posture.min_confirmed_observations",
            self.min_confirmed_observations,
            minimum=1,
        )
        self._validate_integer(
            "posture.max_anonymous_subjects_per_source",
            self.max_anonymous_subjects_per_source,
            minimum=1,
        )

        if self.min_body_length_ratio == 0.0:
            raise ValueError("posture.min_body_length_ratio must be > 0")
        if self.crouching_min_clearance_ratio > self.crouching_max_clearance_ratio:
            raise ValueError(
                "posture.crouching_min_clearance_ratio must be <= "
                "crouching_max_clearance_ratio"
            )
        if self.lying_max_clearance_ratio > self.crouching_min_clearance_ratio:
            raise ValueError(
                "posture.lying_max_clearance_ratio must be <= "
                "crouching_min_clearance_ratio"
            )
        if self.lying_max_clearance_ratio > self.standing_min_clearance_ratio:
            raise ValueError(
                "posture.lying_max_clearance_ratio must be <= "
                "standing_min_clearance_ratio"
            )
        if self.crouching_min_clearance_ratio > self.standing_min_clearance_ratio:
            raise ValueError(
                "posture.crouching_min_clearance_ratio must be <= "
                "standing_min_clearance_ratio"
            )

    @staticmethod
    def _validate_real(
        name: str,
        value: object,
        *,
        minimum: float,
        maximum: float | None = None,
    ) -> None:
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= minimum
        )
        if valid and maximum is not None:
            valid = value <= maximum
        if not valid:
            rule = (
                f"between {minimum:g} and {maximum:g}"
                if maximum is not None
                else f">= {minimum:g}"
            )
            raise ValueError(f"{name} must be a finite number {rule}")

    @staticmethod
    def _validate_integer(
        name: str,
        value: object,
        *,
        minimum: int,
        maximum: int | None = None,
    ) -> None:
        valid = (
            not isinstance(value, bool) and isinstance(value, int) and value >= minimum
        )
        if valid and maximum is not None:
            valid = value <= maximum
        if not valid:
            rule = (
                f"between {minimum} and {maximum}"
                if maximum is not None
                else f">= {minimum}"
            )
            raise ValueError(f"{name} must be an integer {rule}")


@dataclass(frozen=True)
class PostureObservation:
    """One trustworthy per-frame posture observation."""

    posture: CatPosture
    confidence: float
    evidence_source: str
    features: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.posture, CatPosture):
            raise ValueError("posture must be a CatPosture")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("posture confidence must be between 0 and 1")
        if (
            not isinstance(self.evidence_source, str)
            or not self.evidence_source.strip()
        ):
            raise ValueError("posture evidence_source must not be blank")
        seen: set[str] = set()
        for name, value in self.features:
            if not isinstance(name, str) or not name.strip() or name in seen:
                raise ValueError("posture feature names must be unique and non-blank")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("posture feature values must be finite numbers")
            seen.add(name)

    @property
    def chinese_label(self) -> str:
        """Return the user-facing Simplified Chinese label."""
        return self.posture.chinese_label


@dataclass(frozen=True)
class PostureTransition:
    """A confirmed but not yet acknowledged posture change."""

    subject_key: str
    previous_posture: CatPosture | None
    posture: CatPosture
    confidence: float
    timestamp_seconds: float
    candidate_started_seconds: float
    confirmed_observations: int
    transition_sequence: int
    reason: str
    evidence_source: str
    features: tuple[tuple[str, float], ...] = ()

    @property
    def chinese_label(self) -> str:
        """Return the new posture's Simplified Chinese label."""
        return self.posture.chinese_label


_FRONT_ANCHORS = (3, 5, 8)
_REAR_ANCHORS = (4, 11, 14)
_HEAD_POINTS = (0, 1, 2)
_PAWS = (7, 10, 13, 16)
_FRONT_SUPPORTS = ((5, 7), (8, 10))
_HIND_SUPPORTS = ((11, 13), (14, 16))
_LEG_CHAINS = ((5, 6, 7), (8, 9, 10), (11, 12, 13), (14, 15, 16))


class CatPostureClassifier:
    """Conservative fixed-camera geometry classifier for filtered AP-10K poses."""

    def __init__(self, config: PostureConfig | None = None) -> None:
        self.config = config or PostureConfig()

    def classify(
        self,
        prediction: PosePrediction | None,
        pose_status: str,
        feeding_confidence: float = 0.0,
    ) -> PostureObservation | None:
        """Return a supported posture, or ``None`` when evidence is uncertain.

        ``feeding_confidence`` must describe an unambiguous feeding-head to cat
        association.  A lowered head without that association is never enough
        to label the cat as eating.
        """
        feeding_score = self._feeding_score(feeding_confidence)
        if not self.config.enabled:
            return None
        if pose_status not in {"accepted", "raw"} or prediction is None:
            return None

        x1, y1, x2, y2 = prediction.cat_xyxy
        width = float(x2 - x1)
        height = float(y2 - y1)
        diagonal = math.hypot(width, height)
        if width <= 0.0 or height <= 0.0 or diagonal <= 0.0:
            return None
        if len(prediction.keypoints) != len(AP10K_KEYPOINT_NAMES):
            return None

        reliable = self._reliable_points(prediction, width, height)
        if len(reliable) < self.config.min_visible_keypoints:
            return None

        head = [reliable[index] for index in _HEAD_POINTS if index in reliable]
        front_points = [
            reliable[index] for index in _FRONT_ANCHORS if index in reliable
        ]
        if (
            feeding_score > 0.0
            and feeding_score >= self.config.min_feeding_confidence
            and head
            and front_points
        ):
            head_quality = statistics.median(
                point.confidence for point in (*head, *front_points)
            )
            confidence = math.sqrt(feeding_score * head_quality)
            return self._observation(
                CatPosture.EATING,
                confidence,
                "feeding_head_association",
                reliable,
                feeding_confidence=feeding_score,
            )

        rear_points = [reliable[index] for index in _REAR_ANCHORS if index in reliable]
        if not front_points or not rear_points:
            return None
        front_x, front_y = self._median_coordinate(front_points)
        rear_x, rear_y = self._median_coordinate(rear_points)
        body_length_ratio = (
            math.hypot(
                rear_x - front_x,
                rear_y - front_y,
            )
            / diagonal
        )
        if body_length_ratio < self.config.min_body_length_ratio:
            return None

        front_drops = self._support_drops(reliable, _FRONT_SUPPORTS, height)
        hind_drops = self._support_drops(reliable, _HIND_SUPPORTS, height)
        visible_paws = [reliable[index] for index in _PAWS if index in reliable]
        if (
            not front_drops
            or not hind_drops
            or len(visible_paws) < self.config.min_supporting_paws
        ):
            return None

        torso_points = [*front_points, *rear_points]
        torso_y = statistics.median(point.y for point in torso_points)
        clearance_ratio = (
            statistics.median(point.y for point in visible_paws) - torso_y
        ) / height
        front_drop_ratio = statistics.median(front_drops)
        hind_drop_ratio = statistics.median(hind_drops)
        straightness = self._leg_straightness(reliable)
        complete_leg_chains = len(straightness)
        median_straightness = statistics.median(straightness) if straightness else 0.0
        quality = statistics.median(point.confidence for point in reliable.values())
        common_features = {
            "feeding_confidence": feeding_score,
            "body_length_ratio": body_length_ratio,
            "front_drop_ratio": front_drop_ratio,
            "hind_drop_ratio": hind_drop_ratio,
            "paw_clearance_ratio": clearance_ratio,
            "leg_straightness": median_straightness,
            "complete_leg_chains": float(complete_leg_chains),
        }

        if 0.0 <= clearance_ratio <= self.config.lying_max_clearance_ratio:
            margin = self._bounded_margin(
                self.config.lying_max_clearance_ratio - clearance_ratio,
                max(0.05, self.config.lying_max_clearance_ratio),
            )
            return self._observation(
                CatPosture.LYING,
                self._geometry_confidence(quality, margin),
                "pose_geometry",
                reliable,
                **common_features,
            )

        if (
            complete_leg_chains >= self.config.min_complete_leg_chains
            and front_drop_ratio >= self.config.standing_min_front_drop_ratio
            and hind_drop_ratio >= self.config.standing_min_hind_drop_ratio
            and clearance_ratio >= self.config.standing_min_clearance_ratio
            and median_straightness >= self.config.standing_min_leg_straightness
        ):
            margin = min(
                self._bounded_margin(
                    front_drop_ratio - self.config.standing_min_front_drop_ratio,
                    0.10,
                ),
                self._bounded_margin(
                    hind_drop_ratio - self.config.standing_min_hind_drop_ratio,
                    0.10,
                ),
                self._bounded_margin(
                    clearance_ratio - self.config.standing_min_clearance_ratio,
                    0.10,
                ),
                self._bounded_margin(
                    median_straightness - self.config.standing_min_leg_straightness,
                    0.20,
                ),
            )
            return self._observation(
                CatPosture.STANDING,
                self._geometry_confidence(quality, margin),
                "pose_geometry",
                reliable,
                **common_features,
            )

        if (
            complete_leg_chains >= self.config.min_complete_leg_chains
            and self.config.crouching_min_clearance_ratio
            <= clearance_ratio
            <= self.config.crouching_max_clearance_ratio
        ):
            distance_to_boundary = min(
                clearance_ratio - self.config.crouching_min_clearance_ratio,
                self.config.crouching_max_clearance_ratio - clearance_ratio,
            )
            half_range = max(
                0.05,
                (
                    self.config.crouching_max_clearance_ratio
                    - self.config.crouching_min_clearance_ratio
                )
                / 2.0,
            )
            return self._observation(
                CatPosture.CROUCHING,
                self._geometry_confidence(
                    quality,
                    self._bounded_margin(distance_to_boundary, half_range),
                ),
                "pose_geometry",
                reliable,
                **common_features,
            )
        return None

    def _reliable_points(
        self,
        prediction: PosePrediction,
        width: float,
        height: float,
    ) -> dict[int, PoseKeypoint]:
        x1, y1, x2, y2 = prediction.cat_xyxy
        margin_x = 0.20 * width
        margin_y = 0.20 * height
        reliable: dict[int, PoseKeypoint] = {}
        for expected_index, point in enumerate(prediction.keypoints):
            if (
                point.index != expected_index
                or point.name != AP10K_KEYPOINT_NAMES[expected_index]
                or not math.isfinite(point.x)
                or not math.isfinite(point.y)
                or not math.isfinite(point.confidence)
                or not 0.0 <= point.confidence <= 1.0
                or point.confidence < self.config.min_keypoint_confidence
                or not x1 - margin_x <= point.x <= x2 + margin_x
                or not y1 - margin_y <= point.y <= y2 + margin_y
            ):
                continue
            reliable[expected_index] = point
        return reliable

    @staticmethod
    def _support_drops(
        points: dict[int, PoseKeypoint],
        supports: tuple[tuple[int, int], ...],
        height: float,
    ) -> list[float]:
        return [
            (points[paw].y - points[proximal].y) / height
            for proximal, paw in supports
            if proximal in points and paw in points
        ]

    @staticmethod
    def _leg_straightness(points: dict[int, PoseKeypoint]) -> list[float]:
        values: list[float] = []
        for proximal, joint, paw in _LEG_CHAINS:
            if not all(index in points for index in (proximal, joint, paw)):
                continue
            first = points[proximal]
            middle = points[joint]
            last = points[paw]
            direct = math.hypot(last.x - first.x, last.y - first.y)
            segments = math.hypot(
                middle.x - first.x,
                middle.y - first.y,
            ) + math.hypot(last.x - middle.x, last.y - middle.y)
            if segments > 1e-9:
                values.append(max(0.0, min(1.0, direct / segments)))
        return values

    @staticmethod
    def _median_coordinate(points: list[PoseKeypoint]) -> tuple[float, float]:
        return (
            statistics.median(point.x for point in points),
            statistics.median(point.y for point in points),
        )

    @staticmethod
    def _bounded_margin(distance: float, scale: float) -> float:
        return max(0.0, min(1.0, distance / max(scale, 1e-9)))

    @staticmethod
    def _geometry_confidence(quality: float, margin: float) -> float:
        # Geometry is a fixed-camera heuristic, so do not expose model-like
        # certainty even when every keypoint has a very high score.
        return min(0.85, quality * (0.70 + 0.30 * margin))

    @staticmethod
    def _feeding_score(feeding_confidence: float) -> float:
        if (
            isinstance(feeding_confidence, bool)
            or not isinstance(feeding_confidence, (int, float))
            or not math.isfinite(feeding_confidence)
            or not 0.0 <= feeding_confidence <= 1.0
        ):
            raise ValueError("feeding_confidence must be between 0 and 1")
        return float(feeding_confidence)

    @staticmethod
    def _observation(
        posture: CatPosture,
        confidence: float,
        evidence_source: str,
        reliable: dict[int, PoseKeypoint],
        **features: float,
    ) -> PostureObservation:
        complete_features = {
            "visible_keypoints": float(len(reliable)),
            **features,
        }
        return PostureObservation(
            posture=posture,
            confidence=max(0.0, min(1.0, confidence)),
            evidence_source=evidence_source,
            features=tuple(sorted(complete_features.items())),
        )


class PostureStateMachine:
    """Confirm and persist posture changes for one safely isolated subject."""

    def __init__(
        self,
        subject_key: str,
        config: PostureConfig | None = None,
    ) -> None:
        self.subject_key = self._normalize_subject_key(subject_key)
        self.config = config or PostureConfig()
        self._current_posture: CatPosture | None = None
        self._current_confidence: float = 0.0
        self._last_changed_seconds: float | None = None
        self._last_timestamp_seconds: float | None = None
        self._candidate_posture: CatPosture | None = None
        self._candidate_started_seconds: float | None = None
        self._candidate_observations = 0
        self._candidate_confidence_sum = 0.0
        self._candidate_latest_observation: PostureObservation | None = None
        self._pending_transition: PostureTransition | None = None
        self._transition_sequence = 0

    @property
    def current_posture(self) -> CatPosture | None:
        """Return the last successfully persisted posture."""
        return self._current_posture

    @property
    def current_confidence(self) -> float:
        """Return the confidence committed with the current posture."""
        return self._current_confidence

    @property
    def last_changed_seconds(self) -> float | None:
        """Return the media timestamp of the last acknowledged change."""
        return self._last_changed_seconds

    @property
    def transition_sequence(self) -> int:
        """Return the count of successfully acknowledged transitions."""
        return self._transition_sequence

    @property
    def candidate_posture(self) -> CatPosture | None:
        """Return the unconfirmed posture candidate, if any."""
        return self._candidate_posture

    @property
    def pending_transition(self) -> PostureTransition | None:
        """Return the due transition awaiting sink acknowledgement."""
        return self._pending_transition

    def update(
        self,
        timestamp_seconds: float,
        observation: PostureObservation | None,
        visible: bool = True,
    ) -> PostureTransition | None:
        """Advance one observation and return a due, uncommitted transition."""
        if not isinstance(visible, bool):
            raise ValueError("visible must be a boolean")
        if observation is not None and not isinstance(observation, PostureObservation):
            raise ValueError("observation must be a PostureObservation or None")
        timestamp = self._accept_timestamp(timestamp_seconds)

        # Once a transition is due, keep returning that exact proposal until
        # the sink acknowledges it. This preserves ordering and makes failures
        # retryable even if the next detector frame is missing.
        if self._pending_transition is not None:
            return self._pending_transition
        if not self.config.enabled or not visible or observation is None:
            self._clear_candidate()
            return None
        if observation.posture is self._current_posture:
            self._clear_candidate()
            return None

        if observation.posture is not self._candidate_posture:
            self._candidate_posture = observation.posture
            self._candidate_started_seconds = timestamp
            self._candidate_observations = 1
            self._candidate_confidence_sum = observation.confidence
        else:
            self._candidate_observations += 1
            self._candidate_confidence_sum += observation.confidence
        self._candidate_latest_observation = observation

        if self._candidate_started_seconds is None:
            raise RuntimeError("posture candidate requires a start timestamp")
        if (
            timestamp - self._candidate_started_seconds
            < self.config.transition_confirm_seconds
            or self._candidate_observations < self.config.min_confirmed_observations
        ):
            return None

        self._pending_transition = PostureTransition(
            subject_key=self.subject_key,
            previous_posture=self._current_posture,
            posture=observation.posture,
            confidence=(self._candidate_confidence_sum / self._candidate_observations),
            timestamp_seconds=timestamp,
            candidate_started_seconds=self._candidate_started_seconds,
            confirmed_observations=self._candidate_observations,
            transition_sequence=self._transition_sequence + 1,
            reason=(
                "initial_posture"
                if self._current_posture is None
                else "posture_changed"
            ),
            evidence_source=observation.evidence_source,
            features=observation.features,
        )
        return self._pending_transition

    def acknowledge(self, transition: PostureTransition) -> None:
        """Commit a transition after its event has been persisted."""
        if not isinstance(transition, PostureTransition):
            raise ValueError("transition must be a PostureTransition")
        if transition.subject_key != self.subject_key:
            raise ValueError("transition subject does not match state machine")
        if self._pending_transition is None or transition != self._pending_transition:
            raise ValueError("transition is stale or was not proposed")
        if transition.transition_sequence != self._transition_sequence + 1:
            raise ValueError("transition sequence is stale or out of order")
        self._current_posture = transition.posture
        self._current_confidence = transition.confidence
        self._last_changed_seconds = transition.timestamp_seconds
        self._transition_sequence = transition.transition_sequence
        self._pending_transition = None
        self._clear_candidate()

    def rekey(self, new_subject_key: str) -> None:
        """Migrate this machine from an anonymous track key to a confirmed key.

        Collision checks belong to the owner of the machine registry. If a
        transition is already due, it is atomically replaced with an equivalent
        proposal under the new key; a previously returned old-key object then
        fails closed in ``acknowledge``.
        """
        normalized = self._normalize_subject_key(new_subject_key)
        if normalized == self.subject_key:
            return
        self.subject_key = normalized
        if self._pending_transition is not None:
            self._pending_transition = replace(
                self._pending_transition,
                subject_key=normalized,
            )

    def reset_session(self, *, preserve_pending_transition: bool = False) -> None:
        """Reset media time while retaining committed semantic posture state.

        A confirmed-identity owner may also retain an already due transition
        across a media reconnect.  Anonymous tracks must use the default and
        discard it because their identity is session-local.
        """
        if not isinstance(preserve_pending_transition, bool):
            raise ValueError("preserve_pending_transition must be a boolean")
        self._last_timestamp_seconds = None
        if not preserve_pending_transition:
            self._pending_transition = None
        self._clear_candidate()

    def _clear_candidate(self) -> None:
        self._candidate_posture = None
        self._candidate_started_seconds = None
        self._candidate_observations = 0
        self._candidate_confidence_sum = 0.0
        self._candidate_latest_observation = None

    def _accept_timestamp(self, timestamp_seconds: float) -> float:
        if (
            isinstance(timestamp_seconds, bool)
            or not isinstance(timestamp_seconds, (int, float))
            or not math.isfinite(timestamp_seconds)
            or timestamp_seconds < 0.0
        ):
            raise ValueError("timestamp_seconds must be finite and >= 0")
        timestamp = float(timestamp_seconds)
        if (
            self._last_timestamp_seconds is not None
            and timestamp < self._last_timestamp_seconds
        ):
            raise ValueError("timestamp_seconds must be non-decreasing")
        self._last_timestamp_seconds = timestamp
        return timestamp

    @staticmethod
    def _normalize_subject_key(subject_key: str) -> str:
        if not isinstance(subject_key, str) or not subject_key.strip():
            raise ValueError("subject_key must not be blank")
        return subject_key.strip()
