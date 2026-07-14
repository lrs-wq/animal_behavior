"""Pure eating-event state and event-level identity aggregation.

The state machine in this module deliberately has no model, filesystem, or
wall-clock dependencies.  A caller supplies one observation for each processed
frame and immediately consumes any completed events returned by ``update`` or
``finish``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


UNKNOWN_NAMES = {"", "unknown", "unidentified", "none", "null"}
UNKNOWN_COATS = {"", "unknown", "unidentified", "none", "null"}
FINISH_REASONS = {"eof", "disconnect", "manual_stop", "error"}


class EatingState(str, Enum):
    """Observable states of one source-local eating event machine."""

    IDLE = "idle"
    ENTERING = "entering"
    EATING = "eating"
    EXIT_PENDING = "exit_pending"
    MERGE_PENDING = "merge_pending"


def _require_probability(value: float, name: str) -> None:
    """Require a finite probability in the inclusive unit interval."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _require_non_negative_finite(value: float, name: str) -> None:
    """Require a finite, non-negative duration."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    if float(value) < 0.0:
        raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class EatingStateConfig:
    """Timing and event-level identity thresholds."""

    enter_seconds: float = 2.0
    exit_seconds: float = 5.0
    merge_gap_seconds: float = 10.0
    min_event_seconds: float = 2.0
    event_confirm_ratio: float = 0.60
    min_identity_observations: int = 2
    max_identity_candidates: int = 32

    def __post_init__(self) -> None:
        """Reject ambiguous or unbounded runtime settings at construction."""
        _require_non_negative_finite(self.enter_seconds, "enter_seconds")
        _require_non_negative_finite(self.exit_seconds, "exit_seconds")
        _require_non_negative_finite(
            self.merge_gap_seconds,
            "merge_gap_seconds",
        )
        _require_non_negative_finite(
            self.min_event_seconds,
            "min_event_seconds",
        )
        if self.merge_gap_seconds < self.exit_seconds:
            raise ValueError("merge_gap_seconds must be >= exit_seconds")
        _require_probability(self.event_confirm_ratio, "event_confirm_ratio")
        if (
            not isinstance(self.min_identity_observations, int)
            or isinstance(self.min_identity_observations, bool)
            or self.min_identity_observations < 1
        ):
            raise ValueError("min_identity_observations must be >= 1")
        if (
            not isinstance(self.max_identity_candidates, int)
            or isinstance(self.max_identity_candidates, bool)
            or self.max_identity_candidates < 1
        ):
            raise ValueError("max_identity_candidates must be >= 1")


@dataclass(frozen=True)
class IdentityVote:
    """One accepted frame-level identity candidate for an eating target."""

    cat_name: str
    identity_confidence: float
    household_id: str = ""
    coat_class: str = "unknown"

    def __post_init__(self) -> None:
        """Validate confidence while allowing blank/unknown votes to be ignored."""
        _require_probability(self.identity_confidence, "identity_confidence")


@dataclass(frozen=True)
class EatingFrameObservation:
    """One processed frame's evidence for one source-local eating target."""

    timestamp_seconds: float
    feeding_present: bool
    feeding_confidence: float = 0.0
    identity_votes: tuple[IdentityVote, ...] = ()

    def __post_init__(self) -> None:
        """Validate frame-level confidence and contradictory vote input."""
        _require_probability(self.feeding_confidence, "feeding_confidence")
        if not self.feeding_present and self.identity_votes:
            raise ValueError("identity_votes require feeding_present=True")


@dataclass(frozen=True)
class CompletedEatingEvent:
    """One finalized eating interval and its bounded identity aggregation."""

    event_id: str
    source: str
    session_id: str
    event_sequence: int
    household_id: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    cat_name: str
    coat_class: str
    event_identity_confidence: float
    event_identity_margin: float
    observation_count: int
    accepted_identity_observation_count: int
    feeding_confidence_mean: float
    feeding_confidence_max: float
    end_reason: str
    truncated: bool
    identity_rejected_reason: str = ""


class EatingStateMachine:
    """Build completed eating events for one source and one physical target.

    ``update`` timestamps must be finite and strictly increasing.  ``finish``
    may use the last update timestamp because EOF/disconnect cleanup normally
    knows only the timestamp of the final successfully decoded frame.
    """

    def __init__(
        self,
        source: str,
        config: EatingStateConfig | None = None,
        *,
        session_id: str = "session",
        household_id: str = "",
    ) -> None:
        source = str(source).strip()
        session_id = str(session_id).strip()
        if not source:
            raise ValueError("source must not be blank")
        if not session_id:
            raise ValueError("session_id must not be blank")
        self.source = source
        self.session_id = session_id
        self.household_id = str(household_id).strip()
        self.config = config or EatingStateConfig()
        self._next_event_sequence = 1
        self._last_timestamp_seconds: float | None = None
        self._state: EatingState
        self._start_seconds: float | None
        self._last_positive_seconds: float | None
        self._first_absent_seconds: float | None
        self._event_sequence: int | None
        self._event_household_id: str
        self._household_conflict: bool
        self._candidate_overflow: bool
        self._identity_weights: dict[str, float]
        self._identity_counts: dict[str, int]
        self._coat_by_cat: dict[str, str]
        self._accepted_identity_observation_count: int
        self._observation_count: int
        self._feeding_confidence_sum: float
        self._feeding_confidence_max: float
        self._clear_event_state()

    @property
    def state(self) -> EatingState:
        """Return the current public state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Return whether an event candidate or confirmed event is active."""
        return self._state is not EatingState.IDLE

    @property
    def identity_candidate_count(self) -> int:
        """Return retained candidate count for bounded-memory assertions."""
        return len(self._identity_weights)

    @property
    def last_timestamp_seconds(self) -> float | None:
        """Return the last accepted update/finish timestamp."""
        return self._last_timestamp_seconds

    def update(
        self,
        observation: EatingFrameObservation,
    ) -> tuple[CompletedEatingEvent, ...]:
        """Advance one frame and return zero or one completed old event."""
        timestamp = self._accept_update_timestamp(observation.timestamp_seconds)
        present = bool(observation.feeding_present)

        if self._state is EatingState.IDLE:
            if present:
                self._begin_candidate(timestamp, observation)
            return ()

        if self._state is EatingState.ENTERING:
            if not present:
                self._clear_event_state()
                return ()
            if self._start_seconds is None:
                raise RuntimeError("ENTERING state requires start_seconds")
            self._record_positive(observation)
            self._last_positive_seconds = timestamp
            if timestamp - self._start_seconds >= self.config.enter_seconds:
                self._state = EatingState.EATING
            return ()

        if self._state is EatingState.EATING:
            if present:
                self._record_positive(observation)
                self._last_positive_seconds = timestamp
                return ()
            self._first_absent_seconds = timestamp
            if self.config.merge_gap_seconds == 0.0:
                return self._complete_normal(timestamp)
            if self.config.exit_seconds == 0.0:
                self._state = EatingState.MERGE_PENDING
            else:
                self._state = EatingState.EXIT_PENDING
            return ()

        if self._state in {
            EatingState.EXIT_PENDING,
            EatingState.MERGE_PENDING,
        }:
            return self._update_gap(timestamp, observation)

        raise RuntimeError(f"Unhandled eating state: {self._state}")

    def finish(
        self,
        last_timestamp_seconds: float,
        reason: str,
    ) -> tuple[CompletedEatingEvent, ...]:
        """Finalize confirmed state at EOF/disconnect/manual stop.

        Candidate ENTERING state is discarded.  Active confirmed events are
        conservatively marked truncated because the normal merge window could
        not be observed to completion.
        """
        if reason not in FINISH_REASONS:
            allowed = ", ".join(sorted(FINISH_REASONS))
            raise ValueError(f"reason must be one of: {allowed}")
        timestamp = self._accept_finish_timestamp(last_timestamp_seconds)
        if self._state in {EatingState.IDLE, EatingState.ENTERING}:
            self._clear_event_state()
            return ()

        end_seconds = (
            self._first_absent_seconds
            if self._first_absent_seconds is not None
            else self._last_positive_seconds
        )
        if end_seconds is None:
            end_seconds = timestamp
        event = self._finalize_event(
            end_seconds=end_seconds,
            end_reason=reason,
            truncated=True,
        )
        self._clear_event_state()
        return (event,) if event is not None else ()

    def reset(self, *, session_id: str | None = None) -> None:
        """Drop active state and permit a fresh timestamp epoch.

        Supplying a new session ID resets the event sequence to one.  Without a
        new session ID, sequence numbers remain monotonic to avoid duplicate IDs.
        """
        self._clear_event_state()
        self._last_timestamp_seconds = None
        if session_id is not None:
            normalized = str(session_id).strip()
            if not normalized:
                raise ValueError("session_id must not be blank")
            self.session_id = normalized
            self._next_event_sequence = 1

    def _begin_candidate(
        self,
        timestamp: float,
        observation: EatingFrameObservation,
    ) -> None:
        """Start an ENTERING candidate and retain its first positive frame."""
        self._clear_event_state()
        self._state = (
            EatingState.EATING
            if self.config.enter_seconds == 0.0
            else EatingState.ENTERING
        )
        self._start_seconds = timestamp
        self._last_positive_seconds = timestamp
        self._event_sequence = self._next_event_sequence
        self._next_event_sequence += 1
        self._event_household_id = self.household_id
        self._record_positive(observation)

    def _update_gap(
        self,
        timestamp: float,
        observation: EatingFrameObservation,
    ) -> tuple[CompletedEatingEvent, ...]:
        """Advance EXIT_PENDING/MERGE_PENDING with deterministic boundaries."""
        if self._first_absent_seconds is None:
            raise RuntimeError("Gap state requires first_absent_seconds")
        gap_seconds = timestamp - self._first_absent_seconds

        if observation.feeding_present:
            if gap_seconds <= self.config.merge_gap_seconds:
                self._state = EatingState.EATING
                self._first_absent_seconds = None
                self._last_positive_seconds = timestamp
                self._record_positive(observation)
                return ()
            completed = self._complete_normal(self._first_absent_seconds)
            self._begin_candidate(timestamp, observation)
            return completed

        if gap_seconds >= self.config.merge_gap_seconds:
            return self._complete_normal(self._first_absent_seconds)
        if gap_seconds >= self.config.exit_seconds:
            self._state = EatingState.MERGE_PENDING
        return ()

    def _complete_normal(
        self,
        end_seconds: float,
    ) -> tuple[CompletedEatingEvent, ...]:
        """Finalize normal state, reset it, and return an optional event."""
        event = self._finalize_event(
            end_seconds=end_seconds,
            end_reason="normal",
            truncated=False,
        )
        self._clear_event_state()
        return (event,) if event is not None else ()

    def _finalize_event(
        self,
        *,
        end_seconds: float,
        end_reason: str,
        truncated: bool,
    ) -> CompletedEatingEvent | None:
        """Build one immutable event without retaining frame observations."""
        if self._start_seconds is None or self._event_sequence is None:
            return None
        duration_seconds = max(0.0, end_seconds - self._start_seconds)
        if duration_seconds < self.config.min_event_seconds:
            return None

        (
            cat_name,
            coat_class,
            identity_confidence,
            identity_margin,
            rejected_reason,
        ) = self._resolve_identity()
        feeding_mean = (
            self._feeding_confidence_sum / self._observation_count
            if self._observation_count
            else 0.0
        )
        household_id = self._event_household_id
        if self._household_conflict and not self.household_id:
            household_id = ""
        return CompletedEatingEvent(
            event_id=(
                f"{self.source}:{self.session_id}:{self._event_sequence}"
            ),
            source=self.source,
            session_id=self.session_id,
            event_sequence=self._event_sequence,
            household_id=household_id,
            start_seconds=self._start_seconds,
            end_seconds=end_seconds,
            duration_seconds=duration_seconds,
            cat_name=cat_name,
            coat_class=coat_class,
            event_identity_confidence=identity_confidence,
            event_identity_margin=identity_margin,
            observation_count=self._observation_count,
            accepted_identity_observation_count=(
                self._accepted_identity_observation_count
            ),
            feeding_confidence_mean=feeding_mean,
            feeding_confidence_max=self._feeding_confidence_max,
            end_reason=end_reason,
            truncated=truncated,
            identity_rejected_reason=rejected_reason,
        )

    def _resolve_identity(self) -> tuple[str, str, float, float, str]:
        """Resolve event identity from bounded per-cat weight aggregates."""
        total_weight = sum(self._identity_weights.values())
        if total_weight <= 0.0:
            return "", "unknown", 0.0, 0.0, "no_accepted_identity"
        ranked = sorted(
            self._identity_weights.items(),
            key=lambda item: (-item[1], item[0]),
        )
        top_name, top_weight = ranked[0]
        second_weight = ranked[1][1] if len(ranked) > 1 else 0.0
        ratio = top_weight / total_weight
        margin = (top_weight - second_weight) / total_weight

        rejected_reason = ""
        if self._household_conflict:
            rejected_reason = "household_conflict"
        elif self._candidate_overflow:
            rejected_reason = "identity_candidate_overflow"
        elif self._identity_counts.get(top_name, 0) < self.config.min_identity_observations:
            rejected_reason = "insufficient_identity_observations"
        elif ratio < self.config.event_confirm_ratio:
            rejected_reason = "event_confirm_ratio_not_met"
        elif top_weight <= second_weight:
            rejected_reason = "identity_tie"

        if rejected_reason:
            return "", "unknown", ratio, margin, rejected_reason
        return (
            top_name,
            self._coat_by_cat.get(top_name, "unknown"),
            ratio,
            margin,
            "",
        )

    def _record_positive(self, observation: EatingFrameObservation) -> None:
        """Accumulate constant-memory feeding and identity statistics."""
        feeding_confidence = float(observation.feeding_confidence)
        self._observation_count += 1
        self._feeding_confidence_sum += feeding_confidence
        self._feeding_confidence_max = max(
            self._feeding_confidence_max,
            feeding_confidence,
        )

        best_vote_by_cat: dict[str, IdentityVote] = {}
        for vote in observation.identity_votes:
            cat_name = str(vote.cat_name).strip()
            if cat_name.lower() in UNKNOWN_NAMES:
                continue
            previous = best_vote_by_cat.get(cat_name)
            if (
                previous is None
                or vote.identity_confidence > previous.identity_confidence
            ):
                best_vote_by_cat[cat_name] = vote

        for cat_name, vote in best_vote_by_cat.items():
            weight = float(vote.identity_confidence) * feeding_confidence
            if weight <= 0.0:
                continue
            household_id = str(vote.household_id).strip()
            self._observe_household(household_id)
            if cat_name not in self._identity_weights:
                if (
                    len(self._identity_weights)
                    >= self.config.max_identity_candidates
                ):
                    self._candidate_overflow = True
                    continue
                self._identity_weights[cat_name] = 0.0
                self._identity_counts[cat_name] = 0
                self._coat_by_cat[cat_name] = self._normalized_coat(
                    vote.coat_class,
                )
            else:
                observed_coat = self._normalized_coat(vote.coat_class)
                retained_coat = self._coat_by_cat[cat_name]
                if (
                    observed_coat != "unknown"
                    and retained_coat != "unknown"
                    and observed_coat != retained_coat
                ):
                    self._coat_by_cat[cat_name] = "unknown"
                elif retained_coat == "unknown" and observed_coat != "unknown":
                    self._coat_by_cat[cat_name] = observed_coat
            self._identity_weights[cat_name] += weight
            self._identity_counts[cat_name] += 1
            self._accepted_identity_observation_count += 1

    def _observe_household(self, household_id: str) -> None:
        """Retain at most one household ID and flag every disagreement."""
        if not household_id:
            return
        if self.household_id:
            if household_id != self.household_id:
                self._household_conflict = True
            self._event_household_id = self.household_id
            return
        if not self._event_household_id:
            self._event_household_id = household_id
        elif self._event_household_id != household_id:
            self._household_conflict = True

    @staticmethod
    def _normalized_coat(coat_class: str) -> str:
        """Normalize blank/sentinel coat values while preserving configured IDs."""
        coat = str(coat_class).strip()
        return "unknown" if coat.lower() in UNKNOWN_COATS else coat

    def _accept_update_timestamp(self, value: float) -> float:
        """Validate a finite, strictly increasing frame timestamp."""
        timestamp = self._finite_timestamp(value)
        if (
            self._last_timestamp_seconds is not None
            and timestamp <= self._last_timestamp_seconds
        ):
            raise ValueError("update timestamps must be strictly increasing")
        self._last_timestamp_seconds = timestamp
        return timestamp

    def _accept_finish_timestamp(self, value: float) -> float:
        """Validate a finite finish timestamp that does not move backwards."""
        timestamp = self._finite_timestamp(value)
        if (
            self._last_timestamp_seconds is not None
            and timestamp < self._last_timestamp_seconds
        ):
            raise ValueError("finish timestamp must not move backwards")
        self._last_timestamp_seconds = timestamp
        return timestamp

    @staticmethod
    def _finite_timestamp(value: float) -> float:
        """Convert and validate one timestamp."""
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("timestamp_seconds must be finite")
        return float(value)

    def _clear_event_state(self) -> None:
        """Release all per-event state while preserving session sequencing."""
        self._state = EatingState.IDLE
        self._start_seconds = None
        self._last_positive_seconds = None
        self._first_absent_seconds = None
        self._event_sequence = None
        self._event_household_id = self.household_id
        self._household_conflict = False
        self._candidate_overflow = False
        self._identity_weights = {}
        self._identity_counts = {}
        self._coat_by_cat = {}
        self._accepted_identity_observation_count = 0
        self._observation_count = 0
        self._feeding_confidence_sum = 0.0
        self._feeding_confidence_max = 0.0
