"""Source-local cat-presence reminder state.

The public ``cat_passed`` event is a presence reminder: the first frame with
any cat is immediately eligible, and additional reminders are due while a cat
is visible whenever the configured interval has elapsed.  Brief detector gaps
do not create a new presence session, but reminders are never emitted on an
absent frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class CatPresenceState(str, Enum):
    """Observable states for one camera's any-cat presence."""

    ABSENT = "absent"
    PRESENT = "present"
    EXIT_PENDING = "exit_pending"


@dataclass(frozen=True)
class CatPresenceConfig:
    """Reminder interval and detector-gap tolerance, expressed in seconds."""

    reminder_interval_seconds: float = 30.0 * 60.0
    absence_confirm_seconds: float = 2.0

    def __post_init__(self) -> None:
        for name, value in (
            ("reminder_interval_seconds", self.reminder_interval_seconds),
            ("absence_confirm_seconds", self.absence_confirm_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.reminder_interval_seconds <= 0.0:
            raise ValueError("reminder_interval_seconds must be > 0")
        if self.absence_confirm_seconds < 0.0:
            raise ValueError("absence_confirm_seconds must be >= 0")


@dataclass(frozen=True)
class CatPresenceReminder:
    """One reminder that is due but not yet acknowledged as persisted."""

    source: str
    timestamp_seconds: float
    reason: str
    presence_sequence: int
    reminder_sequence: int
    presence_started_seconds: float
    presence_duration_seconds: float
    reminder_interval_seconds: float


class CatPresenceStateMachine:
    """Track any-cat presence and produce periodic reminder candidates.

    ``update`` uses media time and therefore requires non-decreasing timestamps
    within one source session.  A reminder becomes rate-limiting only after
    ``acknowledge`` is called, so a failed event sink can be retried on the next
    visible frame.
    """

    def __init__(
        self,
        source: str,
        config: CatPresenceConfig | None = None,
    ) -> None:
        normalized_source = str(source).strip()
        if not normalized_source:
            raise ValueError("source must not be blank")
        self.source = normalized_source
        self.config = config or CatPresenceConfig()
        self._state = CatPresenceState.ABSENT
        self._last_timestamp_seconds: float | None = None
        self._last_announced_seconds: float | None = None
        self._presence_started_seconds: float | None = None
        self._first_absent_seconds: float | None = None
        self._presence_sequence = 0
        self._reminder_sequence = 0

    @property
    def state(self) -> CatPresenceState:
        """Return the current observable presence state."""
        return self._state

    @property
    def last_announced_seconds(self) -> float | None:
        """Return the last successfully persisted reminder timestamp."""
        return self._last_announced_seconds

    def update(
        self,
        timestamp_seconds: float,
        cat_present: bool,
    ) -> CatPresenceReminder | None:
        """Advance one frame and return at most one due reminder."""
        timestamp = self._accept_timestamp(timestamp_seconds)
        present = bool(cat_present)

        if self._state is CatPresenceState.ABSENT:
            if not present:
                return None
            self._begin_presence(timestamp)
            return self._due_reminder(timestamp)

        if self._state is CatPresenceState.PRESENT:
            if present:
                return self._due_reminder(timestamp)
            if self.config.absence_confirm_seconds == 0.0:
                self._end_presence()
            else:
                self._state = CatPresenceState.EXIT_PENDING
                self._first_absent_seconds = timestamp
            return None

        if self._state is CatPresenceState.EXIT_PENDING:
            if self._first_absent_seconds is None:
                raise RuntimeError("EXIT_PENDING requires first_absent_seconds")
            absent_seconds = timestamp - self._first_absent_seconds
            if present:
                if absent_seconds >= self.config.absence_confirm_seconds:
                    self._end_presence()
                    self._begin_presence(timestamp)
                else:
                    self._state = CatPresenceState.PRESENT
                    self._first_absent_seconds = None
                return self._due_reminder(timestamp)
            if absent_seconds >= self.config.absence_confirm_seconds:
                self._end_presence()
            return None

        raise RuntimeError(f"Unhandled cat presence state: {self._state}")

    def acknowledge(self, reminder: CatPresenceReminder) -> None:
        """Commit one reminder after its event has been persisted."""
        if reminder.source != self.source:
            raise ValueError("reminder source does not match state machine")
        if reminder.reminder_sequence != self._reminder_sequence + 1:
            raise ValueError("reminder sequence is stale or out of order")
        if reminder.timestamp_seconds > (self._last_timestamp_seconds or 0.0):
            raise ValueError("reminder timestamp is ahead of state machine")
        if (
            self._last_announced_seconds is not None
            and reminder.timestamp_seconds - self._last_announced_seconds
            < self.config.reminder_interval_seconds
        ):
            raise ValueError("reminder interval has not elapsed")
        self._last_announced_seconds = reminder.timestamp_seconds
        self._reminder_sequence = reminder.reminder_sequence

    def _begin_presence(self, timestamp: float) -> None:
        self._state = CatPresenceState.PRESENT
        self._first_absent_seconds = None
        self._presence_started_seconds = timestamp
        self._presence_sequence += 1

    def _end_presence(self) -> None:
        self._state = CatPresenceState.ABSENT
        self._presence_started_seconds = None
        self._first_absent_seconds = None

    def _due_reminder(self, timestamp: float) -> CatPresenceReminder | None:
        if self._state is not CatPresenceState.PRESENT:
            return None
        if (
            self._last_announced_seconds is not None
            and timestamp - self._last_announced_seconds
            < self.config.reminder_interval_seconds
        ):
            return None
        if self._presence_started_seconds is None:
            raise RuntimeError("PRESENT requires presence_started_seconds")
        return CatPresenceReminder(
            source=self.source,
            timestamp_seconds=timestamp,
            reason=(
                "initial_presence"
                if self._last_announced_seconds is None
                else "reminder_interval_elapsed"
            ),
            presence_sequence=self._presence_sequence,
            reminder_sequence=self._reminder_sequence + 1,
            presence_started_seconds=self._presence_started_seconds,
            presence_duration_seconds=max(
                0.0,
                timestamp - self._presence_started_seconds,
            ),
            reminder_interval_seconds=self.config.reminder_interval_seconds,
        )

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
