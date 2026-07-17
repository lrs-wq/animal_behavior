"""Boundary tests for source-local cat-presence reminders."""

import math

import pytest

from eating_behavior.presence_state import (
    CatPresenceConfig,
    CatPresenceState,
    CatPresenceStateMachine,
)


def machine(*, interval: float = 10.0, absence: float = 2.0) -> CatPresenceStateMachine:
    return CatPresenceStateMachine(
        "cam01",
        CatPresenceConfig(
            reminder_interval_seconds=interval,
            absence_confirm_seconds=absence,
        ),
    )


def acknowledge_due(
    subject: CatPresenceStateMachine,
    timestamp: float,
    present: bool = True,
):
    reminder = subject.update(timestamp, present)
    assert reminder is not None
    subject.acknowledge(reminder)
    return reminder


def test_first_presence_is_immediate_and_continuous_presence_repeats_on_boundary() -> (
    None
):
    subject = machine()

    first = acknowledge_due(subject, 0.0)
    assert first.reason == "initial_presence"
    assert first.presence_sequence == 1
    assert subject.update(9.999, True) is None

    repeated = acknowledge_due(subject, 10.0)
    assert repeated.reason == "reminder_interval_elapsed"
    assert repeated.presence_sequence == 1
    assert repeated.reminder_sequence == 2
    assert repeated.presence_duration_seconds == 10.0


def test_default_interval_repeats_at_exactly_thirty_minutes() -> None:
    subject = CatPresenceStateMachine("cam01")

    acknowledge_due(subject, 0.0)
    assert subject.update(1799.999, True) is None
    reminder = acknowledge_due(subject, 1800.0)

    assert reminder.reminder_interval_seconds == 1800.0


def test_failed_sink_can_retry_because_due_reminder_requires_acknowledgement() -> None:
    subject = machine()

    failed = subject.update(0.0, True)
    assert failed is not None
    assert subject.last_announced_seconds is None

    retry = subject.update(0.1, True)
    assert retry is not None
    assert retry.reminder_sequence == failed.reminder_sequence == 1
    subject.acknowledge(retry)
    assert subject.last_announced_seconds == 0.1


def test_early_reappearance_waits_until_original_reminder_deadline() -> None:
    subject = machine()
    acknowledge_due(subject, 0.0)

    assert subject.update(2.0, False) is None
    assert subject.update(4.0, False) is None
    assert subject.state is CatPresenceState.ABSENT
    assert subject.update(9.0, True) is None

    repeated = acknowledge_due(subject, 10.0)
    assert repeated.presence_sequence == 2
    assert repeated.presence_started_seconds == 9.0


def test_absence_past_deadline_never_emits_and_late_reappearance_is_immediate() -> None:
    subject = machine()
    acknowledge_due(subject, 0.0)

    assert subject.update(1.0, False) is None
    assert subject.update(12.0, False) is None
    assert subject.state is CatPresenceState.ABSENT

    repeated = acknowledge_due(subject, 15.0)
    assert repeated.reason == "reminder_interval_elapsed"
    assert repeated.presence_sequence == 2


def test_brief_detector_gap_keeps_one_presence_session_and_can_remind_on_return() -> (
    None
):
    subject = machine()
    acknowledge_due(subject, 0.0)

    assert subject.update(9.5, False) is None
    assert subject.state is CatPresenceState.EXIT_PENDING
    repeated = acknowledge_due(subject, 10.0)

    assert subject.state is CatPresenceState.PRESENT
    assert repeated.presence_sequence == 1
    assert repeated.presence_started_seconds == 0.0


def test_crossing_multiple_intervals_emits_only_one_current_reminder() -> None:
    subject = machine()
    acknowledge_due(subject, 0.0)

    repeated = acknowledge_due(subject, 35.0)
    assert repeated.reminder_sequence == 2
    assert subject.update(44.999, True) is None
    assert acknowledge_due(subject, 45.0).reminder_sequence == 3


def test_zero_absence_confirmation_marks_missing_frame_absent_immediately() -> None:
    subject = machine(absence=0.0)
    acknowledge_due(subject, 0.0)

    assert subject.update(1.0, False) is None
    assert subject.state is CatPresenceState.ABSENT
    assert subject.update(2.0, True) is None
    assert subject.state is CatPresenceState.PRESENT


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reminder_interval_seconds": 0.0}, "reminder_interval_seconds"),
        ({"reminder_interval_seconds": -1.0}, "reminder_interval_seconds"),
        ({"reminder_interval_seconds": math.inf}, "reminder_interval_seconds"),
        ({"reminder_interval_seconds": True}, "reminder_interval_seconds"),
        ({"absence_confirm_seconds": -1.0}, "absence_confirm_seconds"),
        ({"absence_confirm_seconds": math.nan}, "absence_confirm_seconds"),
    ],
)
def test_config_rejects_unsafe_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CatPresenceConfig(**kwargs)


def test_timestamps_must_be_finite_non_negative_and_non_decreasing() -> None:
    subject = machine()

    with pytest.raises(ValueError, match="finite"):
        subject.update(math.nan, True)
    with pytest.raises(ValueError, match=">= 0"):
        subject.update(-1.0, True)

    reminder = subject.update(1.0, True)
    assert reminder is not None
    with pytest.raises(ValueError, match="non-decreasing"):
        subject.update(0.5, True)
