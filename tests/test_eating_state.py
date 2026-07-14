"""Boundary tests for the pure eating-event state machine."""

from __future__ import annotations

import math

import pytest

from eating_behavior.eating_state import (
    CompletedEatingEvent,
    EatingFrameObservation,
    EatingState,
    EatingStateConfig,
    EatingStateMachine,
    IdentityVote,
)


def make_config(**overrides) -> EatingStateConfig:
    """Build a compact test configuration."""
    values = {
        "enter_seconds": 2.0,
        "exit_seconds": 2.0,
        "merge_gap_seconds": 4.0,
        "min_event_seconds": 2.0,
        "event_confirm_ratio": 0.60,
        "min_identity_observations": 2,
        "max_identity_candidates": 8,
    }
    values.update(overrides)
    return EatingStateConfig(**values)


def frame(
    timestamp: float,
    present: bool,
    *,
    confidence: float = 0.9,
    votes: tuple[IdentityVote, ...] = (),
) -> EatingFrameObservation:
    """Build one positive or negative frame observation."""
    return EatingFrameObservation(
        timestamp_seconds=timestamp,
        feeding_present=present,
        feeding_confidence=confidence if present else 0.0,
        identity_votes=votes,
    )


def vote(
    cat_name: str,
    confidence: float,
    *,
    household_id: str = "hh",
    coat_class: str = "blue_cat",
) -> IdentityVote:
    """Build one identity vote."""
    return IdentityVote(
        cat_name=cat_name,
        identity_confidence=confidence,
        household_id=household_id,
        coat_class=coat_class,
    )


def finish_immediately(
    machine: EatingStateMachine,
    timestamp: float,
) -> CompletedEatingEvent:
    """Close an enter=exit=merge=0 event with one negative frame."""
    completed = machine.update(frame(timestamp, False))
    assert len(completed) == 1
    return completed[0]


def test_enter_exit_and_merge_boundaries() -> None:
    """Exact enter/exit/merge boundaries have deterministic behavior."""
    machine = EatingStateMachine("cam", make_config())

    assert machine.update(frame(0.0, True)) == ()
    assert machine.state is EatingState.ENTERING
    assert machine.update(frame(1.9, True)) == ()
    assert machine.state is EatingState.ENTERING
    assert machine.update(frame(2.0, True)) == ()
    assert machine.state is EatingState.EATING

    assert machine.update(frame(3.0, False)) == ()
    assert machine.state is EatingState.EXIT_PENDING
    assert machine.update(frame(5.0, False)) == ()
    assert machine.state is EatingState.MERGE_PENDING

    # A positive frame exactly on the merge boundary resumes the same event.
    assert machine.update(frame(7.0, True)) == ()
    assert machine.state is EatingState.EATING
    assert machine.update(frame(8.0, False)) == ()
    completed = machine.update(frame(12.0, False))

    assert len(completed) == 1
    event = completed[0]
    assert event.start_seconds == 0.0
    assert event.end_seconds == 8.0
    assert event.duration_seconds == 8.0
    assert event.observation_count == 4
    assert event.end_reason == "normal"
    assert event.truncated is False
    assert machine.state is EatingState.IDLE


def test_positive_after_merge_window_finishes_old_and_starts_new() -> None:
    """A late positive frame closes the old event before opening a new one."""
    config = make_config(
        enter_seconds=0.0,
        exit_seconds=1.0,
        merge_gap_seconds=2.0,
        min_event_seconds=0.0,
    )
    machine = EatingStateMachine("cam", config, session_id="run")

    machine.update(frame(0.0, True))
    machine.update(frame(1.0, False))
    completed = machine.update(frame(3.1, True))

    assert len(completed) == 1
    assert completed[0].event_id == "cam:run:1"
    assert completed[0].end_seconds == 1.0
    assert machine.state is EatingState.EATING

    machine.update(frame(4.0, False))
    second = machine.update(frame(6.0, False))
    assert second[0].event_id == "cam:run:2"
    assert second[0].start_seconds == 3.1


def test_enter_candidate_break_and_minimum_duration_filter() -> None:
    """Broken enter candidates and short confirmed events emit nothing."""
    entering = EatingStateMachine("cam", make_config())
    entering.update(frame(0.0, True))
    assert entering.update(frame(1.0, False)) == ()
    assert entering.state is EatingState.IDLE

    short = EatingStateMachine(
        "cam",
        make_config(
            enter_seconds=0.0,
            exit_seconds=0.0,
            merge_gap_seconds=0.0,
            min_event_seconds=2.0,
        ),
    )
    short.update(frame(0.0, True))
    assert short.update(frame(1.9, False)) == ()
    assert short.state is EatingState.IDLE

    exact = EatingStateMachine(
        "cam",
        make_config(
            enter_seconds=0.0,
            exit_seconds=0.0,
            merge_gap_seconds=0.0,
            min_event_seconds=2.0,
        ),
    )
    exact.update(frame(0.0, True))
    assert exact.update(frame(2.0, False))[0].duration_seconds == 2.0


@pytest.mark.parametrize("reason", ["eof", "disconnect", "manual_stop", "error"])
def test_finish_closes_confirmed_event_at_last_media_time(reason: str) -> None:
    """All supported finish paths conservatively truncate active events."""
    machine = EatingStateMachine(
        "cam",
        make_config(
            enter_seconds=0.0,
            min_event_seconds=0.0,
        ),
    )
    machine.update(frame(4.0, True))

    completed = machine.finish(4.0, reason)

    assert len(completed) == 1
    event = completed[0]
    assert event.start_seconds == 4.0
    assert event.end_seconds == 4.0
    assert event.end_reason == reason
    assert event.truncated is True
    assert machine.state is EatingState.IDLE


def test_finish_discards_unconfirmed_candidate_and_uses_first_absence() -> None:
    """EOF does not promote ENTERING and does not extend through an absence."""
    candidate = EatingStateMachine("cam", make_config())
    candidate.update(frame(0.0, True))
    assert candidate.finish(0.0, "eof") == ()
    assert candidate.state is EatingState.IDLE

    active = EatingStateMachine(
        "cam",
        make_config(enter_seconds=0.0, min_event_seconds=0.0),
    )
    active.update(frame(0.0, True))
    active.update(frame(2.0, False))
    event = active.finish(3.0, "eof")[0]
    assert event.end_seconds == 2.0
    assert event.duration_seconds == 2.0


def test_weighted_identity_vote_outputs_ratio_margin_and_stats() -> None:
    """Identity and feeding metrics use bounded weighted aggregation."""
    config = make_config(
        enter_seconds=0.0,
        exit_seconds=0.0,
        merge_gap_seconds=0.0,
        min_event_seconds=0.0,
    )
    machine = EatingStateMachine("cam", config, household_id="hh")
    machine.update(
        frame(
            0.0,
            True,
            confidence=0.9,
            votes=(vote("blue", 1.0),),
        )
    )
    machine.update(
        frame(
            1.0,
            True,
            confidence=0.8,
            votes=(vote("blue", 0.5), vote("silver", 0.25)),
        )
    )
    event = finish_immediately(machine, 2.0)

    # blue=(1.0*0.9)+(0.5*0.8)=1.3; silver=0.25*0.8=0.2
    assert event.cat_name == "blue"
    assert event.coat_class == "blue_cat"
    assert event.event_identity_confidence == pytest.approx(1.3 / 1.5)
    assert event.event_identity_margin == pytest.approx(1.1 / 1.5)
    assert event.accepted_identity_observation_count == 3
    assert event.observation_count == 2
    assert event.feeding_confidence_mean == pytest.approx(0.85)
    assert event.feeding_confidence_max == 0.9


def test_ratio_minimum_count_and_tie_reject_identity() -> None:
    """Weak, sparse, and tied event votes remain publicly unidentified."""
    base = {
        "enter_seconds": 0.0,
        "exit_seconds": 0.0,
        "merge_gap_seconds": 0.0,
        "min_event_seconds": 0.0,
    }

    ratio = EatingStateMachine(
        "ratio",
        make_config(min_identity_observations=1, **base),
    )
    ratio.update(frame(0.0, True, votes=(vote("a", 0.55),)))
    ratio.update(frame(1.0, True, votes=(vote("b", 0.45),)))
    ratio_event = finish_immediately(ratio, 2.0)
    assert ratio_event.cat_name == ""
    assert ratio_event.event_identity_confidence == pytest.approx(0.55)
    assert ratio_event.identity_rejected_reason == "event_confirm_ratio_not_met"

    sparse = EatingStateMachine("sparse", make_config(**base))
    sparse.update(frame(0.0, True, votes=(vote("a", 1.0),)))
    sparse_event = finish_immediately(sparse, 1.0)
    assert sparse_event.cat_name == ""
    assert sparse_event.identity_rejected_reason == (
        "insufficient_identity_observations"
    )

    dominant_once = EatingStateMachine("dominant-once", make_config(**base))
    dominant_once.update(frame(0.0, True, votes=(vote("a", 1.0),)))
    dominant_once.update(frame(1.0, True, votes=(vote("b", 0.1),)))
    dominant_once_event = finish_immediately(dominant_once, 2.0)
    assert dominant_once_event.event_identity_confidence > 0.90
    assert dominant_once_event.cat_name == ""
    assert dominant_once_event.identity_rejected_reason == (
        "insufficient_identity_observations"
    )

    tie = EatingStateMachine(
        "tie",
        make_config(
            event_confirm_ratio=0.5,
            min_identity_observations=1,
            **base,
        ),
    )
    tie.update(frame(0.0, True, votes=(vote("a", 1.0),)))
    tie.update(frame(1.0, True, votes=(vote("b", 1.0),)))
    tie_event = finish_immediately(tie, 2.0)
    assert tie_event.cat_name == ""
    assert tie_event.event_identity_margin == 0.0
    assert tie_event.identity_rejected_reason == "identity_tie"


def test_household_conflict_rejects_cat_but_keeps_source_household() -> None:
    """Cross-household votes fail closed without corrupting source routing."""
    config = make_config(
        enter_seconds=0.0,
        exit_seconds=0.0,
        merge_gap_seconds=0.0,
        min_event_seconds=0.0,
    )
    machine = EatingStateMachine("cam", config, household_id="hh-a")
    machine.update(
        frame(
            0.0,
            True,
            votes=(vote("blue", 1.0, household_id="hh-a"),),
        )
    )
    machine.update(
        frame(
            1.0,
            True,
            votes=(vote("blue", 1.0, household_id="hh-b"),),
        )
    )
    event = finish_immediately(machine, 2.0)

    assert event.cat_name == ""
    assert event.household_id == "hh-a"
    assert event.identity_rejected_reason == "household_conflict"


def test_duplicate_votes_count_once_and_unknown_votes_are_ignored() -> None:
    """One frame cannot inflate counts with duplicate or sentinel votes."""
    config = make_config(
        enter_seconds=0.0,
        exit_seconds=0.0,
        merge_gap_seconds=0.0,
        min_event_seconds=0.0,
        min_identity_observations=1,
    )
    machine = EatingStateMachine("cam", config)
    machine.update(
        frame(
            0.0,
            True,
            votes=(
                vote("blue", 0.2),
                vote("blue", 0.9),
                vote("unknown", 1.0),
                vote("", 1.0),
            ),
        )
    )
    event = finish_immediately(machine, 1.0)

    assert event.cat_name == "blue"
    assert event.accepted_identity_observation_count == 1
    assert event.event_identity_confidence == 1.0


def test_identity_candidate_storage_is_bounded_and_cleared() -> None:
    """Long events retain a fixed candidate map rather than frame history."""
    config = make_config(
        enter_seconds=0.0,
        exit_seconds=0.0,
        merge_gap_seconds=0.0,
        min_event_seconds=0.0,
        min_identity_observations=1,
        max_identity_candidates=2,
    )
    machine = EatingStateMachine("cam", config)

    for index in range(100):
        machine.update(
            frame(
                float(index),
                True,
                votes=(vote(f"cat-{index}", 1.0),),
            )
        )
        assert machine.identity_candidate_count <= 2

    event = finish_immediately(machine, 100.0)
    assert event.observation_count == 100
    assert event.cat_name == ""
    assert event.identity_rejected_reason == "identity_candidate_overflow"
    assert machine.identity_candidate_count == 0
    assert machine.state is EatingState.IDLE


def test_event_sequence_and_new_session_timestamp_epoch() -> None:
    """IDs are unique per session and reset only with an explicit new session."""
    config = make_config(
        enter_seconds=0.0,
        exit_seconds=0.0,
        merge_gap_seconds=0.0,
        min_event_seconds=0.0,
    )
    machine = EatingStateMachine("cam", config, session_id="one")

    machine.update(frame(0.0, True))
    first = finish_immediately(machine, 1.0)
    machine.update(frame(2.0, True))
    second = finish_immediately(machine, 3.0)

    assert first.event_id == "cam:one:1"
    assert second.event_id == "cam:one:2"
    machine.reset(session_id="two")
    machine.update(frame(0.0, True))
    third = finish_immediately(machine, 1.0)
    assert third.event_id == "cam:two:1"


@pytest.mark.parametrize("bad_timestamp", [math.nan, math.inf, -math.inf])
def test_update_rejects_non_finite_timestamps(bad_timestamp: float) -> None:
    """State decisions never accept a non-finite media timestamp."""
    machine = EatingStateMachine("cam", make_config())
    with pytest.raises(ValueError, match="must be finite"):
        machine.update(frame(bad_timestamp, True))


def test_update_is_strictly_increasing_and_finish_cannot_regress() -> None:
    """Duplicate/backward frames are rejected while equal-time finish is valid."""
    machine = EatingStateMachine(
        "cam",
        make_config(enter_seconds=0.0, min_event_seconds=0.0),
    )
    machine.update(frame(1.0, True))
    with pytest.raises(ValueError, match="strictly increasing"):
        machine.update(frame(1.0, True))
    with pytest.raises(ValueError, match="strictly increasing"):
        machine.update(frame(0.5, True))
    with pytest.raises(ValueError, match="must not move backwards"):
        machine.finish(0.5, "eof")
    assert machine.finish(1.0, "eof")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"enter_seconds": -1.0}, "enter_seconds"),
        ({"enter_seconds": False}, "enter_seconds"),
        ({"exit_seconds": math.inf}, "exit_seconds"),
        (
            {"exit_seconds": 3.0, "merge_gap_seconds": 2.0},
            "merge_gap_seconds",
        ),
        ({"event_confirm_ratio": 1.1}, "event_confirm_ratio"),
        ({"event_confirm_ratio": False}, "event_confirm_ratio"),
        ({"min_identity_observations": 0}, "min_identity_observations"),
        ({"max_identity_candidates": 0}, "max_identity_candidates"),
    ],
)
def test_config_rejects_invalid_boundaries(overrides, message: str) -> None:
    """Invalid timing and bounded-memory settings fail at startup."""
    with pytest.raises(ValueError, match=message):
        make_config(**overrides)


def test_observation_rejects_votes_without_feeding() -> None:
    """Identity evidence cannot exist without a positive feeding observation."""
    with pytest.raises(ValueError, match="feeding_present"):
        EatingFrameObservation(
            timestamp_seconds=0.0,
            feeding_present=False,
            feeding_confidence=0.0,
            identity_votes=(vote("blue", 1.0),),
        )


def test_finish_reason_is_explicit() -> None:
    """Unexpected termination labels do not silently enter the event schema."""
    machine = EatingStateMachine("cam", make_config())
    with pytest.raises(ValueError, match="reason must be one of"):
        machine.finish(0.0, "crash")
