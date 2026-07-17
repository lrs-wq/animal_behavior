"""Pure posture-change state-machine boundary tests."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from eating_behavior.posture import (
    CatPosture,
    PostureConfig,
    PostureObservation,
    PostureStateMachine,
)


def observation(
    posture: CatPosture,
    confidence: float = 0.8,
) -> PostureObservation:
    return PostureObservation(
        posture,
        confidence,
        "pose_geometry",
        (("metric", 0.5),),
    )


def machine(
    *,
    confirm_seconds: float = 0.0,
    observations: int = 1,
    key: str = "cam01:track:1",
) -> PostureStateMachine:
    return PostureStateMachine(
        key,
        PostureConfig(
            enabled=True,
            transition_confirm_seconds=confirm_seconds,
            min_confirmed_observations=observations,
        ),
    )


def commit(
    subject: PostureStateMachine,
    timestamp: float,
    posture: CatPosture,
):
    transition = subject.update(timestamp, observation(posture))
    assert transition is not None
    subject.acknowledge(transition)
    return transition


def test_initial_posture_and_each_confirmed_change_require_acknowledgement() -> None:
    subject = machine()

    initial = subject.update(0.0, observation(CatPosture.STANDING))
    assert initial is not None
    assert initial.previous_posture is None
    assert initial.posture is CatPosture.STANDING
    assert initial.reason == "initial_posture"
    assert subject.current_posture is None

    subject.acknowledge(initial)
    assert subject.current_posture is CatPosture.STANDING
    assert subject.last_changed_seconds == 0.0
    assert subject.transition_sequence == 1
    assert subject.update(1.0, observation(CatPosture.STANDING)) is None

    changed = subject.update(2.0, observation(CatPosture.LYING))
    assert changed is not None
    assert changed.previous_posture is CatPosture.STANDING
    assert changed.reason == "posture_changed"
    subject.acknowledge(changed)
    assert subject.current_posture is CatPosture.LYING
    assert changed.transition_sequence == 2
    assert subject.last_changed_seconds == 2.0
    assert subject.transition_sequence == 2


def test_change_requires_both_elapsed_time_and_observation_count() -> None:
    subject = machine(confirm_seconds=1.0, observations=3)

    assert subject.update(0.0, observation(CatPosture.STANDING, 0.6)) is None
    assert subject.update(0.5, observation(CatPosture.STANDING, 0.8)) is None
    transition = subject.update(1.0, observation(CatPosture.STANDING, 1.0))

    assert transition is not None
    assert transition.candidate_started_seconds == 0.0
    assert transition.confirmed_observations == 3
    assert transition.confidence == pytest.approx(0.8)


def test_missing_or_invisible_clears_unstable_candidate_but_keeps_current() -> None:
    subject = machine(confirm_seconds=1.0, observations=2)
    assert subject.update(0.0, observation(CatPosture.STANDING)) is None
    initial = subject.update(1.0, observation(CatPosture.STANDING))
    assert initial is not None
    subject.acknowledge(initial)

    assert subject.update(2.0, observation(CatPosture.LYING)) is None
    assert subject.candidate_posture is CatPosture.LYING
    assert subject.update(2.5, None) is None
    assert subject.candidate_posture is None
    assert subject.current_posture is CatPosture.STANDING

    assert subject.update(3.0, observation(CatPosture.LYING)) is None
    assert subject.update(3.5, None, visible=False) is None
    assert subject.current_posture is CatPosture.STANDING
    assert subject.candidate_posture is None


def test_failed_sink_retries_the_exact_pending_transition() -> None:
    subject = machine()

    failed = subject.update(0.0, observation(CatPosture.CROUCHING))
    assert failed is not None
    assert subject.current_posture is None

    retry = subject.update(0.1, None, visible=False)
    assert retry == failed
    assert subject.pending_transition == failed
    subject.acknowledge(retry)
    assert subject.current_posture is CatPosture.CROUCHING


def test_acknowledge_rejects_modified_stale_or_foreign_transition() -> None:
    subject = machine()
    proposal = subject.update(0.0, observation(CatPosture.STANDING))
    assert proposal is not None

    with pytest.raises(ValueError, match="stale"):
        subject.acknowledge(replace(proposal, confidence=0.1))
    with pytest.raises(ValueError, match="subject"):
        subject.acknowledge(replace(proposal, subject_key="other"))

    subject.acknowledge(proposal)
    with pytest.raises(ValueError, match="stale"):
        subject.acknowledge(proposal)


def test_rekey_preserves_candidate_and_future_proposal_uses_confirmed_key() -> None:
    subject = machine(confirm_seconds=1.0, observations=2)

    assert subject.update(0.0, observation(CatPosture.STANDING)) is None
    subject.rekey("cam01:cat:blue")
    proposal = subject.update(1.0, observation(CatPosture.STANDING))

    assert proposal is not None
    assert proposal.subject_key == "cam01:cat:blue"
    assert proposal.candidate_started_seconds == 0.0
    subject.acknowledge(proposal)
    assert subject.current_posture is CatPosture.STANDING


def test_rekey_atomically_replaces_pending_proposal_and_rejects_old_object() -> None:
    subject = machine()
    old = subject.update(0.0, observation(CatPosture.LYING))
    assert old is not None

    subject.rekey("cam01:cat:blue")
    migrated = subject.pending_transition
    assert migrated is not None
    assert migrated.subject_key == "cam01:cat:blue"
    assert migrated.posture is old.posture
    assert migrated.transition_sequence == old.transition_sequence
    with pytest.raises(ValueError, match="subject"):
        subject.acknowledge(old)

    subject.acknowledge(migrated)
    assert subject.current_posture is CatPosture.LYING


def test_reset_session_keeps_current_but_resets_candidates_and_media_time() -> None:
    subject = machine()
    commit(subject, 10.0, CatPosture.STANDING)
    pending = subject.update(11.0, observation(CatPosture.CROUCHING))
    assert pending is not None

    subject.reset_session()
    assert subject.current_posture is CatPosture.STANDING
    assert subject.last_changed_seconds == 10.0
    assert subject.transition_sequence == 1
    assert subject.pending_transition is None
    assert subject.candidate_posture is None
    assert subject.update(0.0, observation(CatPosture.STANDING)) is None

    changed = subject.update(0.1, observation(CatPosture.CROUCHING))
    assert changed is not None
    assert changed.transition_sequence == 2


def test_reset_session_can_preserve_due_transition_for_confirmed_identity() -> None:
    subject = machine()
    pending = subject.update(10.0, observation(CatPosture.STANDING))
    assert pending is not None

    subject.reset_session(preserve_pending_transition=True)

    assert subject.pending_transition == pending
    retried = subject.update(0.0, observation(CatPosture.LYING))
    assert retried == pending
    subject.acknowledge(retried)
    assert subject.current_posture is CatPosture.STANDING


def test_reset_session_rejects_non_boolean_preserve_flag() -> None:
    subject = machine()
    with pytest.raises(ValueError, match="preserve_pending_transition"):
        subject.reset_session(preserve_pending_transition=1)


def test_timestamp_and_public_input_boundaries_fail_closed() -> None:
    subject = machine()

    with pytest.raises(ValueError, match="finite"):
        subject.update(math.nan, None)
    with pytest.raises(ValueError, match=">= 0"):
        subject.update(-1.0, None)
    subject.update(1.0, None)
    with pytest.raises(ValueError, match="non-decreasing"):
        subject.update(0.5, None)
    with pytest.raises(ValueError, match="visible"):
        subject.update(1.0, None, visible=1)
    with pytest.raises(ValueError, match="observation"):
        subject.update(1.0, "standing")


@pytest.mark.parametrize("key", ["", "   ", None, 1])
def test_subject_keys_must_be_nonblank_strings(key) -> None:
    with pytest.raises(ValueError, match="subject_key"):
        PostureStateMachine(key, PostureConfig(enabled=True))

    subject = machine()
    with pytest.raises(ValueError, match="subject_key"):
        subject.rekey(key)
