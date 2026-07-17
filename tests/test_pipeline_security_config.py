"""Security policy integration tests for pipeline construction."""

from pathlib import Path

import pytest

import eating_behavior.pipeline as pipeline_module
from eating_behavior.pipeline import RealtimePipeline, parse_posture_config


def test_posture_events_are_opt_in_and_require_pose_pipeline() -> None:
    assert not parse_posture_config(
        {},
        pose_enabled=True,
        pose_keypoint_confidence=0.3,
    ).enabled
    assert parse_posture_config(
        {"enabled": True},
        pose_enabled=True,
        pose_keypoint_confidence=0.3,
    ).enabled
    assert not parse_posture_config(
        {"enabled": True},
        pose_enabled=False,
        pose_keypoint_confidence=0.3,
    ).enabled


def test_pipeline_requires_both_model_hashes_by_default(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Both detection_model_sha256"):
        RealtimePipeline.from_configs(
            models_config={
                "detection_model": "detector.pt",
                "coat_classifier_model": "coat.pt",
            },
            pipeline_config={},
            output_dir=tmp_path,
        )


def test_pipeline_passes_explicit_model_security_policy(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeDetector:
        def __init__(self, model_path, **kwargs) -> None:
            calls.append((str(model_path), kwargs))

    class FakeClassifier:
        def __init__(self, model_path, **kwargs) -> None:
            calls.append((str(model_path), kwargs))

    monkeypatch.setattr(pipeline_module, "CatDetector", FakeDetector)
    monkeypatch.setattr(pipeline_module, "CoatClassifier", FakeClassifier)
    digest_a = "a" * 64
    digest_b = "b" * 64
    allowed_root = str(tmp_path / "trusted-models")

    RealtimePipeline.from_configs(
        models_config={
            "detection_model": "detector.pt",
            "detection_model_sha256": digest_a,
            "coat_classifier_model": "coat.pt",
            "coat_classifier_model_sha256": digest_b,
            "security": {
                "require_sha256": True,
                "allow_pickle_weights": False,
                "allowed_roots": [allowed_root],
            },
        },
        pipeline_config={},
        output_dir=tmp_path,
    )

    assert calls[0][1]["expected_sha256"] == digest_a
    assert calls[1][1]["expected_sha256"] == digest_b
    assert calls[0][1]["allowed_roots"] == [allowed_root]
    assert calls[1][1]["allow_pickle_weights"] is False


def test_pipeline_requires_and_passes_pinned_pose_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    pose_calls: list[tuple[str, str, dict[str, object]]] = []

    class FakeModel:
        def __init__(self, model_path, **kwargs) -> None:
            pass

    class FakePose:
        def __init__(self, config_path, checkpoint_path, **kwargs) -> None:
            pose_calls.append((str(config_path), str(checkpoint_path), kwargs))

        def predict(self, frame, cat_detections):
            return tuple(None for _ in cat_detections)

    monkeypatch.setattr(pipeline_module, "CatDetector", FakeModel)
    monkeypatch.setattr(pipeline_module, "CoatClassifier", FakeModel)
    monkeypatch.setattr(pipeline_module, "RTMPoseEstimator", FakePose)
    root = str(tmp_path / "trusted")

    pipeline = RealtimePipeline.from_configs(
        models_config={
            "detection_model": "detector.pt",
            "detection_model_sha256": "a" * 64,
            "coat_classifier_model": "coat.pt",
            "coat_classifier_model_sha256": "b" * 64,
            "pose": {
                "config": "cat_pose.py",
                "config_sha256": "c" * 64,
                "checkpoint": "cat_pose.pth",
                "checkpoint_sha256": "d" * 64,
            },
            "security": {
                "require_sha256": True,
                "allow_pickle_weights": True,
                "allowed_roots": [root],
            },
        },
        pipeline_config={
            "pose": {
                "enabled": True,
                "posture": {
                    "enabled": True,
                    "transition_confirm_seconds": 0.75,
                    "min_confirmed_observations": 4,
                },
            }
        },
        output_dir=tmp_path,
        device="0",
    )

    assert pose_calls == [
        (
            "cat_pose.py",
            "cat_pose.pth",
            {
                "device": "0",
                "min_cat_area": 0,
                "max_instances_per_frame": 8,
                "allowed_roots": [root],
                "expected_config_sha256": "c" * 64,
                "expected_checkpoint_sha256": "d" * 64,
                "allow_pickle_weights": True,
            },
        )
    ]
    assert pipeline.config.posture.enabled is True
    assert pipeline.config.posture.transition_confirm_seconds == 0.75
    assert pipeline.config.posture.min_confirmed_observations == 4


def test_pipeline_rejects_enabled_pose_without_both_hashes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"models\.pose\.config_sha256"):
        RealtimePipeline.from_configs(
            models_config={
                "detection_model": "detector.pt",
                "detection_model_sha256": "a" * 64,
                "coat_classifier_model": "coat.pt",
                "coat_classifier_model_sha256": "b" * 64,
                "pose": {
                    "config": "cat_pose.py",
                    "checkpoint": "cat_pose.pth",
                },
            },
            pipeline_config={"pose": {"enabled": True}},
            output_dir=tmp_path,
        )


def test_pose_hashes_remain_mandatory_when_global_hash_policy_is_disabled(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"models\.pose\.config_sha256"):
        RealtimePipeline.from_configs(
            models_config={
                "detection_model": "detector.onnx",
                "coat_classifier_model": "coat.onnx",
                "pose": {
                    "config": "cat_pose.py",
                    "checkpoint": "cat_pose.pth",
                },
                "security": {
                    "require_sha256": False,
                    "allow_pickle_weights": True,
                    "allowed_roots": [str(tmp_path)],
                },
            },
            pipeline_config={"pose": {"enabled": True}},
            output_dir=tmp_path,
        )


@pytest.mark.parametrize("value", ["false", "no", "0", "off"])
def test_pipeline_security_parses_false_strings(
    monkeypatch, tmp_path: Path, value: str
) -> None:
    captured: list[bool] = []

    class FakeModel:
        def __init__(self, model_path, **kwargs) -> None:
            captured.append(kwargs["allow_pickle_weights"])

    monkeypatch.setattr(pipeline_module, "CatDetector", FakeModel)
    monkeypatch.setattr(pipeline_module, "CoatClassifier", FakeModel)

    RealtimePipeline.from_configs(
        models_config={
            "detection_model": "detector.onnx",
            "coat_classifier_model": "coat.onnx",
            "security": {
                "require_sha256": False,
                "allow_pickle_weights": value,
            },
        },
        pipeline_config={},
        output_dir=tmp_path,
    )

    assert captured == [False, False]


@pytest.mark.parametrize(
    ("events_config", "expected_interval", "expected_absence"),
    [
        ({}, 1800.0, 2.0),
        (
            {
                "cat_presence": {
                    "repeat_interval_seconds": 600,
                    "absence_confirm_seconds": 3,
                }
            },
            600.0,
            3.0,
        ),
        (
            {
                "pass_event_interval_seconds": 900,
                "pass_event_absence_confirm_seconds": 4,
            },
            900.0,
            4.0,
        ),
    ],
)
def test_cat_presence_config_defaults_and_legacy_aliases(
    monkeypatch,
    tmp_path: Path,
    events_config: dict,
    expected_interval: float,
    expected_absence: float,
) -> None:
    class FakeModel:
        def __init__(self, model_path, **kwargs) -> None:
            pass

    monkeypatch.setattr(pipeline_module, "CatDetector", FakeModel)
    monkeypatch.setattr(pipeline_module, "CoatClassifier", FakeModel)

    pipeline = RealtimePipeline.from_configs(
        models_config={
            "detection_model": "detector.onnx",
            "coat_classifier_model": "coat.onnx",
            "security": {"require_sha256": False},
        },
        pipeline_config={"events": events_config},
        output_dir=tmp_path,
    )

    assert pipeline.config.pass_event_interval_seconds == expected_interval
    assert pipeline.config.pass_absence_confirm_seconds == expected_absence


def test_cat_presence_parses_household_interval_overrides(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeModel:
        def __init__(self, model_path, **kwargs) -> None:
            pass

    monkeypatch.setattr(pipeline_module, "CatDetector", FakeModel)
    monkeypatch.setattr(pipeline_module, "CoatClassifier", FakeModel)

    pipeline = RealtimePipeline.from_configs(
        models_config={
            "detection_model": "detector.onnx",
            "coat_classifier_model": "coat.onnx",
            "security": {"require_sha256": False},
        },
        pipeline_config={
            "events": {
                "cat_presence": {
                    "household_repeat_interval_seconds": {
                        " home_a ": "600",
                    }
                }
            }
        },
        output_dir=tmp_path,
    )

    assert pipeline.config.pass_event_intervals_by_household == {"home_a": 600.0}


@pytest.mark.parametrize(
    ("pipeline_config", "field_name"),
    [
        ({"coat": {"confidence": False}}, "coat.confidence"),
        (
            {"association": {"ambiguity_margin": False}},
            "association.ambiguity_margin",
        ),
        ({"events": {"event_confirm_ratio": False}}, "events.event_confirm_ratio"),
        (
            {"events": {"cat_presence": {"repeat_interval_seconds": False}}},
            "events.cat_presence.repeat_interval_seconds",
        ),
        (
            {"events": {"cat_presence": {"repeat_interval_seconds": 0}}},
            "events.cat_presence.repeat_interval_seconds",
        ),
        (
            {"events": {"cat_presence": {"absence_confirm_seconds": -0.1}}},
            "events.cat_presence.absence_confirm_seconds",
        ),
        (
            {
                "events": {
                    "cat_presence": {
                        "household_repeat_interval_seconds": {"home_a": False}
                    }
                }
            },
            "events.cat_presence.household_repeat_interval_seconds.home_a",
        ),
        (
            {"events": {"min_identity_observations": True}},
            "events.min_identity_observations",
        ),
        ({"output": {"max_recent_events": True}}, "output.max_recent_events"),
        ({"detection": {"confidence": -0.1}}, "detection.confidence"),
        ({"detection": {"iou": 1.1}}, "detection.iou"),
        ({"detection": {"nms_iou": -0.1}}, "detection.nms_iou"),
        ({"detection": {"imgsz": 0}}, "detection.imgsz"),
        ({"coat": {"imgsz": 0}}, "coat.imgsz"),
        ({"pose": {"keypoint_confidence": 1.1}}, "pose.keypoint_confidence"),
        ({"pose": {"min_cat_area": -1}}, "pose.min_cat_area"),
        ({"pose": {"max_instances_per_frame": 0}}, "pose.max_instances_per_frame"),
        (
            {"pose": {"posture": {"enabled": "sometimes"}}},
            "pose.posture.enabled",
        ),
        (
            {"pose": {"posture": {"min_keypoint_confidence": False}}},
            "pose.posture.min_keypoint_confidence",
        ),
        (
            {"pose": {"posture": {"min_visible_keypoints": True}}},
            "pose.posture.min_visible_keypoints",
        ),
        (
            {"pose": {"posture": {"transition_confirm_seconds": -0.1}}},
            "posture.transition_confirm_seconds",
        ),
        (
            {
                "pose": {
                    "posture": {"max_anonymous_subjects_per_source": 0}
                }
            },
            "posture.max_anonymous_subjects_per_source",
        ),
        (
            {"pose": {"quality": {"enabled": "sometimes"}}},
            "pose.quality.enabled",
        ),
        (
            {"pose": {"quality": {"point_enter_confidence": 1.1}}},
            "pose.quality.point_enter_confidence",
        ),
        (
            {"pose": {"quality": {"min_bbox_area_enter": -1}}},
            "pose.quality.min_bbox_area_enter",
        ),
        (
            {"pose": {"smoothing": {"max_bridge_gap_seconds": 0}}},
            "pose.smoothing.max_bridge_gap_seconds",
        ),
        (
            {"pose": {"smoothing": {"core_hold_frames": -1}}},
            "pose.smoothing.core_hold_frames",
        ),
        ({"tracking": {"match_iou": -0.1}}, "tracking.match_iou"),
        (
            {"tracking": {"max_center_distance_ratio": 1.1}},
            "tracking.max_center_distance_ratio",
        ),
        (
            {"tracking": {"ambiguity_margin": -0.1}},
            "tracking.ambiguity_margin",
        ),
        ({"tracking": {"max_age_seconds": 0}}, "tracking.max_age_seconds"),
        (
            {"tracking": {"bbox_smoothing_seconds": 0}},
            "tracking.bbox_smoothing_seconds",
        ),
        ({"tracking": {"max_tracks_per_source": 0}}, "tracking.max_tracks_per_source"),
    ],
)
def test_numeric_security_settings_reject_invalid_values(
    monkeypatch,
    tmp_path: Path,
    pipeline_config: dict,
    field_name: str,
) -> None:
    class FakeModel:
        def __init__(self, model_path, **kwargs) -> None:
            pass

    monkeypatch.setattr(pipeline_module, "CatDetector", FakeModel)
    monkeypatch.setattr(pipeline_module, "CoatClassifier", FakeModel)

    with pytest.raises(ValueError, match=field_name.replace(".", r"\.")):
        RealtimePipeline.from_configs(
            models_config={
                "detection_model": "detector.onnx",
                "coat_classifier_model": "coat.onnx",
                "security": {
                    "require_sha256": False,
                    "allow_pickle_weights": False,
                },
            },
            pipeline_config=pipeline_config,
            output_dir=tmp_path,
        )
