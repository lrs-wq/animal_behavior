"""Security policy integration tests for pipeline construction."""

from pathlib import Path

import pytest

import eating_behavior.pipeline as pipeline_module
from eating_behavior.pipeline import RealtimePipeline


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

    RealtimePipeline.from_configs(
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
        pipeline_config={"pose": {"enabled": True}},
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
    ("pipeline_config", "field_name"),
    [
        ({"coat": {"confidence": False}}, "coat.confidence"),
        (
            {"association": {"ambiguity_margin": False}},
            "association.ambiguity_margin",
        ),
        ({"events": {"event_confirm_ratio": False}}, "events.event_confirm_ratio"),
        (
            {"events": {"min_identity_observations": True}},
            "events.min_identity_observations",
        ),
        ({"output": {"max_recent_events": True}}, "output.max_recent_events"),
        ({"detection": {"confidence": -0.1}}, "detection.confidence"),
        ({"detection": {"iou": 1.1}}, "detection.iou"),
        ({"detection": {"imgsz": 0}}, "detection.imgsz"),
        ({"coat": {"imgsz": 0}}, "coat.imgsz"),
        ({"pose": {"keypoint_confidence": 1.1}}, "pose.keypoint_confidence"),
        ({"pose": {"min_cat_area": -1}}, "pose.min_cat_area"),
        ({"pose": {"max_instances_per_frame": 0}}, "pose.max_instances_per_frame"),
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
