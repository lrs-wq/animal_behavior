"""Unit tests for the optional RTMPose adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eating_behavior.models import Detection, ModelArtifactSecurityError
from eating_behavior.pose import (
    AP10K_KEYPOINT_NAMES,
    DisabledPoseEstimator,
    PoseDependencyError,
    PoseInferenceError,
    RTMPoseEstimator,
    draw_pose_predictions,
    normalize_mmpose_device,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifacts(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "cat_pose.py"
    checkpoint = tmp_path / "cat_pose.pth"
    config.write_text("model = dict(type='TopdownPoseEstimator')\n", encoding="utf-8")
    checkpoint.write_bytes(b"trusted-test-checkpoint")
    return config, checkpoint


class FakePoseApi:
    def __init__(self, *, wrong_shape: bool = False) -> None:
        self.init_calls: list[tuple[str, str, str]] = []
        self.inference_calls: list[np.ndarray] = []
        self.wrong_shape = wrong_shape

    def init_model(self, config: str, checkpoint: str, *, device: str):
        self.init_calls.append((config, checkpoint, device))
        return SimpleNamespace(
            dataset_meta={
                "num_keypoints": 17,
                "keypoint_id2name": {
                    index: name for index, name in enumerate(AP10K_KEYPOINT_NAMES)
                },
            }
        )

    def inference_topdown(self, model, frame, *, bboxes, bbox_format):
        del model, frame
        assert bbox_format == "xyxy"
        self.inference_calls.append(bboxes.copy())
        results = []
        for _ in bboxes:
            count = 16 if self.wrong_shape else 17
            coordinates = np.arange(count * 2, dtype=np.float32).reshape(1, count, 2)
            if not self.wrong_shape:
                coordinates[0, 0] = (-5.0, 500.0)
            scores = np.linspace(0.1, 0.9, count, dtype=np.float32).reshape(1, count)
            results.append(
                SimpleNamespace(
                    pred_instances=SimpleNamespace(
                        keypoints=coordinates,
                        keypoint_scores=scores,
                    )
                )
            )
        return results


def make_estimator(
    tmp_path: Path,
    api: FakePoseApi,
    **kwargs,
) -> RTMPoseEstimator:
    config, checkpoint = artifacts(tmp_path)
    return RTMPoseEstimator(
        config,
        checkpoint,
        device="0",
        allowed_roots=[tmp_path],
        expected_config_sha256=digest(config),
        expected_checkpoint_sha256=digest(checkpoint),
        allow_pickle_weights=True,
        api_module=api,
        **kwargs,
    )


def test_pose_batches_selected_boxes_and_preserves_alignment(tmp_path: Path) -> None:
    api = FakePoseApi()
    estimator = make_estimator(
        tmp_path,
        api,
        min_cat_area=100,
        max_instances_per_frame=1,
    )
    detections = [
        Detection(0, "cat", 0.99, (0, 0, 5, 5)),  # too small
        Detection(0, "cat", 0.70, (1, 1, 21, 21)),
        Detection(0, "cat", 0.90, (10, 10, 40, 40)),
    ]

    predictions = estimator.predict(np.zeros((64, 64, 3), dtype=np.uint8), detections)

    assert len(predictions) == 3
    assert predictions[0] is None
    assert predictions[1] is None
    assert predictions[2] is not None
    assert predictions[2].cat_xyxy == detections[2].xyxy
    assert len(predictions[2].keypoints) == 17
    assert predictions[2].keypoints[0].x == 0.0
    assert predictions[2].keypoints[0].y == 63.0
    assert predictions[2].keypoints[0].confidence == 0.0
    np.testing.assert_array_equal(
        api.inference_calls[0],
        np.asarray([[10, 10, 40, 40]], dtype=np.float32),
    )
    assert api.init_calls[0][2] == "cuda:0"


def test_pose_rejects_non_ap10k_output_as_frame_error(tmp_path: Path) -> None:
    estimator = make_estimator(tmp_path, FakePoseApi(wrong_shape=True))

    with pytest.raises(PoseInferenceError, match="pose_inference_failed"):
        estimator.predict(
            np.zeros((64, 64, 3), dtype=np.uint8),
            [Detection(0, "cat", 0.9, (0, 0, 32, 32))],
        )


def test_pose_hash_is_checked_before_framework_initialization(tmp_path: Path) -> None:
    config, checkpoint = artifacts(tmp_path)
    api = FakePoseApi()

    with pytest.raises(ModelArtifactSecurityError, match="integrity"):
        RTMPoseEstimator(
            config,
            checkpoint,
            allowed_roots=[tmp_path],
            expected_config_sha256="0" * 64,
            expected_checkpoint_sha256=digest(checkpoint),
            allow_pickle_weights=True,
            api_module=api,
        )

    assert api.init_calls == []


def test_missing_framework_has_clear_optional_dependency_error(
    monkeypatch, tmp_path: Path
) -> None:
    config, checkpoint = artifacts(tmp_path)

    def fail_import(name: str):
        assert name == "mmpose.apis"
        raise ImportError("missing")

    monkeypatch.setattr("eating_behavior.pose.importlib.import_module", fail_import)
    with pytest.raises(PoseDependencyError, match="dependencies"):
        RTMPoseEstimator(
            config,
            checkpoint,
            allowed_roots=[tmp_path],
            expected_config_sha256=digest(config),
            expected_checkpoint_sha256=digest(checkpoint),
            allow_pickle_weights=True,
        )


def test_disabled_pose_returns_one_empty_slot_per_cat() -> None:
    detections = [
        Detection(0, "cat", 0.9, (0, 0, 10, 10)),
        Detection(0, "cat", 0.8, (10, 10, 20, 20)),
    ]
    assert DisabledPoseEstimator().predict(
        np.zeros((20, 20, 3), dtype=np.uint8), detections
    ) == (None, None)


@pytest.mark.parametrize(
    ("device", "expected"),
    [(None, "cpu"), (0, "cuda:0"), ("1", "cuda:1"), ("cuda", "cuda:0"), ("cpu", "cpu")],
)
def test_normalize_mmpose_device(device, expected: str) -> None:
    assert normalize_mmpose_device(device) == expected


def test_pose_renderer_draws_only_confident_points(tmp_path: Path) -> None:
    estimator = make_estimator(tmp_path, FakePoseApi())
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    prediction = estimator.predict(
        frame,
        [Detection(0, "cat", 0.9, (0, 0, 32, 32))],
    )[0]
    assert prediction is not None

    rendered = draw_pose_predictions(frame, [prediction], keypoint_threshold=0.5)

    assert np.count_nonzero(rendered) > 0
    assert np.count_nonzero(frame) == 0
