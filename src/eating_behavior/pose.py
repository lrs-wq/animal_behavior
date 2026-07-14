"""Optional RTMPose integration for top-down cat pose estimation.

The detector owns object discovery.  This module receives only trusted cat
boxes, batches them through MMPose, and converts the framework-specific result
objects into immutable project data structures.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .models import Detection, ModelArtifactSecurityError, validate_model_artifact


AP10K_KEYPOINT_NAMES: tuple[str, ...] = (
    "L_Eye",
    "R_Eye",
    "Nose",
    "Neck",
    "Root_of_tail",
    "L_Shoulder",
    "L_Elbow",
    "L_F_Paw",
    "R_Shoulder",
    "R_Elbow",
    "R_F_Paw",
    "L_Hip",
    "L_Knee",
    "L_B_Paw",
    "R_Hip",
    "R_Knee",
    "R_B_Paw",
)

# Zero-based AP-10K links.  The training dataset and checkpoint both declare
# this exact 17-keypoint schema.
AP10K_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 3),
    (3, 4),
    (3, 5),
    (5, 6),
    (6, 7),
    (3, 8),
    (8, 9),
    (9, 10),
    (4, 11),
    (11, 12),
    (12, 13),
    (4, 14),
    (14, 15),
    (15, 16),
)


class PoseDependencyError(RuntimeError):
    """The optional OpenMMLab runtime is absent or binary-incompatible."""


class PoseModelCompatibilityError(RuntimeError):
    """A loaded pose model does not implement the expected AP-10K schema."""


class PoseInferenceError(RuntimeError):
    """One frame could not be processed by an otherwise loaded pose model."""

    def __init__(self, code: str = "pose_inference_failed") -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class PoseKeypoint:
    """One AP-10K keypoint in original-frame coordinates."""

    index: int
    name: str
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class PosePrediction:
    """One cat pose aligned with the detector box that produced it."""

    cat_xyxy: tuple[int, int, int, int]
    keypoints: tuple[PoseKeypoint, ...]
    mean_confidence: float

    def visible_keypoint_count(self, threshold: float) -> int:
        """Count keypoints meeting a caller-selected confidence threshold."""
        return sum(point.confidence >= threshold for point in self.keypoints)


class PoseEstimator(Protocol):
    """Interface consumed by the realtime pipeline."""

    def predict(
        self,
        frame: np.ndarray,
        cat_detections: Sequence[Detection],
    ) -> tuple[PosePrediction | None, ...]:
        """Return one aligned result slot for every input cat box."""


class DisabledPoseEstimator:
    """No-op implementation used when pose is explicitly disabled."""

    def predict(
        self,
        frame: np.ndarray,
        cat_detections: Sequence[Detection],
    ) -> tuple[PosePrediction | None, ...]:
        del frame
        return tuple(None for _ in cat_detections)


def normalize_mmpose_device(device: str | int | None) -> str:
    """Translate the project's YOLO device spelling to a torch device."""
    if device is None or (isinstance(device, str) and not device.strip()):
        return "cpu"
    if isinstance(device, bool):
        raise ValueError("pose device must be a CUDA index, cuda device, or cpu")
    if isinstance(device, int):
        if device < 0:
            raise ValueError("pose device index must be >= 0")
        return f"cuda:{device}"
    normalized = str(device).strip().lower()
    if normalized.isdecimal():
        return f"cuda:{int(normalized)}"
    if normalized == "cuda":
        return "cuda:0"
    if normalized == "cpu" or normalized.startswith("cuda:"):
        return normalized
    raise ValueError("pose device must be a CUDA index, cuda device, or cpu")


def _as_numpy(value: Any) -> np.ndarray:
    """Convert torch-like model output without importing torch here."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class RTMPoseEstimator:
    """MMPose top-down inference wrapper with bounded per-frame work."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        *,
        device: str | int | None = None,
        min_cat_area: int = 0,
        max_instances_per_frame: int = 8,
        allowed_roots: Iterable[str | Path] | str | Path | None = None,
        expected_config_sha256: str | None = None,
        expected_checkpoint_sha256: str | None = None,
        allow_pickle_weights: bool = False,
        api_module: Any | None = None,
    ) -> None:
        if (
            isinstance(min_cat_area, bool)
            or not isinstance(min_cat_area, int)
            or min_cat_area < 0
        ):
            raise ValueError("pose.min_cat_area must be an integer >= 0")
        if (
            isinstance(max_instances_per_frame, bool)
            or not isinstance(max_instances_per_frame, int)
            or max_instances_per_frame <= 0
        ):
            raise ValueError("pose.max_instances_per_frame must be an integer > 0")

        # MMEngine executes Python config files and MMPose deserializes the
        # checkpoint.  Verify both artifacts before importing or invoking the
        # optional framework runtime.
        self.config_path = validate_model_artifact(
            config_path,
            allowed_roots=allowed_roots,
            expected_sha256=expected_config_sha256,
            allow_pickle_weights=True,
        )
        self.checkpoint_path = validate_model_artifact(
            checkpoint_path,
            allowed_roots=allowed_roots,
            expected_sha256=expected_checkpoint_sha256,
            allow_pickle_weights=allow_pickle_weights,
        )
        if self.config_path.suffix.lower() != ".py":
            raise ModelArtifactSecurityError(
                "Pose config must be a pinned Python config file."
            )
        if self.checkpoint_path.suffix.lower() not in {".pth", ".pt"}:
            raise ModelArtifactSecurityError(
                "Pose checkpoint must be a pinned .pth or .pt file."
            )

        if api_module is None:
            try:
                api_module = importlib.import_module("mmpose.apis")
            except Exception as exc:
                raise PoseDependencyError(
                    "RTMPose dependencies are unavailable or binary-incompatible."
                ) from exc
        try:
            init_model = api_module.init_model
            self._inference_topdown = api_module.inference_topdown
        except AttributeError as exc:
            raise PoseDependencyError(
                "The installed MMPose API is incompatible."
            ) from exc

        self.device = normalize_mmpose_device(device)
        self.min_cat_area = min_cat_area
        self.max_instances_per_frame = max_instances_per_frame
        try:
            self.model = init_model(
                str(self.config_path),
                str(self.checkpoint_path),
                device=self.device,
            )
        except Exception as exc:
            raise PoseModelCompatibilityError(
                "RTMPose model initialization failed."
            ) from exc
        self._validate_dataset_meta()

    def _validate_dataset_meta(self) -> None:
        """Reject an explicitly different dataset schema at startup."""
        metadata = getattr(self.model, "dataset_meta", None)
        if not isinstance(metadata, dict):
            return
        declared_count = metadata.get("num_keypoints")
        if declared_count is not None and declared_count != len(AP10K_KEYPOINT_NAMES):
            raise PoseModelCompatibilityError(
                "RTMPose model is not a 17-keypoint AP-10K model."
            )
        id_to_name = metadata.get("keypoint_id2name")
        if isinstance(id_to_name, dict) and len(id_to_name) == len(
            AP10K_KEYPOINT_NAMES
        ):
            actual = tuple(
                str(id_to_name.get(index, "")).replace(" ", "_") for index in range(17)
            )
            expected = tuple(name.replace(" ", "_") for name in AP10K_KEYPOINT_NAMES)
            if actual != expected:
                raise PoseModelCompatibilityError(
                    "RTMPose keypoint names do not match AP-10K."
                )

    def predict(
        self,
        frame: np.ndarray,
        cat_detections: Sequence[Detection],
    ) -> tuple[PosePrediction | None, ...]:
        """Estimate all eligible cat poses with one MMPose batch call."""
        output: list[PosePrediction | None] = [None] * len(cat_detections)
        if not cat_detections:
            return tuple(output)
        if not isinstance(frame, np.ndarray) or frame.ndim < 2 or frame.size == 0:
            raise PoseInferenceError("invalid_pose_frame")

        eligible = [
            index
            for index, detection in enumerate(cat_detections)
            if detection.area >= self.min_cat_area
        ]
        eligible.sort(key=lambda index: cat_detections[index].confidence, reverse=True)
        selected = eligible[: self.max_instances_per_frame]
        if not selected:
            return tuple(output)
        bboxes = np.asarray(
            [cat_detections[index].xyxy for index in selected],
            dtype=np.float32,
        )

        try:
            results = self._inference_topdown(
                self.model,
                frame,
                bboxes=bboxes,
                bbox_format="xyxy",
            )
            if len(results) != len(selected):
                raise ValueError("pose result count mismatch")
            for detection_index, result in zip(selected, results):
                output[detection_index] = self._convert_result(
                    result,
                    cat_detections[detection_index],
                    frame.shape,
                )
        except PoseInferenceError:
            raise
        except Exception as exc:
            raise PoseInferenceError() from exc
        return tuple(output)

    @staticmethod
    def _convert_result(
        result: Any,
        detection: Detection,
        frame_shape: tuple[int, ...],
    ) -> PosePrediction:
        instances = getattr(result, "pred_instances", None)
        if instances is None:
            raise ValueError("pose result has no pred_instances")
        coordinates = _as_numpy(getattr(instances, "keypoints", None))
        scores = _as_numpy(getattr(instances, "keypoint_scores", None))
        if coordinates.ndim == 3 and coordinates.shape[0] == 1:
            coordinates = coordinates[0]
        if scores.ndim == 2 and scores.shape[0] == 1:
            scores = scores[0]
        expected_count = len(AP10K_KEYPOINT_NAMES)
        if coordinates.shape != (expected_count, 2) or scores.shape != (
            expected_count,
        ):
            raise ValueError("pose output is not AP-10K shaped")

        height, width = frame_shape[:2]
        points: list[PoseKeypoint] = []
        for index, (coordinate, raw_score) in enumerate(zip(coordinates, scores)):
            x, y = (float(coordinate[0]), float(coordinate[1]))
            score = float(raw_score)
            if not math.isfinite(score):
                score = 0.0
            score = max(0.0, min(1.0, score))
            if not math.isfinite(x) or not math.isfinite(y):
                x, y, score = 0.0, 0.0, 0.0
            else:
                x = max(0.0, min(float(width - 1), x))
                y = max(0.0, min(float(height - 1), y))
            points.append(
                PoseKeypoint(
                    index=index,
                    name=AP10K_KEYPOINT_NAMES[index],
                    x=x,
                    y=y,
                    confidence=score,
                )
            )
        mean_confidence = sum(point.confidence for point in points) / expected_count
        return PosePrediction(
            cat_xyxy=detection.xyxy,
            keypoints=tuple(points),
            mean_confidence=mean_confidence,
        )


def draw_pose_predictions(
    frame: np.ndarray,
    predictions: Sequence[PosePrediction],
    *,
    keypoint_threshold: float = 0.3,
    copy: bool = True,
) -> np.ndarray:
    """Draw AP-10K skeletons and keypoints on a BGR frame."""
    if not math.isfinite(keypoint_threshold) or not 0.0 <= keypoint_threshold <= 1.0:
        raise ValueError("pose keypoint threshold must be between 0 and 1")
    rendered = frame.copy() if copy else frame
    for prediction in predictions:
        points = prediction.keypoints
        for start, end in AP10K_SKELETON:
            first = points[start]
            second = points[end]
            if (
                first.confidence >= keypoint_threshold
                and second.confidence >= keypoint_threshold
            ):
                cv2.line(
                    rendered,
                    (int(round(first.x)), int(round(first.y))),
                    (int(round(second.x)), int(round(second.y))),
                    (255, 170, 30),
                    2,
                    cv2.LINE_AA,
                )
        for point in points:
            if point.confidence >= keypoint_threshold:
                cv2.circle(
                    rendered,
                    (int(round(point.x)), int(round(point.y))),
                    3,
                    (40, 230, 255),
                    -1,
                    cv2.LINE_AA,
                )
    return rendered
