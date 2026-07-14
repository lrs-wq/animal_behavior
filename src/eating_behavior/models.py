"""Thin wrappers around external model weights.

This repository does not train models. It only loads detector/classifier
artifacts and converts their outputs into stable project data structures.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Ultralytics can otherwise attempt to install missing optional dependencies while
# loading a model. Workers must never mutate their Python environment as a side
# effect of importing or loading an artifact.
os.environ["YOLO_AUTOINSTALL"] = "false"

from ultralytics import YOLO  # noqa: E402


_PICKLE_MODEL_SUFFIXES = frozenset({".pt", ".pth"})
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


class ModelArtifactSecurityError(ValueError):
    """A model artifact failed a local trust-boundary check."""


def _resolved_directory(path: str | Path) -> Path:
    """Resolve one policy root without exposing it through an OS exception."""
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise ModelArtifactSecurityError("Model artifact policy is invalid.")
    except ModelArtifactSecurityError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError):
        raise ModelArtifactSecurityError("Model artifact policy is invalid.") from None
    return resolved


def _is_within_root(candidate: Path, root: Path) -> bool:
    """Return whether a resolved path is below root, including on Windows."""
    candidate_text = os.path.normcase(str(candidate))
    root_text = os.path.normcase(str(root))
    try:
        return os.path.commonpath((candidate_text, root_text)) == root_text
    except ValueError:
        # Different Windows drives, for example, can have no common path.
        return False


def _sha256(path: Path) -> str:
    """Hash an artifact without returning filesystem details on read errors."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        raise ModelArtifactSecurityError("Model artifact could not be verified.") from None
    return digest.hexdigest()


def validate_model_artifact(
    model_path: str | Path,
    *,
    allowed_roots: Iterable[str | Path] | str | Path | None = None,
    expected_sha256: str | None = None,
    allow_pickle_weights: bool = True,
) -> Path:
    """Validate a model artifact before an external loader can deserialize it.

    ``.pt`` and ``.pth`` files may use Python pickle internally. They remain
    enabled by default for backwards compatibility, but production callers can
    explicitly reject them with ``allow_pickle_weights=False``. If provided,
    ``expected_sha256`` must be an exact 64-character hexadecimal digest.
    """
    if not isinstance(allow_pickle_weights, bool):
        raise ModelArtifactSecurityError("Model artifact policy is invalid.")
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or _SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        raise ModelArtifactSecurityError("Model artifact policy is invalid.")

    try:
        candidate = Path(model_path).expanduser()
        metadata = candidate.lstat()
    except (FileNotFoundError, NotADirectoryError):
        raise FileNotFoundError("Model artifact is unavailable.") from None
    except (OSError, RuntimeError, ValueError, TypeError):
        raise ModelArtifactSecurityError("Model artifact is unavailable.") from None

    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ModelArtifactSecurityError(
            "Model artifact must be a regular, non-linked file."
        )

    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ModelArtifactSecurityError("Model artifact is unavailable.") from None

    if allowed_roots is not None:
        roots: Iterable[str | Path]
        if isinstance(allowed_roots, (str, Path)):
            roots = (allowed_roots,)
        else:
            roots = allowed_roots
        try:
            resolved_roots = tuple(_resolved_directory(root) for root in roots)
        except TypeError:
            raise ModelArtifactSecurityError("Model artifact policy is invalid.") from None
        if not resolved_roots or not any(
            _is_within_root(resolved, root) for root in resolved_roots
        ):
            raise ModelArtifactSecurityError(
                "Model artifact is outside the permitted directories."
            )

    if not allow_pickle_weights and resolved.suffix.lower() in _PICKLE_MODEL_SUFFIXES:
        raise ModelArtifactSecurityError("Pickle-based model artifacts are disabled.")

    if expected_sha256 is not None:
        actual_sha256 = _sha256(resolved)
        if not hmac.compare_digest(actual_sha256, expected_sha256.lower()):
            raise ModelArtifactSecurityError("Model artifact integrity check failed.")

    return resolved


@dataclass(frozen=True)
class Detection:
    """One object detection result."""

    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[int, int, int, int]

    @property
    def area(self) -> int:
        """Bounding-box area in pixels."""
        x1, y1, x2, y2 = self.xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass(frozen=True)
class CoatPrediction:
    """Coat/appearance classifier result for one cat crop."""

    class_id: int
    class_name: str
    confidence: float
    second_class_name: str
    second_confidence: float


def class_name(names: Any, class_id: int) -> str:
    """Read a class name from ultralytics dict/list metadata."""
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


class CatDetector:
    """YOLO wrapper for cat and feeding_head detection."""

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
        iou: float = 0.70,
        imgsz: int = 640,
        device: str | None = None,
        *,
        allowed_roots: Iterable[str | Path] | str | Path | None = None,
        expected_sha256: str | None = None,
        allow_pickle_weights: bool = True,
    ) -> None:
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("Detection confidence must be between 0 and 1")
        if (
            isinstance(iou, bool)
            or not isinstance(iou, (int, float))
            or not math.isfinite(iou)
            or not 0.0 <= iou <= 1.0
        ):
            raise ValueError("Detection IoU must be between 0 and 1")
        if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
            raise ValueError("Detection imgsz must be a positive integer")
        self.model_path = validate_model_artifact(
            model_path,
            allowed_roots=allowed_roots,
            expected_sha256=expected_sha256,
            allow_pickle_weights=allow_pickle_weights,
        )
        self.model = YOLO(str(self.model_path))
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run object detection on a frame."""
        kwargs: dict[str, Any] = {
            "imgsz": self.imgsz,
            "conf": self.confidence,
            "iou": self.iou,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        result = self.model.predict(frame, **kwargs)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        names = getattr(result, "names", {})
        h, w = frame.shape[:2]
        detections: list[Detection] = []
        for xyxy, cls_value, conf_value in zip(
            boxes.xyxy.detach().cpu().numpy(),
            boxes.cls.detach().cpu().numpy(),
            boxes.conf.detach().cpu().numpy(),
        ):
            x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
            # Clamp model boxes to valid image bounds before cropping.
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(cls_value)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name(names, class_id),
                    confidence=float(conf_value),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return detections


class CoatClassifier:
    """YOLO classification wrapper for cat coat/appearance class."""

    def __init__(
        self,
        model_path: str | Path,
        imgsz: int = 128,
        device: str | None = None,
        *,
        allowed_roots: Iterable[str | Path] | str | Path | None = None,
        expected_sha256: str | None = None,
        allow_pickle_weights: bool = True,
    ) -> None:
        if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
            raise ValueError("Coat classifier imgsz must be a positive integer")
        self.model_path = validate_model_artifact(
            model_path,
            allowed_roots=allowed_roots,
            expected_sha256=expected_sha256,
            allow_pickle_weights=allow_pickle_weights,
        )
        self.model = YOLO(str(self.model_path))
        self.imgsz = imgsz
        self.device = device

    def classify(self, crop: np.ndarray) -> CoatPrediction:
        """Classify a cat crop into a coat/appearance class."""
        kwargs: dict[str, Any] = {"imgsz": self.imgsz, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        result = self.model.predict(crop, **kwargs)[0]
        probs = getattr(result, "probs", None)
        if probs is None:
            # Fallback keeps the pipeline alive if the classifier output is invalid.
            return CoatPrediction(-1, "unknown", 0.0, "unknown", 0.0)
        values = probs.data.detach().cpu().numpy().astype(float)
        order = np.argsort(values)[::-1]
        names = getattr(result, "names", {})
        top = int(order[0])
        second = int(order[1]) if len(order) > 1 else top
        return CoatPrediction(
            class_id=top,
            class_name=class_name(names, top),
            confidence=float(values[top]),
            second_class_name=class_name(names, second),
            second_confidence=float(values[second]) if second != top else 0.0,
        )

