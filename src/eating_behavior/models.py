from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[int, int, int, int]

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass(frozen=True)
class CoatPrediction:
    class_id: int
    class_name: str
    confidence: float
    second_class_name: str
    second_confidence: float


def class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


class CatDetector:
    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
        iou: float = 0.70,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Detection model does not exist: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
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
    def __init__(
        self,
        model_path: str | Path,
        imgsz: int = 128,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Coat classifier model does not exist: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.imgsz = imgsz
        self.device = device

    def classify(self, crop: np.ndarray) -> CoatPrediction:
        kwargs: dict[str, Any] = {"imgsz": self.imgsz, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        result = self.model.predict(crop, **kwargs)[0]
        probs = getattr(result, "probs", None)
        if probs is None:
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

