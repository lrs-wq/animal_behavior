"""Shared classification and pose overlays for live and batch video."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from .association import associate_feeding_to_cat
from .identity import IdentityDecision
from .models import CoatPrediction, Detection
from .pose import PosePrediction, draw_pose_predictions


CatDecision = tuple[Detection, CoatPrediction, IdentityDecision]


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw one compact high-contrast label."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(
        frame,
        (x, y - height - baseline - 6),
        (x + width + 8, y + 3),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 4, y - 3),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def cat_posture_label(cat: object) -> str:
    """Return committed posture, then the current observation, then unknown."""
    posture = (
        getattr(cat, "posture", "")
        or getattr(cat, "observed_posture", "")
        or "unknown"
    )
    return f"posture={posture}"


def _is_named_class(detection: Detection, names: set[str]) -> bool:
    return detection.class_name.lower() in names or str(detection.class_id) in names


def render_classification_pose_frame(
    frame: np.ndarray,
    detections: Sequence[Detection],
    cat_decisions: Sequence[CatDecision],
    poses: Sequence[PosePrediction],
    *,
    cat_classes: set[str],
    feeding_classes: set[str],
    association_max_distance_ratio: float,
    association_ambiguity_margin: float,
    pose_keypoint_confidence: float,
    cat_pose_labels: Mapping[Detection, str] | None = None,
) -> np.ndarray:
    """Render boxes, compact class/posture labels, and pose skeletons."""
    rendered = draw_pose_predictions(
        frame,
        poses,
        keypoint_threshold=pose_keypoint_confidence,
    )
    cat_by_detection = {decision[0]: decision for decision in cat_decisions}
    for detection in detections:
        if _is_named_class(detection, cat_classes):
            color = (40, 190, 40)
            decision = cat_by_detection.get(detection)
            coat_class = decision[1].class_name if decision is not None else "unknown"
            posture_label = (
                cat_pose_labels.get(detection, "posture=unknown")
                if cat_pose_labels is not None
                else "posture=unknown"
            )
            label = f"coat={coat_class} {posture_label}"
        elif _is_named_class(detection, feeding_classes):
            color = (20, 90, 230)
            association = associate_feeding_to_cat(
                detection,
                list(cat_decisions),
                max_distance_ratio=association_max_distance_ratio,
                ambiguity_margin=association_ambiguity_margin,
            )
            if association is not None:
                fx1, fy1, fx2, fy2 = detection.xyxy
                cx1, cy1, cx2, cy2 = association.cat_detection.xyxy
                cv2.line(
                    rendered,
                    ((fx1 + fx2) // 2, (fy1 + fy2) // 2),
                    ((cx1 + cx2) // 2, (cy1 + cy2) // 2),
                    color,
                    2,
                    cv2.LINE_AA,
                )
            label = detection.class_name
        else:
            color = (220, 160, 30)
            label = detection.class_name

        x1, y1, x2, y2 = detection.xyxy
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 3)
        draw_text(rendered, label, (x1, max(24, y1)), color)
    return rendered
