"""Pure spatial association between feeding-head and cat detections."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .identity import IdentityDecision
from .models import CoatPrediction, Detection


CatDecision = tuple[Detection, CoatPrediction, IdentityDecision]


@dataclass(frozen=True)
class FeedingAssociation:
    """A spatially supported feeding-head to cat association."""

    feeding: Detection
    cat_detection: Detection
    coat: CoatPrediction
    identity: IdentityDecision
    score: float
    center_distance: float
    normalized_distance: float
    feeding_coverage: float
    feeding_center_inside: bool


def _center(detection: Detection) -> tuple[float, float]:
    x1, y1, x2, y2 = detection.xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _point_to_rectangle_distance(
    point: tuple[float, float],
    rectangle: tuple[int, int, int, int],
) -> float:
    """Return the true Euclidean distance from a point to a rectangle."""
    px, py = point
    x1, y1, x2, y2 = rectangle
    dx = max(float(x1) - px, 0.0, px - float(x2))
    dy = max(float(y1) - py, 0.0, py - float(y2))
    return math.hypot(dx, dy)


def _intersection_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return float(max(0, right - left) * max(0, bottom - top))


def _feeding_coverage(feeding: Detection, cat: Detection) -> float:
    if feeding.area <= 0:
        return 0.0
    return _intersection_area(feeding.xyxy, cat.xyxy) / feeding.area


def _candidate(
    feeding: Detection,
    cat_detection: Detection,
    coat: CoatPrediction,
    identity: IdentityDecision,
    max_distance_ratio: float,
) -> FeedingAssociation | None:
    if feeding.area <= 0 or cat_detection.area <= 0:
        return None
    center = _center(feeding)
    x1, y1, x2, y2 = cat_detection.xyxy
    inside = x1 <= center[0] <= x2 and y1 <= center[1] <= y2
    center_distance = _point_to_rectangle_distance(center, cat_detection.xyxy)
    diagonal = math.hypot(max(0, x2 - x1), max(0, y2 - y1))
    if diagonal <= 0.0:
        return None
    normalized_distance = center_distance / diagonal
    coverage = _feeding_coverage(feeding, cat_detection)

    # Center containment is direct spatial evidence. Otherwise enforce the
    # configured maximum distance even when two poorly localized boxes barely
    # overlap; coverage still contributes to ranking reasonable candidates.
    if not inside and normalized_distance > max_distance_ratio:
        return None

    if max_distance_ratio == 0.0:
        proximity = 1.0 if center_distance == 0.0 else 0.0
    else:
        proximity = max(0.0, 1.0 - normalized_distance / max_distance_ratio)
    score = (2.0 if inside else 0.0) + coverage + proximity
    return FeedingAssociation(
        feeding=feeding,
        cat_detection=cat_detection,
        coat=coat,
        identity=identity,
        score=score,
        center_distance=center_distance,
        normalized_distance=normalized_distance,
        feeding_coverage=coverage,
        feeding_center_inside=inside,
    )


def associate_feeding_to_cat(
    feeding: Detection,
    cat_decisions: Sequence[CatDecision],
    *,
    max_distance_ratio: float = 0.5,
    ambiguity_margin: float = 0.1,
) -> FeedingAssociation | None:
    """Return the best spatial association, or ``None`` when unsupported.

    ``max_distance_ratio`` compares feeding-center distance with each cat-box
    diagonal. ``ambiguity_margin`` is the minimum absolute score lead required
    over the second-best candidate; an exact tie is ambiguous even when the
    configured margin is zero.
    """
    if not math.isfinite(max_distance_ratio) or max_distance_ratio < 0.0:
        raise ValueError("max_distance_ratio must be a finite value >= 0")
    if not math.isfinite(ambiguity_margin) or ambiguity_margin < 0.0:
        raise ValueError("ambiguity_margin must be a finite value >= 0")

    candidates = [
        candidate
        for cat_detection, coat, identity in cat_decisions
        if (
            candidate := _candidate(
                feeding,
                cat_detection,
                coat,
                identity,
                max_distance_ratio,
            )
        )
        is not None
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    if (
        len(candidates) > 1
        and candidates[0].score - candidates[1].score <= ambiguity_margin
    ):
        return None
    return candidates[0]
