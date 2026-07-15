"""Project-owned NMS for end-to-end detector checkpoints."""

from __future__ import annotations

import pytest

from eating_behavior.models import Detection, class_aware_nms, detection_iou


def detection(
    confidence: float,
    xyxy: tuple[int, int, int, int],
    *,
    class_id: int = 0,
    class_name: str = "cat",
) -> Detection:
    return Detection(class_id, class_name, confidence, xyxy)


def test_detection_iou_uses_half_open_box_geometry() -> None:
    first = detection(0.9, (0, 0, 10, 10))
    second = detection(0.8, (5, 0, 15, 10))

    assert detection_iou(first, second) == pytest.approx(1 / 3)


def test_class_aware_nms_keeps_highest_confidence_duplicate() -> None:
    low = detection(0.4, (0, 0, 100, 100))
    high = detection(0.9, (1, 1, 101, 101))
    separate = detection(0.7, (200, 200, 260, 260))

    assert class_aware_nms([low, high, separate], 0.7) == [high, separate]


def test_class_aware_nms_does_not_mix_cat_and_feeding_head() -> None:
    cat = detection(0.9, (0, 0, 100, 100))
    feeding_head = detection(
        0.8,
        (0, 0, 100, 100),
        class_id=1,
        class_name="feeding_head",
    )

    assert class_aware_nms([cat, feeding_head], 0.1) == [cat, feeding_head]


def test_class_aware_nms_keeps_boxes_at_exact_threshold() -> None:
    first = detection(0.9, (0, 0, 10, 10))
    second = detection(0.8, (5, 0, 15, 10))

    assert class_aware_nms([first, second], detection_iou(first, second)) == [
        first,
        second,
    ]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True])
def test_class_aware_nms_rejects_invalid_threshold(threshold) -> None:
    with pytest.raises(ValueError, match="Explicit NMS IoU"):
        class_aware_nms([], threshold)
