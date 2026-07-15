#!/usr/bin/env python3
"""Build a leak-audited RTMPose dataset from local-camera human labels.

The v4 pose labels were annotated on detector crops.  This builder maps those
human keypoints back to the original local-camera frame and uses frozen boxes
from the production detector as top-down *input proposals*.  Detector outputs
are never interpreted as keypoint labels.

Safety invariants:

* the detector artifact is SHA256 pinned before the pickle-backed weight loads;
* production detections pass an explicit class-aware NMS stage;
* instances in a frame are matched one-to-one (Hungarian assignment);
* every match passes source-anchor IoU and padded-keypoint-coverage gates;
* train/val/test are split by source ``clip_id``, never by individual frame;
* any gate failure aborts the default build instead of silently dropping data;
* all non-v4 clean labels, when requested, are training-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import cv2
import numpy as np


SOURCE_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "neck",
    "tail_root",
    "left_shoulder",
    "left_elbow",
    "left_front_paw",
    "right_shoulder",
    "right_elbow",
    "right_front_paw",
    "left_hip",
    "left_knee",
    "left_back_paw",
    "right_hip",
    "right_knee",
    "right_back_paw",
]

AP10K_KEYPOINTS = [
    "L_Eye",
    "R_Eye",
    "Nose",
    "Neck",
    "Root of tail",
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
]

AP10K_FROM_SOURCE = [1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# COCO skeleton edges use one-based keypoint indexes.
AP10K_SKELETON = [
    [1, 2],
    [1, 3],
    [2, 3],
    [3, 4],
    [4, 5],
    [4, 6],
    [6, 7],
    [7, 8],
    [4, 9],
    [9, 10],
    [10, 11],
    [5, 12],
    [12, 13],
    [13, 14],
    [5, 15],
    [15, 16],
    [16, 17],
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
V4_STEM_RE = re.compile(
    r"(?:(?:train|val|test)__)?"
    r"(?P<full_stem>v4__[0-9a-fA-F]{8}-frame_(?P<serial>[0-9]+))"
    r"__ann(?P<ann>[0-9]+)__cls(?P<class_id>[0-9]+)"
)
FRAME_STEM_RE = re.compile(r"frame_(?P<serial>[0-9]+)$")


class BuildError(RuntimeError):
    """The source data or requested build is unsafe or inconsistent."""


class GateFailure(BuildError):
    """At least one camera instance failed proposal matching/quality gates."""

    def __init__(self, rejections: list[dict[str, Any]]) -> None:
        self.rejections = rejections
        reasons = Counter(
            reason
            for rejection in rejections
            for reason in str(rejection.get("reasons", "unknown")).split(";")
            if reason
        )
        super().__init__(
            f"{len(rejections)} camera pose instance(s) failed mandatory gates: "
            f"{dict(sorted(reasons.items()))}"
        )


@dataclass(frozen=True)
class Detection:
    """One upstream detector output before this builder's explicit NMS."""

    bbox_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int


class ProposalProvider(Protocol):
    """A detector or frozen cache that returns upstream box detections."""

    @property
    def cat_class_id(self) -> int: ...

    @property
    def descriptor(self) -> Mapping[str, Any]: ...

    def predict(self, images: Mapping[str, Path]) -> Mapping[str, list[Detection]]: ...


@dataclass
class BuildConfig:
    pose_root: Path
    metadata_csv: Path
    full_frame_root: Path
    output: Path
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    split_seed: int = 21
    explicit_nms_iou: float = 0.70
    proposal_min_score: float = 0.25
    min_source_iou: float = 0.50
    proposal_padding: float = 1.25
    min_keypoint_coverage: float = 1.0
    include_non_v4_train: bool = False
    allow_review_rejections: bool = False
    image_mode: str = "hardlink"
    video_root: Path | None = None
    verify_video_pixels: bool = False
    max_video_mae: float = 3.0

    def validate(self) -> None:
        for path, label in (
            (self.pose_root, "pose root"),
            (self.full_frame_root, "full-frame root"),
        ):
            if not path.is_dir():
                raise BuildError(f"missing {label}: {path}")
        if not self.metadata_csv.is_file():
            raise BuildError(f"missing metadata CSV: {self.metadata_csv}")
        if self.output.exists() or self.output.is_symlink():
            raise BuildError(f"refusing to overwrite existing output: {self.output}")
        if self.image_mode not in {"copy", "hardlink"}:
            raise BuildError("image_mode must be 'copy' or 'hardlink'")
        if len(self.split_ratios) != 3 or any(value <= 0 for value in self.split_ratios):
            raise BuildError("split ratios must contain three positive values")
        if not math.isclose(sum(self.split_ratios), 1.0, abs_tol=1e-9):
            raise BuildError(f"split ratios must sum to 1, got {self.split_ratios}")
        if not 0 < self.explicit_nms_iou < 1:
            raise BuildError("explicit_nms_iou must be in (0, 1)")
        if not 0 <= self.proposal_min_score <= 1:
            raise BuildError("proposal_min_score must be in [0, 1]")
        if not 0 <= self.min_source_iou <= 1:
            raise BuildError("min_source_iou must be in [0, 1]")
        if self.proposal_padding < 1:
            raise BuildError("proposal_padding must be >= 1")
        if not 0 <= self.min_keypoint_coverage <= 1:
            raise BuildError("min_keypoint_coverage must be in [0, 1]")
        if self.verify_video_pixels and self.video_root is None:
            raise BuildError("--verify-video-pixels requires --video-root")
        if self.video_root is not None and not self.video_root.is_dir():
            raise BuildError(f"missing video root: {self.video_root}")


@dataclass
class CameraInstance:
    instance_key: str
    pose_label: Path
    pose_image: Path
    pose_label_relative: str
    full_stem: str
    full_image: Path
    full_label: Path
    full_width: int
    full_height: int
    frame_serial: str
    source_ann_index: int
    source_class_id: int
    source_anchor_xyxy: tuple[float, float, float, float]
    keypoints: list[float]
    num_keypoints: int
    metadata: dict[str, str]
    manifest: dict[str, str]


@dataclass
class MatchedInstance:
    instance: CameraInstance
    proposal_xyxy: tuple[float, float, float, float]
    padded_proposal_xyxy: tuple[float, float, float, float]
    proposal_score: float
    source_iou: float
    keypoint_coverage: float
    raw_detection_count: int
    post_nms_detection_count: int
    split: str = ""


@dataclass
class GenericInstance:
    source_label: Path
    source_image: Path
    source_relative: str
    output_name: str
    width: int
    height: int
    bbox_xywh: list[float]
    keypoints: list[float]
    num_keypoints: int
    manifest: dict[str, str]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, label: str) -> str:
    normalized = value.lower().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise BuildError(f"{label} must be a 64-character SHA256 digest")
    return normalized


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise BuildError(f"cannot decode image: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def _safe_relative_path(base: Path, raw: str) -> Path:
    relative = Path(raw.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise BuildError(f"unsafe manifest path: {raw!r}")
    resolved_base = base.resolve()
    resolved = (base / relative).resolve()
    if os.path.commonpath((str(resolved_base), str(resolved))) != str(resolved_base):
        raise BuildError(f"manifest path escapes dataset root: {raw!r}")
    return resolved


def _image_stem_from_metadata(raw: str) -> str:
    name = re.split(r"[\\/]", raw.strip())[-1]
    return Path(name).stem


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    required = {
        "image",
        "source_video",
        "clip_id",
        "frame_index",
        "video_time_sec",
        "fps",
    }
    by_image_stem: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise BuildError(f"metadata CSV lacks required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            stem = _image_stem_from_metadata(row["image"])
            if not FRAME_STEM_RE.fullmatch(stem):
                raise BuildError(f"{path}:{row_number}: invalid metadata image stem {stem!r}")
            if stem in by_image_stem:
                raise BuildError(f"{path}:{row_number}: duplicate metadata image {stem!r}")
            if not row["clip_id"].strip():
                raise BuildError(f"{path}:{row_number}: empty clip_id")
            try:
                frame_index = int(row["frame_index"])
                fps = float(row["fps"])
            except ValueError as exc:
                raise BuildError(f"{path}:{row_number}: invalid numeric provenance") from exc
            if frame_index < 0 or fps <= 0:
                raise BuildError(f"{path}:{row_number}: invalid frame_index/fps")
            normalized = {key: (value or "").strip() for key, value in row.items()}
            normalized["metadata_row"] = str(row_number)
            by_image_stem[stem] = normalized
    if not by_image_stem:
        raise BuildError(f"empty metadata CSV: {path}")
    return by_image_stem


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "source", "decision"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise BuildError(f"manifest lacks required columns: {sorted(missing)}")
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise BuildError(f"empty manifest: {path}")
    return rows


def _index_by_stem(root: Path, suffixes: set[str]) -> dict[str, Path]:
    if not root.is_dir():
        raise BuildError(f"missing source directory: {root}")
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in index:
            duplicates[path.stem].extend([index[path.stem], path])
        else:
            index[path.stem] = path
    if duplicates:
        sample = {key: [str(p) for p in value] for key, value in list(duplicates.items())[:3]}
        raise BuildError(f"duplicate source stems under {root}: {sample}")
    return index


def _find_pose_image(pose_root: Path, label_relative: str, stem: str) -> Path:
    relative = Path(label_relative.replace("\\", "/"))
    if not relative.parts or relative.parts[0] != "labels":
        raise BuildError(f"pose label is not below labels/: {label_relative}")
    directory = pose_root / "images" / Path(*relative.parts[1:-1])
    candidates = sorted(
        path
        for path in directory.glob(f"{stem}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(candidates) != 1:
        raise BuildError(
            f"expected exactly one pose image for {label_relative}, found {len(candidates)}"
        )
    return candidates[0].resolve()


def _nonempty_lines(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def yolo_box_to_xyxy(
    values: Sequence[float], width: int, height: int, *, integer_crop: bool = False
) -> tuple[float, float, float, float]:
    if len(values) != 4 or not all(math.isfinite(float(value)) for value in values):
        raise BuildError(f"invalid YOLO box: {values}")
    cx, cy, box_width, box_height = (float(value) for value in values)
    if box_width <= 0 or box_height <= 0:
        raise BuildError(f"non-positive YOLO box: {values}")
    raw = (
        (cx - box_width / 2) * width,
        (cy - box_height / 2) * height,
        (cx + box_width / 2) * width,
        (cy + box_height / 2) * height,
    )
    if integer_crop:
        raw = (math.floor(raw[0]), math.floor(raw[1]), math.ceil(raw[2]), math.ceil(raw[3]))
    x1 = _clip(float(raw[0]), 0.0, float(width))
    y1 = _clip(float(raw[1]), 0.0, float(height))
    x2 = _clip(float(raw[2]), 0.0, float(width))
    y2 = _clip(float(raw[3]), 0.0, float(height))
    if x2 <= x1 or y2 <= y1:
        raise BuildError(f"empty YOLO box after clipping: {values}")
    return x1, y1, x2, y2


def xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    return [x1, y1, x2 - x1, y2 - y1]


def _parse_pose_values(line: str, label: Path) -> list[float]:
    try:
        values = [float(token) for token in line.split()]
    except ValueError as exc:
        raise BuildError(f"non-numeric pose label: {label}") from exc
    expected = 1 + 4 + len(SOURCE_KEYPOINTS) * 3
    if len(values) != expected:
        raise BuildError(f"{label}: expected {expected} pose values, got {len(values)}")
    if int(round(values[0])) != 0:
        raise BuildError(f"{label}: expected cat class 0, got {values[0]}")
    return values


def _human_keypoints_from_crop(
    values: Sequence[float],
    crop_box_xyxy: Sequence[float],
    full_width: int,
    full_height: int,
) -> tuple[list[float], int]:
    x1, y1, x2, y2 = (float(value) for value in crop_box_xyxy)
    crop_width = x2 - x1
    crop_height = y2 - y1
    source = [values[5 + index * 3 : 8 + index * 3] for index in range(17)]
    keypoints: list[float] = []
    num_keypoints = 0
    for source_index in AP10K_FROM_SOURCE:
        x_norm, y_norm, visibility_raw = (float(value) for value in source[source_index])
        visibility = int(round(visibility_raw))
        if visibility not in (0, 1, 2) or not math.isclose(
            visibility_raw, visibility, abs_tol=1e-6
        ):
            raise BuildError(f"invalid keypoint visibility {visibility_raw}")
        if visibility == 0:
            x = y = 0.0
        else:
            if not (math.isfinite(x_norm) and math.isfinite(y_norm)):
                raise BuildError("non-finite human keypoint coordinate")
            if not (-1e-6 <= x_norm <= 1 + 1e-6 and -1e-6 <= y_norm <= 1 + 1e-6):
                raise BuildError(f"human keypoint outside annotated crop: {(x_norm, y_norm)}")
            x = _clip(x1 + x_norm * crop_width, 0.0, max(0.0, full_width - 1.0))
            y = _clip(y1 + y_norm * crop_height, 0.0, max(0.0, full_height - 1.0))
            num_keypoints += 1
        keypoints.extend([x, y, visibility])
    if num_keypoints == 0:
        raise BuildError("pose annotation has no labeled human keypoints")
    return keypoints, num_keypoints


def _human_keypoints_in_image(
    values: Sequence[float], width: int, height: int
) -> tuple[list[float], int]:
    return _human_keypoints_from_crop(values, (0.0, 0.0, float(width), float(height)), width, height)


def load_camera_instances(
    config: BuildConfig,
    manifest_rows: Sequence[dict[str, str]],
    metadata: Mapping[str, dict[str, str]],
) -> list[CameraInstance]:
    full_images = _index_by_stem(config.full_frame_root / "images", IMAGE_SUFFIXES)
    full_labels = _index_by_stem(config.full_frame_root / "labels", {".txt"})
    instances: list[CameraInstance] = []
    accepted = [
        row
        for row in manifest_rows
        if row.get("source", "").lower() == "v4"
        and row.get("decision", "").lower() == "accepted"
    ]
    if not accepted:
        raise BuildError("manifest contains no accepted v4 camera annotations")

    for row in accepted:
        relative = row["path"]
        pose_label = _safe_relative_path(config.pose_root, relative)
        if not pose_label.is_file():
            raise BuildError(f"missing pose label: {pose_label}")
        match = V4_STEM_RE.fullmatch(pose_label.stem)
        if match is None:
            raise BuildError(f"unrecognized v4 pose label stem: {pose_label.stem}")
        full_stem = match.group("full_stem")
        source_ann_index = int(match.group("ann"))
        source_class_id = int(match.group("class_id"))
        if source_class_id != 0:
            raise BuildError(f"expected v4 cat class 0: {pose_label}")
        frame_stem = f"frame_{int(match.group('serial')):06d}"
        metadata_row = metadata.get(frame_stem)
        if metadata_row is None:
            raise BuildError(f"no metadata mapping for {pose_label.name} ({frame_stem})")
        full_image = full_images.get(full_stem)
        full_label = full_labels.get(full_stem)
        if full_image is None or full_label is None:
            raise BuildError(f"missing original full frame/label for {full_stem}")
        full_width, full_height = _read_image_size(full_image)
        full_label_lines = _nonempty_lines(full_label)
        if source_ann_index >= len(full_label_lines):
            raise BuildError(
                f"{full_label}: annotation index {source_ann_index} is out of range"
            )
        try:
            source_values = [float(token) for token in full_label_lines[source_ann_index].split()]
        except ValueError as exc:
            raise BuildError(f"non-numeric source box: {full_label}") from exc
        if len(source_values) < 5:
            raise BuildError(f"malformed source box: {full_label}")
        if int(round(source_values[0])) != source_class_id:
            raise BuildError(f"source class does not match pose filename: {pose_label}")
        source_anchor = yolo_box_to_xyxy(
            source_values[1:5], full_width, full_height, integer_crop=True
        )

        pose_image = _find_pose_image(config.pose_root, relative, pose_label.stem)
        crop_width, crop_height = _read_image_size(pose_image)
        expected_crop_size = (
            int(round(source_anchor[2] - source_anchor[0])),
            int(round(source_anchor[3] - source_anchor[1])),
        )
        if (crop_width, crop_height) != expected_crop_size:
            raise BuildError(
                f"{pose_image}: crop size {(crop_width, crop_height)} does not match "
                f"source-anchor geometry {expected_crop_size}"
            )
        pose_lines = _nonempty_lines(pose_label)
        try:
            line_number = int(row.get("line") or "1")
        except ValueError as exc:
            raise BuildError(f"invalid manifest line number for {pose_label}") from exc
        if line_number < 1 or line_number > len(pose_lines):
            raise BuildError(f"manifest line {line_number} is out of range for {pose_label}")
        values = _parse_pose_values(pose_lines[line_number - 1], pose_label)
        keypoints, num_keypoints = _human_keypoints_from_crop(
            values, source_anchor, full_width, full_height
        )
        instances.append(
            CameraInstance(
                instance_key=relative.replace("\\", "/"),
                pose_label=pose_label,
                pose_image=pose_image,
                pose_label_relative=relative.replace("\\", "/"),
                full_stem=full_stem,
                full_image=full_image,
                full_label=full_label,
                full_width=full_width,
                full_height=full_height,
                frame_serial=frame_stem,
                source_ann_index=source_ann_index,
                source_class_id=source_class_id,
                source_anchor_xyxy=source_anchor,
                keypoints=keypoints,
                num_keypoints=num_keypoints,
                metadata=dict(metadata_row),
                manifest=dict(row),
            )
        )

    keys = [instance.instance_key for instance in instances]
    if len(keys) != len(set(keys)):
        raise BuildError("duplicate accepted v4 pose-label paths in manifest")
    return sorted(instances, key=lambda item: item.instance_key)


def load_generic_instances(
    config: BuildConfig, manifest_rows: Sequence[dict[str, str]]
) -> list[GenericInstance]:
    if not config.include_non_v4_train:
        return []
    rows = [
        row
        for row in manifest_rows
        if row.get("source", "").lower() != "v4"
        and row.get("decision", "").lower() == "accepted"
    ]
    instances: list[GenericInstance] = []
    output_names: set[str] = set()
    for row in rows:
        relative = row["path"]
        label = _safe_relative_path(config.pose_root, relative)
        if not label.is_file():
            raise BuildError(f"missing generic pose label: {label}")
        image = _find_pose_image(config.pose_root, relative, label.stem)
        width, height = _read_image_size(image)
        lines = _nonempty_lines(label)
        try:
            line_number = int(row.get("line") or "1")
        except ValueError as exc:
            raise BuildError(f"invalid manifest line number for {label}") from exc
        if line_number < 1 or line_number > len(lines):
            raise BuildError(f"manifest line {line_number} is out of range for {label}")
        values = _parse_pose_values(lines[line_number - 1], label)
        bbox = xyxy_to_xywh(yolo_box_to_xyxy(values[1:5], width, height))
        keypoints, num_keypoints = _human_keypoints_in_image(values, width, height)
        name_hash = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:10]
        output_name = f"generic__{name_hash}__{image.name}"
        if output_name in output_names:
            raise BuildError(f"generic output filename collision: {output_name}")
        output_names.add(output_name)
        instances.append(
            GenericInstance(
                source_label=label,
                source_image=image,
                source_relative=relative.replace("\\", "/"),
                output_name=output_name,
                width=width,
                height=height,
                bbox_xywh=bbox,
                keypoints=keypoints,
                num_keypoints=num_keypoints,
                manifest=dict(row),
            )
        )
    return sorted(instances, key=lambda item: item.source_relative)


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def class_aware_nms(detections: Sequence[Detection], iou_threshold: float) -> list[Detection]:
    """Apply deterministic, explicit NMS independently for every class."""

    if not 0 < iou_threshold < 1:
        raise BuildError("NMS IoU threshold must be in (0, 1)")
    validated: list[tuple[int, Detection]] = []
    for index, detection in enumerate(detections):
        values = (*detection.bbox_xyxy, detection.score)
        if not all(math.isfinite(float(value)) for value in values):
            raise BuildError("detector returned non-finite box/score")
        x1, y1, x2, y2 = detection.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise BuildError(f"detector returned empty box: {detection.bbox_xyxy}")
        if not 0 <= detection.score <= 1:
            raise BuildError(f"detector returned invalid score: {detection.score}")
        validated.append((index, detection))

    kept: list[tuple[int, Detection]] = []
    by_class: dict[int, list[tuple[int, Detection]]] = defaultdict(list)
    for indexed in validated:
        by_class[indexed[1].class_id].append(indexed)
    for class_id in sorted(by_class):
        remaining = sorted(by_class[class_id], key=lambda item: (-item[1].score, item[0]))
        while remaining:
            current = remaining.pop(0)
            kept.append(current)
            remaining = [
                candidate
                for candidate in remaining
                if box_iou(current[1].bbox_xyxy, candidate[1].bbox_xyxy) <= iou_threshold
            ]
    return [item[1] for item in sorted(kept, key=lambda item: item[0])]


def _clip_detection(
    detection: Detection, width: int, height: int
) -> Detection | None:
    x1, y1, x2, y2 = detection.bbox_xyxy
    clipped = (
        _clip(float(x1), 0.0, float(width)),
        _clip(float(y1), 0.0, float(height)),
        _clip(float(x2), 0.0, float(width)),
        _clip(float(y2), 0.0, float(height)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return Detection(clipped, float(detection.score), int(detection.class_id))


def expand_box(
    box: Sequence[float], factor: float, width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    half_width = (x2 - x1) * factor / 2
    half_height = (y2 - y1) * factor / 2
    return (
        _clip(center_x - half_width, 0.0, float(width)),
        _clip(center_y - half_height, 0.0, float(height)),
        _clip(center_x + half_width, 0.0, float(width)),
        _clip(center_y + half_height, 0.0, float(height)),
    )


def keypoint_coverage(keypoints: Sequence[float], box: Sequence[float]) -> float:
    if len(keypoints) != 17 * 3:
        raise BuildError(f"expected 51 keypoint values, got {len(keypoints)}")
    x1, y1, x2, y2 = (float(value) for value in box)
    labeled = 0
    inside = 0
    epsilon = 1e-6
    for index in range(17):
        x, y, visibility = keypoints[index * 3 : index * 3 + 3]
        if int(round(visibility)) <= 0:
            continue
        labeled += 1
        if x1 - epsilon <= x <= x2 + epsilon and y1 - epsilon <= y <= y2 + epsilon:
            inside += 1
    return inside / labeled if labeled else 0.0


def one_to_one_assignment(
    anchors: Sequence[Sequence[float]], proposals: Sequence[Detection]
) -> tuple[list[int], np.ndarray]:
    """Maximize total anchor/proposal IoU without reusing a proposal."""

    ious = np.zeros((len(anchors), len(proposals)), dtype=np.float64)
    for row, anchor in enumerate(anchors):
        for column, proposal in enumerate(proposals):
            ious[row, column] = box_iou(anchor, proposal.bbox_xyxy)
    assignments = [-1] * len(anchors)
    if not anchors or not proposals:
        return assignments, ious
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - server RTMPose env includes scipy
        raise BuildError("scipy is required for one-to-one proposal matching") from exc
    rows, columns = linear_sum_assignment(-ious)
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        assignments[row] = column
    return assignments, ious


def _detector_cache_payload(
    descriptor: Mapping[str, Any], predictions: Mapping[str, Sequence[Detection]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "upstream_detector_outputs_before_builder_explicit_nms",
        "detector": dict(descriptor),
        "images": {
            key: [
                {
                    "bbox_xyxy": list(detection.bbox_xyxy),
                    "score": detection.score,
                    "class_id": detection.class_id,
                }
                for detection in value
            ]
            for key, value in sorted(predictions.items())
        },
    }


class JsonProposalProvider:
    """Read frozen upstream detector outputs from a reproducible JSON cache."""

    def __init__(self, path: Path, expected_detector_sha256: str) -> None:
        self.path = path
        if not path.is_file():
            raise BuildError(f"missing proposal cache: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BuildError(f"invalid proposal cache: {path}") from exc
        if payload.get("schema_version") != 1 or not isinstance(payload.get("images"), dict):
            raise BuildError(f"unsupported proposal-cache schema: {path}")
        descriptor = payload.get("detector")
        if not isinstance(descriptor, dict):
            raise BuildError("proposal cache lacks detector provenance")
        expected = _validate_sha256(expected_detector_sha256, "expected detector SHA256")
        actual = _validate_sha256(
            str(descriptor.get("sha256", "")), "cached detector SHA256"
        )
        if actual != expected:
            raise BuildError(
                f"proposal cache detector SHA256 mismatch: expected {expected}, got {actual}"
            )
        try:
            self._cat_class_id = int(descriptor["cat_class_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BuildError("proposal cache lacks cat_class_id") from exc
        self._descriptor = dict(descriptor)
        self._descriptor["proposal_cache"] = str(path.resolve())
        self._descriptor["proposal_cache_sha256"] = sha256_file(path)
        self._images = payload["images"]

    @property
    def cat_class_id(self) -> int:
        return self._cat_class_id

    @property
    def descriptor(self) -> Mapping[str, Any]:
        return self._descriptor

    def predict(self, images: Mapping[str, Path]) -> Mapping[str, list[Detection]]:
        missing = sorted(set(images) - set(self._images))
        extra = sorted(set(self._images) - set(images))
        if missing:
            raise BuildError(f"proposal cache lacks {len(missing)} images: {missing[:3]}")
        # Extra cached images are benign, but are recorded in the descriptor.
        self._descriptor["unused_cached_images"] = len(extra)
        output: dict[str, list[Detection]] = {}
        for key in images:
            rows = self._images[key]
            if not isinstance(rows, list):
                raise BuildError(f"proposal cache entry is not a list: {key}")
            parsed: list[Detection] = []
            for row in rows:
                try:
                    bbox = tuple(float(value) for value in row["bbox_xyxy"])
                    score = float(row["score"])
                    class_id = int(row["class_id"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise BuildError(f"invalid cached detection for {key}") from exc
                if len(bbox) != 4:
                    raise BuildError(f"invalid cached bbox for {key}")
                parsed.append(Detection(bbox, score, class_id))
            output[key] = parsed
        return output


class UltralyticsProposalProvider:
    """Run the production YOLO detector with a pinned artifact."""

    def __init__(
        self,
        weights: Path,
        expected_sha256: str,
        *,
        raw_confidence: float = 0.05,
        upstream_iou: float = 0.99,
        imgsz: int = 640,
        device: str = "0",
        batch_size: int = 16,
        cat_class_id: int | None = None,
    ) -> None:
        if not weights.is_file() or weights.is_symlink():
            raise BuildError(f"detector must be a regular non-symlink file: {weights}")
        expected = _validate_sha256(expected_sha256, "expected detector SHA256")
        actual = sha256_file(weights)
        if actual != expected:
            raise BuildError(
                f"detector SHA256 mismatch: expected {expected}, got {actual}; refusing to load"
            )
        if not 0 <= raw_confidence <= 1:
            raise BuildError("raw detector confidence must be in [0, 1]")
        if not 0 < upstream_iou <= 1:
            raise BuildError("upstream detector IoU must be in (0, 1]")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise BuildError("ultralytics is required for live detector proposals") from exc

        # Loading happens only after the exact SHA has been checked because .pt
        # artifacts use pickle-backed deserialization.
        self._model = YOLO(str(weights))
        names_raw = self._model.names
        if isinstance(names_raw, dict):
            names = {int(key): str(value) for key, value in names_raw.items()}
        else:
            names = {index: str(value) for index, value in enumerate(names_raw)}
        if cat_class_id is None:
            candidates = [key for key, value in names.items() if value.strip().lower() == "cat"]
            if len(candidates) != 1:
                raise BuildError(
                    f"cannot uniquely resolve detector cat class from names={names}; "
                    "pass --cat-class-id"
                )
            cat_class_id = candidates[0]
        if cat_class_id not in names:
            raise BuildError(f"cat class id {cat_class_id} is absent from detector names={names}")
        self._cat_class_id = int(cat_class_id)
        self._raw_confidence = float(raw_confidence)
        self._upstream_iou = float(upstream_iou)
        self._imgsz = int(imgsz)
        self._device = str(device)
        self._batch_size = int(batch_size)
        self._descriptor: dict[str, Any] = {
            "path": str(weights.resolve()),
            "sha256": actual,
            "framework": "ultralytics",
            "class_names": names,
            "cat_class_id": self._cat_class_id,
            "raw_confidence": self._raw_confidence,
            "upstream_iou": self._upstream_iou,
            "imgsz": self._imgsz,
            "device": self._device,
            "batch_size": self._batch_size,
            "note": (
                "Ultralytics inference is requested with IoU=0.99, then this builder "
                "always applies its own class-aware NMS at the recorded build threshold."
            ),
        }

    @property
    def cat_class_id(self) -> int:
        return self._cat_class_id

    @property
    def descriptor(self) -> Mapping[str, Any]:
        return self._descriptor

    def predict(self, images: Mapping[str, Path]) -> Mapping[str, list[Detection]]:
        keys = sorted(images)
        paths = [str(images[key]) for key in keys]
        results = self._model.predict(
            source=paths,
            conf=self._raw_confidence,
            iou=self._upstream_iou,
            imgsz=self._imgsz,
            device=self._device,
            batch=self._batch_size,
            max_det=300,
            agnostic_nms=False,
            verbose=False,
            save=False,
        )
        if len(results) != len(keys):
            raise BuildError(
                f"detector returned {len(results)} results for {len(keys)} input images"
            )
        predictions: dict[str, list[Detection]] = {}
        for key, result in zip(keys, results, strict=True):
            if result.boxes is None:
                predictions[key] = []
                continue
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy()
            predictions[key] = [
                Detection(
                    tuple(float(value) for value in box.tolist()),
                    float(score),
                    int(round(float(class_id))),
                )
                for box, score, class_id in zip(xyxy, scores, classes, strict=True)
            ]
        return predictions


def validate_video_mapping(
    instances: Sequence[CameraInstance], config: BuildConfig
) -> dict[str, dict[str, Any]]:
    if config.video_root is None:
        return {}
    by_video: dict[Path, dict[int, CameraInstance]] = defaultdict(dict)
    for instance in instances:
        source_name = re.split(r"[\\/]", instance.metadata["source_video"])[-1]
        if not source_name:
            raise BuildError(f"empty source video name for {instance.instance_key}")
        video_path = (config.video_root / instance.metadata["clip_id"] / source_name).resolve()
        video_root = config.video_root.resolve()
        if os.path.commonpath((str(video_root), str(video_path))) != str(video_root):
            raise BuildError(f"metadata video path escapes video root: {video_path}")
        try:
            frame_index = int(instance.metadata["frame_index"])
        except ValueError as exc:
            raise BuildError(f"invalid frame index for {instance.instance_key}") from exc
        existing = by_video[video_path].get(frame_index)
        if existing is not None and existing.full_stem != instance.full_stem:
            raise BuildError(
                f"two full frames claim the same source video frame: {video_path}:{frame_index}"
            )
        by_video[video_path][frame_index] = instance

    checks: dict[str, dict[str, Any]] = {}
    for video_path, frame_map in sorted(by_video.items(), key=lambda item: str(item[0])):
        if not video_path.is_file():
            raise BuildError(f"mapped source video is missing: {video_path}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise BuildError(f"cannot open mapped source video: {video_path}")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_index, instance in sorted(frame_map.items()):
                if frame_count > 0 and frame_index >= frame_count:
                    raise BuildError(
                        f"mapped frame {frame_index} exceeds {video_path} frame count {frame_count}"
                    )
                check: dict[str, Any] = {
                    "video": str(video_path),
                    "frame_index": frame_index,
                    "video_frame_count": frame_count,
                }
                if config.verify_video_pixels:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, decoded = capture.read()
                    if not ok or decoded is None:
                        raise BuildError(f"cannot read mapped frame {video_path}:{frame_index}")
                    full = cv2.imread(str(instance.full_image), cv2.IMREAD_COLOR)
                    if full is None:
                        raise BuildError(f"cannot decode full frame: {instance.full_image}")
                    if decoded.shape[:2] != full.shape[:2]:
                        decoded = cv2.resize(
                            decoded,
                            (full.shape[1], full.shape[0]),
                            interpolation=cv2.INTER_AREA,
                        )
                    mae = float(
                        np.mean(
                            np.abs(
                                decoded.astype(np.float32, copy=False)
                                - full.astype(np.float32, copy=False)
                            )
                        )
                    )
                    check["pixel_mae"] = mae
                    if mae > config.max_video_mae:
                        raise BuildError(
                            f"video/full-frame mapping MAE {mae:.4f} exceeds "
                            f"{config.max_video_mae:.4f}: {instance.full_stem}"
                        )
                checks[instance.full_stem] = check
        finally:
            capture.release()
    return checks


def match_camera_proposals(
    instances: Sequence[CameraInstance],
    predictions: Mapping[str, Sequence[Detection]],
    config: BuildConfig,
    cat_class_id: int,
) -> tuple[list[MatchedInstance], list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Match frozen detector proposals to human pose instances with hard gates."""

    by_frame: dict[str, list[CameraInstance]] = defaultdict(list)
    for instance in instances:
        by_frame[instance.full_stem].append(instance)
    missing = sorted(set(by_frame) - set(predictions))
    if missing:
        raise BuildError(f"proposal provider lacks {len(missing)} required frames: {missing[:3]}")

    matched: list[MatchedInstance] = []
    rejections: list[dict[str, Any]] = []
    detection_counts: dict[str, dict[str, int]] = {}
    for full_stem, frame_instances in sorted(by_frame.items()):
        first = frame_instances[0]
        clipped = [
            candidate
            for detection in predictions[full_stem]
            if (
                candidate := _clip_detection(
                    detection, first.full_width, first.full_height
                )
            )
            is not None
        ]
        after_nms = class_aware_nms(clipped, config.explicit_nms_iou)
        cat_proposals = [
            detection
            for detection in after_nms
            if detection.class_id == cat_class_id
            and detection.score >= config.proposal_min_score
        ]
        detection_counts[full_stem] = {
            "raw_all_classes": len(predictions[full_stem]),
            "clipped_all_classes": len(clipped),
            "post_nms_all_classes": len(after_nms),
            "eligible_cat": len(cat_proposals),
        }
        ordered_instances = sorted(frame_instances, key=lambda item: item.instance_key)
        assignments, iou_matrix = one_to_one_assignment(
            [instance.source_anchor_xyxy for instance in ordered_instances], cat_proposals
        )
        for row, instance in enumerate(ordered_instances):
            proposal_index = assignments[row]
            if proposal_index < 0:
                rejections.append(
                    {
                        "instance_key": instance.instance_key,
                        "full_stem": full_stem,
                        "clip_id": instance.metadata["clip_id"],
                        "reasons": "no_unique_cat_proposal",
                        "source_iou": "",
                        "keypoint_coverage": "",
                        "eligible_cat_proposals": len(cat_proposals),
                    }
                )
                continue
            proposal = cat_proposals[proposal_index]
            source_iou = float(iou_matrix[row, proposal_index])
            padded = expand_box(
                proposal.bbox_xyxy,
                config.proposal_padding,
                instance.full_width,
                instance.full_height,
            )
            coverage = keypoint_coverage(instance.keypoints, padded)
            reasons = []
            if source_iou + 1e-12 < config.min_source_iou:
                reasons.append("source_iou_below_threshold")
            if coverage + 1e-12 < config.min_keypoint_coverage:
                reasons.append("keypoint_coverage_below_threshold")
            if reasons:
                rejections.append(
                    {
                        "instance_key": instance.instance_key,
                        "full_stem": full_stem,
                        "clip_id": instance.metadata["clip_id"],
                        "reasons": ";".join(reasons),
                        "source_iou": source_iou,
                        "keypoint_coverage": coverage,
                        "proposal_score": proposal.score,
                        "proposal_xyxy": json.dumps(list(proposal.bbox_xyxy)),
                        "padded_proposal_xyxy": json.dumps(list(padded)),
                        "source_anchor_xyxy": json.dumps(list(instance.source_anchor_xyxy)),
                        "eligible_cat_proposals": len(cat_proposals),
                    }
                )
                continue
            matched.append(
                MatchedInstance(
                    instance=instance,
                    proposal_xyxy=proposal.bbox_xyxy,
                    padded_proposal_xyxy=padded,
                    proposal_score=proposal.score,
                    source_iou=source_iou,
                    keypoint_coverage=coverage,
                    raw_detection_count=len(predictions[full_stem]),
                    post_nms_detection_count=len(after_nms),
                )
            )
    return matched, rejections, detection_counts


def assign_clip_splits(
    clip_weights: Mapping[str, int],
    ratios: Sequence[float],
    seed: int,
) -> dict[str, str]:
    """Deterministically balance annotations while keeping each clip intact."""

    if len(clip_weights) < len(SPLITS):
        raise BuildError(
            f"need at least {len(SPLITS)} camera clips for non-empty splits; "
            f"found {len(clip_weights)}"
        )
    if any(weight <= 0 for weight in clip_weights.values()):
        raise BuildError("clip weights must be positive")
    total = float(sum(clip_weights.values()))
    targets = {split: total * float(ratio) for split, ratio in zip(SPLITS, ratios, strict=True)}
    counts = {split: 0 for split in SPLITS}
    assignments: dict[str, str] = {}

    def tie_value(clip_id: str, split: str = "") -> str:
        return hashlib.sha256(f"{seed}\0{clip_id}\0{split}".encode("utf-8")).hexdigest()

    ordered = sorted(clip_weights, key=lambda clip: (-clip_weights[clip], tie_value(clip)))
    for index, clip_id in enumerate(ordered):
        empty = [split for split in SPLITS if split not in assignments.values()]
        remaining_groups = len(ordered) - index
        candidates = empty if remaining_groups == len(empty) else list(SPLITS)

        def score(split: str) -> tuple[float, float, str]:
            new_count = counts[split] + clip_weights[clip_id]
            # Fill ratio, followed by absolute overshoot, with a seeded stable tie.
            fill = new_count / max(targets[split], 1e-12)
            overshoot = max(0.0, new_count - targets[split])
            return fill, overshoot, tie_value(clip_id, split)

        split = min(candidates, key=score)
        assignments[clip_id] = split
        counts[split] += clip_weights[clip_id]
    if set(assignments.values()) != set(SPLITS):
        raise BuildError(f"group splitter produced an empty split: {counts}")
    return assignments


def _category_definition() -> dict[str, Any]:
    return {
        "id": 1,
        "name": "cat",
        "supercategory": "animal",
        "keypoints": AP10K_KEYPOINTS,
        "skeleton": AP10K_SKELETON,
    }


def _copy_or_link(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise BuildError(f"refusing to overwrite materialized image: {target}")
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "hardlink":
        os.link(source, target)
    else:  # guarded by BuildConfig.validate
        raise BuildError(f"unsupported image materialization mode: {mode}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_file_hashes(items: Sequence[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for relative, path in sorted(items):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _camera_coco_payload(
    split: str,
    rows: Sequence[MatchedInstance],
    final_output: Path,
    build_info: Mapping[str, Any],
    working_output: Path,
    image_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_frame: dict[str, list[MatchedInstance]] = defaultdict(list)
    for row in rows:
        by_frame[row.instance.full_stem].append(row)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    detector_bboxes: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, full_stem in enumerate(sorted(by_frame), start=1):
        frame_rows = sorted(by_frame[full_stem], key=lambda item: item.instance.instance_key)
        first = frame_rows[0].instance
        target_name = f"{full_stem}{first.full_image.suffix.lower()}"
        _copy_or_link(first.full_image, working_output / "images" / split / target_name, image_mode)
        images.append(
            {
                "id": image_id,
                "file_name": target_name,
                "width": first.full_width,
                "height": first.full_height,
                "clip_id": first.metadata["clip_id"],
                "source_video": first.metadata["source_video"],
                "source_frame_index": int(first.metadata["frame_index"]),
                "frame_serial": first.frame_serial,
            }
        )
        for matched in frame_rows:
            instance = matched.instance
            proposal_bbox = xyxy_to_xywh(matched.proposal_xyxy)
            if split == "train":
                annotation_bbox = proposal_bbox
                annotation_bbox_role = "topdown_input_proposal_not_ground_truth"
            else:
                annotation_bbox = xyxy_to_xywh(instance.source_anchor_xyxy)
                annotation_bbox_role = (
                    "source_annotation_reference_bbox_mixed_prediction_provenance"
                )
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    # This is intentionally the frozen production proposal used
                    # as top-down model input, not a detector pseudo-label.
                    "bbox": annotation_bbox,
                    "area": annotation_bbox[2] * annotation_bbox[3],
                    "iscrowd": 0,
                    "keypoints": instance.keypoints,
                    "num_keypoints": instance.num_keypoints,
                    "segmentation": [],
                    "bbox_role": annotation_bbox_role,
                    "bbox_source": (
                        "production_detector_after_explicit_class_aware_nms"
                        if split == "train"
                        else "source_crop_annotation_mixed_manual_and_prediction_provenance"
                    ),
                    "proposal_bbox_xywh": proposal_bbox,
                    "proposal_role": "topdown_input_from_frozen_detector_bbox_file",
                    "proposal_score": matched.proposal_score,
                    "proposal_padding_for_gate": build_info["proposal_padding"],
                    "padded_proposal_xyxy": list(matched.padded_proposal_xyxy),
                    "source_anchor_xyxy": list(instance.source_anchor_xyxy),
                    "source_anchor_role": "crop_provenance_and_matching_only_not_ground_truth",
                    "source_anchor_iou": matched.source_iou,
                    "labeled_keypoint_coverage_in_padded_proposal": matched.keypoint_coverage,
                    "keypoint_source": "human_annotation_only",
                    "pose_label_relative": instance.pose_label_relative,
                }
            )
            detector_bboxes.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": proposal_bbox,
                    "score": matched.proposal_score,
                }
            )
            manifest_rows.append(
                {
                    "status": "accepted",
                    "split": split,
                    "annotation_id": annotation_id,
                    "instance_key": instance.instance_key,
                    "full_stem": full_stem,
                    "clip_id": instance.metadata["clip_id"],
                    "source_video": instance.metadata["source_video"],
                    "source_frame_index": instance.metadata["frame_index"],
                    "frame_serial": instance.frame_serial,
                    "source_ann_index": instance.source_ann_index,
                    "human_keypoints": instance.num_keypoints,
                    "proposal_score": matched.proposal_score,
                    "source_iou": matched.source_iou,
                    "keypoint_coverage": matched.keypoint_coverage,
                    "raw_detection_count": matched.raw_detection_count,
                    "post_nms_detection_count": matched.post_nms_detection_count,
                    "proposal_xyxy": json.dumps(list(matched.proposal_xyxy)),
                    "padded_proposal_xyxy": json.dumps(list(matched.padded_proposal_xyxy)),
                    "source_anchor_xyxy": json.dumps(list(instance.source_anchor_xyxy)),
                    "pose_label_sha256": sha256_file(instance.pose_label),
                    "full_frame_sha256": sha256_file(instance.full_image),
                }
            )
            annotation_id += 1
    payload = {
        "info": {
            "description": (
                "Local-camera cat pose: human keypoints on original frames with frozen "
                "production-detector top-down proposals"
            ),
            "version": "1.0",
            "date_created": build_info["date_created"],
            "split": split,
            "bbox_semantics": (
                "topdown_input_proposal_not_ground_truth"
                if split == "train"
                else "source_annotation_reference_bbox_mixed_prediction_provenance"
            ),
            "inference_bbox_semantics": (
                "annotation_bbox"
                if split == "train"
                else f"frozen_detector_bbox_file:{split}_detector_bboxes.json"
            ),
            "keypoint_semantics": "human_annotation_only",
            "detector_keypoints_used_as_labels": False,
            "output_root": str(final_output),
            "build": dict(build_info),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [_category_definition()],
    }
    return payload, manifest_rows, detector_bboxes


def _generic_coco_payload(
    instances: Sequence[GenericInstance], working_output: Path, image_mode: str, created: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for index, instance in enumerate(instances, start=1):
        _copy_or_link(
            instance.source_image,
            working_output / "images" / "generic_train" / instance.output_name,
            image_mode,
        )
        images.append(
            {
                "id": index,
                "file_name": instance.output_name,
                "width": instance.width,
                "height": instance.height,
            }
        )
        annotations.append(
            {
                "id": index,
                "image_id": index,
                "category_id": 1,
                "bbox": instance.bbox_xywh,
                "area": instance.bbox_xywh[2] * instance.bbox_xywh[3],
                "iscrowd": 0,
                "keypoints": instance.keypoints,
                "num_keypoints": instance.num_keypoints,
                "segmentation": [],
                "bbox_role": "legacy_clean_crop_topdown_input",
                "keypoint_source": "human_annotation_only",
                "pose_label_relative": instance.source_relative,
            }
        )
        manifest.append(
            {
                "status": "accepted_generic_train_only",
                "split": "generic_train",
                "annotation_id": index,
                "instance_key": instance.source_relative,
                "human_keypoints": instance.num_keypoints,
                "pose_label_sha256": sha256_file(instance.source_label),
                "source_image_sha256": sha256_file(instance.source_image),
            }
        )
    return (
        {
            "info": {
                "description": "All accepted clean non-v4 pose crops; training only",
                "version": "1.0",
                "date_created": created,
                "split": "generic_train_only",
                "camera_validation_or_test_eligible": False,
                "keypoint_semantics": "human_annotation_only",
                "detector_keypoints_used_as_labels": False,
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [_category_definition()],
        },
        manifest,
    )


def build_dataset(config: BuildConfig, provider: ProposalProvider) -> dict[str, Any]:
    """Build and atomically publish the audited dataset."""

    config.validate()
    manifest_path = config.pose_root / "manifest.csv"
    if not manifest_path.is_file():
        raise BuildError(f"missing clean-dataset manifest: {manifest_path}")
    manifest_rows = load_manifest(manifest_path)
    metadata = load_metadata(config.metadata_csv)
    camera_instances = load_camera_instances(config, manifest_rows, metadata)
    generic_instances = load_generic_instances(config, manifest_rows)
    video_checks = validate_video_mapping(camera_instances, config)

    descriptor = dict(provider.descriptor)
    detector_sha256 = _validate_sha256(
        str(descriptor.get("sha256", "")), "proposal-provider detector SHA256"
    )
    frame_images = {
        instance.full_stem: instance.full_image
        for instance in camera_instances
    }
    if len(frame_images) != len({instance.full_stem for instance in camera_instances}):
        raise BuildError("inconsistent full-frame image mapping")
    predictions = {
        key: list(value) for key, value in provider.predict(frame_images).items()
    }
    matched, rejections, detection_counts = match_camera_proposals(
        camera_instances, predictions, config, provider.cat_class_id
    )
    if rejections and not config.allow_review_rejections:
        raise GateFailure(rejections)
    if not matched:
        raise BuildError("no camera annotations remain after proposal quality gates")

    clip_weights = Counter(item.instance.metadata["clip_id"] for item in matched)
    clip_splits = assign_clip_splits(clip_weights, config.split_ratios, config.split_seed)
    for item in matched:
        item.split = clip_splits[item.instance.metadata["clip_id"]]
    split_clip_sets = {
        split: {clip for clip, assigned in clip_splits.items() if assigned == split}
        for split in SPLITS
    }
    for first_index, first in enumerate(SPLITS):
        for second in SPLITS[first_index + 1 :]:
            overlap = split_clip_sets[first] & split_clip_sets[second]
            if overlap:
                raise BuildError(f"clip leakage between {first}/{second}: {sorted(overlap)}")

    created = _utc_now()
    build_info = {
        "date_created": created,
        "split_unit": "clip_id",
        "split_seed": config.split_seed,
        "split_ratios": dict(zip(SPLITS, config.split_ratios, strict=True)),
        "explicit_class_aware_nms": True,
        "explicit_nms_iou": config.explicit_nms_iou,
        "proposal_min_score": config.proposal_min_score,
        "min_source_iou": config.min_source_iou,
        "proposal_padding": config.proposal_padding,
        "min_labeled_keypoint_coverage": config.min_keypoint_coverage,
        "train_annotation_bbox_role": "topdown_input_proposal_not_ground_truth",
        "val_test_annotation_bbox_role": (
            "source_annotation_reference_bbox_mixed_prediction_provenance"
        ),
        "val_test_inference_bbox_role": "separate_frozen_detector_bbox_file",
        "keypoint_source": "human_annotation_only",
        "detector_keypoints_used_as_labels": False,
        "detector_sha256": detector_sha256,
    }
    split_counts = {
        split: {
            "clips": len(split_clip_sets[split]),
            "images": len(
                {
                    item.instance.full_stem for item in matched if item.split == split
                }
            ),
            "annotations": sum(item.split == split for item in matched),
        }
        for split in SPLITS
    }
    camera_train_count = split_counts["train"]["annotations"]
    generic_count = len(generic_instances)
    source_sampling = None
    if generic_count:
        # The accompanying config uses batch_size=32 and a 1:1
        # MultiSourceSampler.  Its iterator rounds up to complete batches, so
        # report actual yielded draws rather than only the nominal length.
        combined_length = generic_count + camera_train_count
        configured_batch_size = 32
        samples_per_source_per_batch = configured_batch_size // 2
        rounded_batches = math.ceil(combined_length / configured_batch_size)
        draws_per_source = rounded_batches * samples_per_source_per_batch
        source_sampling = {
            "config_source_ratio_generic_to_camera": [1, 1],
            "configured_batch_size": configured_batch_size,
            "samples_per_source_per_batch": [
                samples_per_source_per_batch,
                samples_per_source_per_batch,
            ],
            "nominal_combined_epoch_length": combined_length,
            "rounded_batches_per_single_gpu_epoch": rounded_batches,
            "actual_yielded_samples_per_single_gpu_epoch": (
                rounded_batches * configured_batch_size
            ),
            "generic_draws_per_single_gpu_epoch": draws_per_source,
            "camera_draws_per_single_gpu_epoch": draws_per_source,
            "generic_repeats_per_single_gpu_epoch": draws_per_source / generic_count,
            "camera_repeats_per_single_gpu_epoch": draws_per_source / camera_train_count,
            "note": "Distributed world size can alter rank-local draws.",
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "partial_review_build" if rejections else "complete",
        "date_created": created,
        "output": str(config.output.resolve()),
        "inputs": {
            "pose_root": str(config.pose_root.resolve()),
            "clean_manifest": str(manifest_path.resolve()),
            "clean_manifest_sha256": sha256_file(manifest_path),
            "metadata_csv": str(config.metadata_csv.resolve()),
            "metadata_csv_sha256": sha256_file(config.metadata_csv),
            "full_frame_root": str(config.full_frame_root.resolve()),
            "video_root": str(config.video_root.resolve()) if config.video_root else None,
            "detector": descriptor,
        },
        "invariants": {
            "explicit_class_aware_nms": True,
            "one_to_one_proposal_assignment": True,
            "source_iou_gate": config.min_source_iou,
            "padded_labeled_keypoint_coverage_gate": config.min_keypoint_coverage,
            "camera_split_unit": "clip_id",
            "camera_clip_overlap": {
                "train_val": sorted(split_clip_sets["train"] & split_clip_sets["val"]),
                "train_test": sorted(split_clip_sets["train"] & split_clip_sets["test"]),
                "val_test": sorted(split_clip_sets["val"] & split_clip_sets["test"]),
            },
            "generic_non_v4_is_train_only": True,
            "detector_keypoints_used_as_labels": False,
            "bbox_is_ground_truth": False,
            "train_annotation_bbox_role": "frozen_detector_topdown_input_proposal",
            "val_test_annotation_bbox_role": (
                "source_annotation_reference_bbox_mixed_prediction_provenance"
            ),
            "val_test_inference_bbox_role": "separate_frozen_detector_bbox_file",
        },
        "contamination_boundary": {
            "known_historical_fact": (
                "All v4 labels participated in the previous clean516 fine-tune, including "
                "clips now assigned to validation/test."
            ),
            "required_initialization": "official_AP10K_base_only",
            "forbidden_for_uncontaminated_metrics": (
                "Do not initialize from, select with, or compare as an uncontaminated "
                "checkpoint the previous clean516 best weight."
            ),
            "generic_non_v4": (
                "Legacy clean non-v4 crops are allowed only in training and never in "
                "camera validation/test."
            ),
            "source_anchor_provenance": (
                "Not verified ground truth. The source-task audit for these 68 v4 "
                "instances found 19 manual, 18 prediction-changed, and 31 prediction "
                "boxes. Consequently coco/AP is an internal model-selection metric, "
                "not a standard AP-10K benchmark claim."
            ),
        },
        "counts": {
            "camera_manifest_instances": len(camera_instances),
            "camera_unique_frames": len(frame_images),
            "camera_unique_clips": len(clip_weights),
            "camera_matched_instances": len(matched),
            "camera_rejected_instances": len(rejections),
            "generic_non_v4_train_instances": generic_count,
            "splits": split_counts,
        },
        "detector_counts": {
            "raw_all_classes": sum(value["raw_all_classes"] for value in detection_counts.values()),
            "post_nms_all_classes": sum(
                value["post_nms_all_classes"] for value in detection_counts.values()
            ),
            "eligible_cat": sum(value["eligible_cat"] for value in detection_counts.values()),
            "per_frame": detection_counts,
        },
        "gate_metrics": {
            "source_iou_min": min(item.source_iou for item in matched),
            "source_iou_median": float(np.median([item.source_iou for item in matched])),
            "keypoint_coverage_min": min(item.keypoint_coverage for item in matched),
            "proposal_score_min": min(item.proposal_score for item in matched),
        },
        "video_mapping": {
            "validated": config.video_root is not None,
            "pixels_compared": config.verify_video_pixels,
            "max_allowed_mae": config.max_video_mae if config.verify_video_pixels else None,
            "max_observed_mae": max(
                (float(value.get("pixel_mae", 0.0)) for value in video_checks.values()),
                default=None,
            ),
            "frames": video_checks,
        },
        "training_source_sampling": source_sampling,
    }

    output_parent = config.output.resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{config.output.name}.tmp-", dir=str(output_parent))
    ).resolve()
    if temporary.parent != output_parent:
        raise BuildError(f"unsafe temporary output location: {temporary}")
    try:
        all_manifest_rows: list[dict[str, Any]] = []
        for split in SPLITS:
            payload, rows, detector_bboxes = _camera_coco_payload(
                split,
                [item for item in matched if item.split == split],
                config.output.resolve(),
                build_info,
                temporary,
                config.image_mode,
            )
            _write_json(temporary / "annotations" / f"{split}.json", payload)
            if split in {"val", "test"}:
                _write_json(
                    temporary / "annotations" / f"{split}_detector_bboxes.json",
                    detector_bboxes,
                )
            all_manifest_rows.extend(rows)
        if generic_instances:
            payload, rows = _generic_coco_payload(
                generic_instances, temporary, config.image_mode, created
            )
            _write_json(temporary / "annotations" / "generic_train.json", payload)
            all_manifest_rows.extend(rows)

        if rejections:
            all_manifest_rows.extend(
                {
                    "status": "quarantined_rejection",
                    "split": "",
                    **rejection,
                }
                for rejection in rejections
            )
        _write_csv(temporary / "build_manifest.csv", all_manifest_rows)
        _write_csv(temporary / "review_rejections.csv", rejections)
        _write_csv(
            temporary / "clip_splits.csv",
            [
                {
                    "clip_id": clip_id,
                    "split": clip_splits[clip_id],
                    "annotations": clip_weights[clip_id],
                }
                for clip_id in sorted(clip_splits)
            ],
        )
        used_mapping: list[dict[str, Any]] = []
        seen_frames: set[str] = set()
        for item in sorted(matched, key=lambda row: row.instance.full_stem):
            instance = item.instance
            if instance.full_stem in seen_frames:
                continue
            seen_frames.add(instance.full_stem)
            used_mapping.append(
                {
                    "full_stem": instance.full_stem,
                    "split": item.split,
                    "frame_serial": instance.frame_serial,
                    "clip_id": instance.metadata["clip_id"],
                    "source_video": instance.metadata["source_video"],
                    "source_frame_index": instance.metadata["frame_index"],
                    "video_time_sec": instance.metadata["video_time_sec"],
                    "fps": instance.metadata["fps"],
                    "metadata_row": instance.metadata["metadata_row"],
                    "full_frame_sha256": sha256_file(instance.full_image),
                }
            )
        _write_csv(temporary / "provenance" / "used_source_mapping.csv", used_mapping)
        shutil.copy2(config.metadata_csv, temporary / "provenance" / "metadata.csv")
        shutil.copy2(manifest_path, temporary / "provenance" / "clean_manifest.csv")
        _write_json(
            temporary / "provenance" / "proposals_upstream.json",
            _detector_cache_payload(descriptor, predictions),
        )
        report["inputs"]["camera_pose_labels_aggregate_sha256"] = _aggregate_file_hashes(
            [(instance.pose_label_relative, instance.pose_label) for instance in camera_instances]
        )
        report["inputs"]["camera_full_frames_aggregate_sha256"] = _aggregate_file_hashes(
            [(stem, path) for stem, path in frame_images.items()]
        )
        _write_json(temporary / "build_report.json", report)
        os.replace(temporary, config.output.resolve())
    except BaseException:
        if temporary.exists() and temporary.parent == output_parent:
            shutil.rmtree(temporary)
        raise
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--full-frame-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--verify-video-pixels", action="store_true")
    parser.add_argument("--max-video-mae", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--detector", type=Path)
    source.add_argument("--proposals-json", type=Path)
    parser.add_argument(
        "--detector-sha256",
        required=True,
        help="Exact SHA256 of the production detector that generated proposals",
    )
    parser.add_argument("--cat-class-id", type=int)
    parser.add_argument("--raw-confidence", type=float, default=0.05)
    parser.add_argument("--upstream-iou", type=float, default=0.99)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--detector-batch-size", type=int, default=16)
    parser.add_argument("--proposal-min-score", type=float, default=0.25)
    parser.add_argument("--explicit-nms-iou", type=float, default=0.70)
    parser.add_argument("--min-source-iou", type=float, default=0.50)
    parser.add_argument("--proposal-padding", type=float, default=1.25)
    parser.add_argument("--min-keypoint-coverage", type=float, default=1.0)
    parser.add_argument(
        "--split-ratios", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
        default=(0.70, 0.15, 0.15)
    )
    parser.add_argument("--split-seed", type=int, default=21)
    parser.add_argument(
        "--include-non-v4-train",
        action="store_true",
        help="Add all accepted non-v4 clean pose crops to training only",
    )
    parser.add_argument("--image-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument(
        "--allow-review-rejections",
        action="store_true",
        help=(
            "Explicitly build a partial dataset after quarantining failed instances; "
            "without this flag any failure aborts the build"
        ),
    )
    parser.add_argument(
        "--review-report",
        type=Path,
        help="On a gate-aborted build, write the rejection CSV here (never a partial dataset)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = BuildConfig(
        pose_root=args.pose_root,
        metadata_csv=args.metadata_csv,
        full_frame_root=args.full_frame_root,
        output=args.output,
        split_ratios=tuple(args.split_ratios),
        split_seed=args.split_seed,
        explicit_nms_iou=args.explicit_nms_iou,
        proposal_min_score=args.proposal_min_score,
        min_source_iou=args.min_source_iou,
        proposal_padding=args.proposal_padding,
        min_keypoint_coverage=args.min_keypoint_coverage,
        include_non_v4_train=args.include_non_v4_train,
        allow_review_rejections=args.allow_review_rejections,
        image_mode=args.image_mode,
        video_root=args.video_root,
        verify_video_pixels=args.verify_video_pixels,
        max_video_mae=args.max_video_mae,
    )
    try:
        if args.proposals_json:
            provider: ProposalProvider = JsonProposalProvider(
                args.proposals_json, args.detector_sha256
            )
        else:
            provider = UltralyticsProposalProvider(
                args.detector,
                args.detector_sha256,
                raw_confidence=args.raw_confidence,
                upstream_iou=args.upstream_iou,
                imgsz=args.imgsz,
                device=args.device,
                batch_size=args.detector_batch_size,
                cat_class_id=args.cat_class_id,
            )
        report = build_dataset(config, provider)
    except GateFailure as exc:
        if args.review_report:
            if args.review_report.exists() or args.review_report.is_symlink():
                print(
                    f"error: refusing to overwrite review report: {args.review_report}",
                    file=sys.stderr,
                )
                return 2
            _write_csv(args.review_report, exc.rejections)
            print(f"review rejections written to {args.review_report}", file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": report["output"],
                "counts": report["counts"],
                "gate_metrics": report["gate_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
