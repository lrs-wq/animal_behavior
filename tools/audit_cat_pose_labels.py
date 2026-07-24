#!/usr/bin/env python3
"""Audit YOLO pose labels for the 17-keypoint cat skeleton.

The script is intentionally read-only.  It reports structural errors separately
from semantic review candidates so that an audit never silently changes labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "neck",
    "tail_base",
    "left_front_shoulder",
    "left_front_elbow",
    "left_front_paw",
    "right_front_shoulder",
    "right_front_elbow",
    "right_front_paw",
    "left_hind_hip",
    "left_hind_knee",
    "left_hind_paw",
    "right_hind_hip",
    "right_hind_knee",
    "right_hind_paw",
)

REGIONS = {
    "head": (0, 1, 2),
    "axis": (3, 4),
    "front_legs": (5, 6, 7, 8, 9, 10),
    "hind_legs": (11, 12, 13, 14, 15, 16),
}

EXPECTED_COLUMNS = 5 + len(KEYPOINTS) * 3
SPLITS = ("train", "val", "test")


def source_name(path: Path) -> str:
    parts = path.stem.split("__")
    if parts and parts[0] in SPLITS and len(parts) > 1:
        return parts[1]
    return parts[0] if parts else "unknown"


def bucket_count(value: int) -> str:
    if value <= 2:
        return "00-02"
    if value <= 4:
        return "03-04"
    if value <= 8:
        return "05-08"
    if value <= 12:
        return "09-12"
    return "13-17"


def is_finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def audit_line(tokens: list[str]) -> dict[str, object]:
    flags: list[str] = []
    if len(tokens) != EXPECTED_COLUMNS:
        return {
            "structural_ok": False,
            "flags": [f"column_count_{len(tokens)}"],
            "class_id": None,
            "bbox": None,
            "keypoints": None,
        }

    try:
        values = [float(token) for token in tokens]
    except ValueError:
        return {
            "structural_ok": False,
            "flags": ["non_numeric"],
            "class_id": None,
            "bbox": None,
            "keypoints": None,
        }

    if not is_finite(values):
        flags.append("non_finite")

    class_id = values[0]
    bbox = values[1:5]
    if class_id != 0:
        flags.append("class_not_zero")
    if not all(0.0 <= value <= 1.0 for value in bbox):
        flags.append("bbox_out_of_range")
    if bbox[2] <= 0.0 or bbox[3] <= 0.0:
        flags.append("bbox_non_positive")

    keypoints: list[tuple[float, float, int]] = []
    for index in range(len(KEYPOINTS)):
        x, y, raw_v = values[5 + index * 3 : 8 + index * 3]
        rounded_v = round(raw_v)
        if abs(raw_v - rounded_v) > 1e-6 or rounded_v not in (0, 1, 2):
            flags.append(f"kpt_{index}_invalid_visibility")
            v = int(max(0, min(2, rounded_v)))
        else:
            v = int(rounded_v)
        if v == 0 and (abs(x) > 1e-9 or abs(y) > 1e-9):
            flags.append(f"kpt_{index}_absent_nonzero_xy")
        if v > 0:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                flags.append(f"kpt_{index}_out_of_range")
            if abs(x) <= 1e-9 and abs(y) <= 1e-9:
                flags.append(f"kpt_{index}_present_zero_xy")
        keypoints.append((x, y, v))

    present = [index for index, (_, _, v) in enumerate(keypoints) if v > 0]
    clear = [index for index, (_, _, v) in enumerate(keypoints) if v == 2]
    occluded = [index for index, (_, _, v) in enumerate(keypoints) if v == 1]
    region_counts = {
        region: sum(keypoints[index][2] > 0 for index in indices)
        for region, indices in REGIONS.items()
    }
    covered_regions = sum(count > 0 for count in region_counts.values())

    if len(present) <= 2:
        flags.append("semantic_sparse_0_2")
    elif len(present) <= 4:
        flags.append("semantic_sparse_3_4")
    if region_counts["head"] == 0:
        flags.append("semantic_no_head")
    if region_counts["axis"] == 0:
        flags.append("semantic_no_axis")
    if region_counts["front_legs"] + region_counts["hind_legs"] == 0:
        flags.append("semantic_no_limbs")
    if covered_regions <= 1:
        flags.append("semantic_single_region")

    seen: dict[tuple[float, float], int] = {}
    for index in present:
        x, y, _ = keypoints[index]
        coordinate = (round(x, 6), round(y, 6))
        if coordinate in seen:
            flags.append(f"semantic_duplicate_kpt_{seen[coordinate]}_{index}")
        else:
            seen[coordinate] = index

    x_center, y_center, width, height = bbox
    left, right = x_center - width / 2, x_center + width / 2
    top, bottom = y_center - height / 2, y_center + height / 2
    if present and any(
        not (left - 1e-6 <= keypoints[index][0] <= right + 1e-6)
        or not (top - 1e-6 <= keypoints[index][1] <= bottom + 1e-6)
        for index in present
    ):
        flags.append("semantic_keypoint_outside_bbox")
    if all(abs(a - b) <= 1e-6 for a, b in zip(bbox, (0.5, 0.5, 1.0, 1.0))):
        flags.append("info_full_frame_bbox")

    structural_prefixes = (
        "column_count_",
        "non_numeric",
        "non_finite",
        "class_not_zero",
        "bbox_",
        "kpt_",
    )
    structural_ok = not any(flag.startswith(structural_prefixes) for flag in flags)
    return {
        "structural_ok": structural_ok,
        "flags": sorted(set(flags)),
        "class_id": class_id,
        "bbox": bbox,
        "keypoints": keypoints,
        "present_count": len(present),
        "clear_count": len(clear),
        "occluded_count": len(occluded),
        "covered_regions": covered_regions,
        **{f"region_{name}": count for name, count in region_counts.items()},
    }


def audit_dataset(dataset: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    flag_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    visibility_counts: Counter[str] = Counter()
    source_visibility: dict[str, Counter[str]] = defaultdict(Counter)
    line_count = 0

    for split in SPLITS:
        label_dir = dataset / "labels" / split
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.rglob("*.txt")):
            relative = path.relative_to(dataset).as_posix()
            source = source_name(path)
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                record = {
                    "path": relative,
                    "split": split,
                    "source": source,
                    "line": 0,
                    "structural_ok": False,
                    "flags": ["empty_label"],
                }
                records.append(record)
                flag_counts["empty_label"] += 1
                continue
            for line_number, line in enumerate(lines, start=1):
                result = audit_line(line.split())
                record = {
                    "path": relative,
                    "split": split,
                    "source": source,
                    "line": line_number,
                    **result,
                }
                records.append(record)
                line_count += 1
                split_counts[split] += 1
                source_counts[source] += 1
                for flag in result["flags"]:
                    flag_counts[str(flag)] += 1
                present_count = result.get("present_count")
                if isinstance(present_count, int):
                    bucket = bucket_count(present_count)
                    visibility_counts[bucket] += 1
                    source_visibility[source][bucket] += 1

    summary: dict[str, object] = {
        "dataset": str(dataset.resolve()),
        "label_files": len({record["path"] for record in records}),
        "annotation_lines": line_count,
        "structural_errors": sum(not bool(record["structural_ok"]) for record in records),
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "present_keypoint_buckets": dict(visibility_counts),
        "flag_counts": dict(flag_counts.most_common()),
        "source_present_keypoint_buckets": {
            source: dict(counts) for source, counts in sorted(source_visibility.items())
        },
    }
    return records, summary


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    columns = (
        "path",
        "split",
        "source",
        "line",
        "structural_ok",
        "present_count",
        "clear_count",
        "occluded_count",
        "covered_regions",
        "region_head",
        "region_axis",
        "region_front_legs",
        "region_hind_legs",
        "flags",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["flags"] = ";".join(str(flag) for flag in record["flags"])
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Dataset root containing labels/{split}")
    parser.add_argument("--output-dir", type=Path, help="Directory for JSON and CSV reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    output_dir = (args.output_dir or dataset / "audit").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, summary = audit_dataset(dataset)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "annotations.csv", records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Reports: {output_dir}")
    return 1 if summary["structural_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
