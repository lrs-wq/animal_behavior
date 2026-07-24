#!/usr/bin/env python3
"""Build a quality-gated cat-pose dataset without modifying the source dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_image(dataset: Path, label_relative: Path) -> Path:
    image_base = dataset / "images" / Path(*label_relative.parts[1:])
    for suffix in IMAGE_SUFFIXES:
        candidate = image_base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for {label_relative}")


def load_duplicate_paths(export_summary: Path | None) -> set[str]:
    if export_summary is None:
        return set()
    data = json.loads(export_summary.read_text(encoding="utf-8"))
    duplicate_task_ids = {int(item["task_id"]) for item in data.get("duplicate_labels", [])}
    return {
        Path(str(row["relative_path"]).replace("\\", "/")).with_suffix(".txt").as_posix()
        for row in data.get("rows", [])
        if int(row["task_id"]) in duplicate_task_ids
    }


def image_relative_from_label(label_relative: Path, image_path: Path) -> Path:
    return Path("images", *label_relative.parts[1:]).with_suffix(image_path.suffix.lower())


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def write_data_yaml(source_yaml: Path, output_yaml: Path, output_root: Path) -> None:
    lines = source_yaml.read_text(encoding="utf-8").splitlines()
    replaced = False
    result = []
    for line in lines:
        if line.lstrip().startswith("path:") and not replaced:
            result.append(f"path: {output_root.as_posix()}")
            replaced = True
        else:
            result.append(line)
    if not replaced:
        result.insert(0, f"path: {output_root.as_posix()}")
    output_yaml.write_text("\n".join(result) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("audit_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--export-summary", type=Path)
    parser.add_argument("--min-keypoints", type=int, default=5)
    parser.add_argument("--min-regions", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    output.mkdir(parents=True)

    duplicate_paths = load_duplicate_paths(args.export_summary)
    with args.audit_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    accepted: list[dict[str, str]] = []
    review: list[dict[str, str]] = []
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    link_modes: Counter[str] = Counter()
    duplicate_matches = 0

    for row in rows:
        reasons = []
        if row.get("structural_ok", "").lower() != "true":
            reasons.append("structural_error")
        if int(row.get("present_count") or 0) < args.min_keypoints:
            reasons.append(f"keypoints_lt_{args.min_keypoints}")
        if int(row.get("covered_regions") or 0) < args.min_regions:
            reasons.append(f"regions_lt_{args.min_regions}")

        label_relative = Path(row["path"])
        source_relative = Path(*label_relative.parts[1:]).as_posix()
        if source_relative in duplicate_paths:
            reasons.append("duplicate_keypoint_in_source_annotation")
            duplicate_matches += 1

        row_with_decision = dict(row)
        row_with_decision["decision"] = "review" if reasons else "accepted"
        row_with_decision["decision_reasons"] = ";".join(reasons)
        if reasons:
            review.append(row_with_decision)
            continue

        source_label = dataset / label_relative
        source_image = find_image(dataset, label_relative)
        target_label = output / label_relative
        target_image = output / image_relative_from_label(label_relative, source_image)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_label, target_label)
        link_modes[link_or_copy(source_image, target_image)] += 1
        accepted.append(row_with_decision)
        split_counts[row["split"]] += 1
        source_counts[row["source"]] += 1

    write_data_yaml(dataset / "data.yaml", output / "data.yaml", output)
    columns = list(rows[0].keys()) + ["decision", "decision_reasons"]
    for filename, selected in (("manifest.csv", accepted), ("review_queue.csv", review)):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(selected)

    summary = {
        "source_dataset": str(dataset),
        "output_dataset": str(output),
        "policy": {
            "min_keypoints": args.min_keypoints,
            "min_regions": args.min_regions,
            "duplicate_source_annotations_require_review": args.export_summary is not None,
        },
        "source_annotations": len(rows),
        "accepted_annotations": len(accepted),
        "review_annotations": len(review),
        "duplicate_source_paths_loaded": len(duplicate_paths),
        "duplicate_source_annotations_matched": duplicate_matches,
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "image_storage": dict(link_modes),
    }
    (output / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
