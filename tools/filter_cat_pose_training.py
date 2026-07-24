#!/usr/bin/env python3
"""Build a clean cat-pose training dataset without modifying its source.

The filter removes samples that provide incomplete or contradictory supervision,
while retaining legitimate hard poses such as occlusion and rear views.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("audit_csv", type=Path)
    parser.add_argument("generation_summary", type=Path)
    parser.add_argument("label_studio_db", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-keypoints", type=int, default=5)
    parser.add_argument("--min-regions", type=int, default=3)
    parser.add_argument("--min-total", type=int, default=400)
    parser.add_argument("--min-train", type=int, default=400)
    return parser.parse_args()


def load_generation_info(path: Path) -> tuple[dict[str, object], list[int], set[tuple[int, int]], list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = payload.get("sources") or {}
    project_ids = sorted(int(project_id) for project_id in sources)
    duplicate_tasks = {
        (int(project_id), int(item["task_id"]))
        for project_id, source in sources.items()
        for item in source.get("duplicate_labels", [])
    }
    priority = [int(value) for value in (payload.get("merge_policy") or {}).get("source_priority", [])]
    for project_id in project_ids:
        if project_id not in priority:
            priority.append(project_id)
    return payload, project_ids, duplicate_tasks, priority


def load_task_metadata(
    database: Path, project_ids: list[int], priority: list[int]
) -> tuple[dict[str, dict[str, object]], list[str]]:
    placeholders = ",".join("?" for _ in project_ids)
    query = f"""
    with ranked as (
        select
            c.task_id,
            c.id as annotation_id,
            row_number() over (
                partition by c.task_id
                order by c.updated_at desc, c.id desc
            ) as rn
        from task_completion c
        where c.project_id in ({placeholders}) and c.was_cancelled = 0
    )
    select
        t.id as task_id,
        t.project_id,
        t.data,
        r.annotation_id
    from ranked r
    join task t on t.id = r.task_id
    where r.rn = 1
    """
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(query, project_ids).fetchall()
    connection.close()

    rank = {project_id: index for index, project_id in enumerate(priority)}
    candidates: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        data = json.loads(row["data"] or "{}")
        filename = str(data.get("filename") or "")
        if not filename:
            continue
        item: dict[str, object] = {
            "task_id": int(row["task_id"]),
            "project_id": int(row["project_id"]),
            "annotation_id": int(row["annotation_id"]),
            **data,
        }
        candidates.setdefault(Path(filename).stem, []).append(item)

    collisions = sorted(stem for stem, items in candidates.items() if len(items) > 1)
    selected: dict[str, dict[str, object]] = {}
    for stem, items in candidates.items():
        items.sort(key=lambda item: rank.get(int(item["project_id"]), len(rank)))
        selected[stem] = items[0]
    return selected, collisions


def find_image(dataset: Path, label_relative: Path) -> Path:
    base = dataset / "images" / Path(*label_relative.parts[1:])
    for suffix in IMAGE_SUFFIXES:
        candidate = base.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image for {label_relative.as_posix()}")


def image_relative(label_relative: Path, image: Path) -> Path:
    return Path("images", *label_relative.parts[1:]).with_suffix(image.suffix.lower())


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def write_data_yaml(source: Path, output: Path, output_root: Path) -> None:
    result: list[str] = []
    replaced = False
    for line in source.read_text(encoding="utf-8").splitlines():
        if not replaced and line.lstrip().startswith("path:"):
            result.append(f"path: {output_root.as_posix()}")
            replaced = True
        else:
            result.append(line)
    if not replaced:
        result.insert(0, f"path: {output_root.as_posix()}")
    output.write_text("\n".join(result) + "\n", encoding="utf-8")


def numeric_int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")

    generation, project_ids, duplicate_tasks, priority = load_generation_info(args.generation_summary)
    metadata, metadata_collisions = load_task_metadata(args.label_studio_db, project_ids, priority)
    with args.audit_csv.open(encoding="utf-8-sig", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    if not audit_rows:
        raise RuntimeError("Audit CSV is empty")

    decisions: list[dict[str, object]] = []
    accepted_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    accepted_split_counts: Counter[str] = Counter()
    accepted_source_counts: Counter[str] = Counter()
    accepted_project_counts: Counter[str] = Counter()
    rejected_reason_counts: Counter[str] = Counter()
    storage_modes: Counter[str] = Counter()

    output.mkdir(parents=True)
    for audit in audit_rows:
        label_relative = Path(audit["path"])
        meta = metadata.get(label_relative.stem)
        reasons: list[str] = []
        if audit.get("structural_ok", "").lower() != "true":
            reasons.append("structural_error")
        if int(audit.get("present_count") or 0) < args.min_keypoints:
            reasons.append(f"keypoints_lt_{args.min_keypoints}")
        if int(audit.get("covered_regions") or 0) < args.min_regions:
            reasons.append(f"regions_lt_{args.min_regions}")

        flags = {flag for flag in (audit.get("flags") or "").split(";") if flag}
        if any(flag.startswith("semantic_duplicate_kpt_") for flag in flags):
            reasons.append("duplicate_keypoint_coordinates")
        if "semantic_keypoint_outside_bbox" in flags:
            reasons.append("keypoint_outside_bbox")

        if meta is None:
            reasons.append("missing_label_studio_task_metadata")
            project_id = task_id = annotation_id = None
        else:
            project_id = int(meta["project_id"])
            task_id = int(meta["task_id"])
            annotation_id = int(meta["annotation_id"])
            if (project_id, task_id) in duplicate_tasks:
                reasons.append("duplicate_keypoint_label_in_annotation")
            status = str(meta.get("status") or "").strip()
            if status and status != "high_quality":
                reasons.append("source_not_high_quality")
            cat_count = numeric_int(meta.get("cat_count"))
            if cat_count is not None and cat_count != 1:
                reasons.append("source_not_single_cat")

        reasons = sorted(set(reasons))
        decision: dict[str, object] = {
            **audit,
            "project_id": project_id,
            "task_id": task_id,
            "annotation_id": annotation_id,
            "quality_score": meta.get("quality_score") if meta else None,
            "source_status": meta.get("status") if meta else None,
            "decision": "rejected" if reasons else "accepted",
            "decision_reasons": ";".join(reasons),
        }
        decisions.append(decision)
        if reasons:
            rejected_rows.append(decision)
            rejected_reason_counts.update(reasons)
            continue

        source_label = dataset / label_relative
        source_image = find_image(dataset, label_relative)
        target_label = output / label_relative
        target_image = output / image_relative(label_relative, source_image)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_label, target_label)
        storage_modes[link_or_copy(source_image, target_image)] += 1
        accepted_rows.append(decision)
        accepted_split_counts[str(audit["split"])] += 1
        accepted_source_counts[str(audit["source"])] += 1
        accepted_project_counts[str(project_id)] += 1

    if len(accepted_rows) < args.min_total:
        raise RuntimeError(f"Only {len(accepted_rows)} accepted samples; minimum is {args.min_total}")
    if accepted_split_counts["train"] < args.min_train:
        raise RuntimeError(
            f"Only {accepted_split_counts['train']} accepted training samples; minimum is {args.min_train}"
        )

    write_data_yaml(dataset / "data.yaml", output / "data.yaml", output)
    columns = list(decisions[0].keys())
    for filename, rows in (
        ("manifest.csv", accepted_rows),
        ("rejected.csv", rejected_rows),
        ("all_decisions.csv", decisions),
    ):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "source_dataset": str(dataset),
        "source_generation_summary": str(args.generation_summary.resolve()),
        "source_generated_at": generation.get("generated_at"),
        "output_dataset": str(output),
        "policy": {
            "min_keypoints": args.min_keypoints,
            "min_regions": args.min_regions,
            "reject_duplicate_keypoint_labels": True,
            "reject_duplicate_keypoint_coordinates": True,
            "reject_non_high_quality_source": True,
            "reject_non_single_cat_source": True,
            "retain_legitimate_occlusion_and_rear_views": True,
        },
        "source_samples": len(audit_rows),
        "accepted_samples": len(accepted_rows),
        "rejected_samples": len(rejected_rows),
        "accepted_split_counts": dict(sorted(accepted_split_counts.items())),
        "accepted_source_counts": dict(sorted(accepted_source_counts.items())),
        "accepted_project_counts": dict(sorted(accepted_project_counts.items())),
        "rejected_reason_counts": dict(rejected_reason_counts.most_common()),
        "duplicate_annotation_tasks_from_export": len(duplicate_tasks),
        "task_metadata_rows": len(metadata),
        "task_metadata_stem_collisions": metadata_collisions,
        "image_storage": dict(storage_modes),
    }
    (output / "filter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
