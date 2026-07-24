#!/usr/bin/env python3
"""Validate a cat-pose dataset with both project and Ultralytics checks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from ultralytics.data.utils import check_det_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--training-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset.resolve()
    data = check_det_dataset(str(root / "data.yaml"))
    report: dict[str, object] = {
        "ultralytics": {
            key: data.get(key) for key in ("nc", "names", "kpt_shape", "flip_idx")
        }
    }

    review_path = root / "review_queue.csv"
    if review_path.exists():
        with review_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        report["review_queue"] = {
            "annotations": len(rows),
            "reasons": dict(
                Counter(
                    reason
                    for row in rows
                    for reason in row.get("decision_reasons", "").split(";")
                    if reason
                )
            ),
        }

    if args.training_dir:
        sys.path.insert(0, str(args.training_dir.resolve()))
        from train_pose import validate_pose_dataset

        report["training_script"] = validate_pose_dataset(root / "data.yaml")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
