#!/usr/bin/env python3
"""Publish extracted JPEG frames in one flat directory with a CSV mapping."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path


CSV_NAME = "frame_mapping.csv"
MANIFEST_NAME = "_flat_manifest.json"


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def discover(source: Path) -> list[Path]:
    return sorted(
        (path for path in source.rglob("*.jpg") if path.is_file()),
        key=lambda path: path.relative_to(source).as_posix(),
    )


def publish(source: Path, target: Path, prefix: str, mode: str) -> int:
    images = discover(source)
    if not images:
        raise RuntimeError(f"no JPEG images found under {source}")
    if target.exists():
        raise RuntimeError(f"target already exists: {target}")
    staging = target.parent / f".{target.name}.staging-{os.getpid()}"
    if staging.exists():
        raise RuntimeError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    total_bytes = 0
    try:
        with (staging / CSV_NAME).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["flat_filename", "source_relative_path", "source_size_bytes"]
            )
            for index, source_image in enumerate(images, start=1):
                flat_name = f"{prefix}_{index:08d}.jpg"
                destination = staging / flat_name
                if mode == "hardlink":
                    os.link(source_image, destination)
                else:
                    shutil.copy2(source_image, destination)
                size = source_image.stat().st_size
                total_bytes += size
                writer.writerow(
                    [flat_name, source_image.relative_to(source).as_posix(), size]
                )
        manifest = {
            "schema_version": 1,
            "created_at": timestamp(),
            "source_root": str(source),
            "target_root": str(target),
            "link_mode": mode,
            "filename_prefix": prefix,
            "image_count": len(images),
            "logical_image_bytes": total_bytes,
            "mapping_csv": CSV_NAME,
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        actual = list(staging.glob("*.jpg"))
        if len(actual) != len(images):
            raise RuntimeError(
                f"staging count mismatch: expected {len(images)}, found {len(actual)}"
            )
        os.replace(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(
        f"PUBLISHED images={len(images)} logical_bytes={total_bytes} "
        f"mode={mode} target={target}",
        flush=True,
    )
    return 0


def verify(source: Path, target: Path, prefix: str) -> int:
    mapping_path = target / CSV_NAME
    manifest_path = target / MANIFEST_NAME
    if not mapping_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("mapping CSV or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    flat_images = sorted(target.glob("*.jpg"))
    if len(rows) != len(flat_images) or len(rows) != manifest.get("image_count"):
        raise RuntimeError(
            f"count mismatch: mapping={len(rows)}, images={len(flat_images)}, "
            f"manifest={manifest.get('image_count')}"
        )
    logical_bytes = 0
    hardlinks_verified = 0
    for index, row in enumerate(rows, start=1):
        expected_name = f"{prefix}_{index:08d}.jpg"
        if row["flat_filename"] != expected_name:
            raise RuntimeError(f"unexpected flat name at row {index}: {row['flat_filename']}")
        source_image = source / row["source_relative_path"]
        flat_image = target / expected_name
        source_stat = source_image.stat()
        flat_stat = flat_image.stat()
        expected_size = int(row["source_size_bytes"])
        if source_stat.st_size != expected_size or flat_stat.st_size != expected_size:
            raise RuntimeError(f"size mismatch: {expected_name}")
        if source_stat.st_dev == flat_stat.st_dev and source_stat.st_ino == flat_stat.st_ino:
            hardlinks_verified += 1
        with flat_image.open("rb") as handle:
            if handle.read(2) != b"\xff\xd8":
                raise RuntimeError(f"invalid JPEG header: {expected_name}")
            handle.seek(-2, os.SEEK_END)
            if handle.read(2) != b"\xff\xd9":
                raise RuntimeError(f"invalid JPEG trailer: {expected_name}")
        logical_bytes += flat_stat.st_size
    if logical_bytes != manifest.get("logical_image_bytes"):
        raise RuntimeError("logical byte total does not match manifest")
    print(
        f"VERIFIED images={len(rows)} logical_bytes={logical_bytes} "
        f"hardlinks={hardlinks_verified} direct_subdirectories="
        f"{sum(path.is_dir() for path in target.iterdir())}",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--prefix", default="dog")
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.target = args.target.resolve()
    if not args.prefix or any(char in args.prefix for char in "/\\"):
        parser.error("--prefix must be a non-empty filename-safe prefix")
    return args


def main() -> int:
    args = parse_args()
    if not args.source.is_dir():
        print(f"ERROR source directory does not exist: {args.source}", file=sys.stderr)
        return 2
    if args.target == args.source or args.source in args.target.parents:
        print("ERROR target must not be the source or a child of it", file=sys.stderr)
        return 2
    try:
        if args.verify_only:
            return verify(args.source, args.target, args.prefix)
        return publish(args.source, args.target, args.prefix, args.mode)
    except Exception as exc:  # noqa: BLE001 - CLI reports a concise failure
        print(f"ERROR {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
