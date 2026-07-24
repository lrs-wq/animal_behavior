#!/usr/bin/env python3
"""Render contact sheets for semantic review of audited cat-pose labels."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 3),
    (3, 4), (3, 5), (5, 6), (6, 7),
    (3, 8), (8, 9), (9, 10),
    (4, 11), (11, 12), (12, 13),
    (4, 14), (14, 15), (15, 16),
)


def find_image(dataset: Path, label_relative: str) -> Path | None:
    relative = Path(label_relative)
    image_base = dataset / "images" / Path(*relative.parts[1:])
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = image_base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def read_keypoints(label_path: Path) -> list[tuple[float, float, int]]:
    values = [float(token) for token in label_path.read_text(encoding="utf-8").splitlines()[0].split()]
    return [
        (values[5 + index * 3], values[6 + index * 3], int(round(values[7 + index * 3])))
        for index in range(17)
    ]


def render_tile(dataset: Path, row: dict[str, str], size: int) -> Image.Image | None:
    label_path = dataset / row["path"]
    image_path = find_image(dataset, row["path"])
    if image_path is None:
        return None
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image.thumbnail((size, size - 58), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size), "#171717")
    x_offset = (size - image.width) // 2
    y_offset = 25 + (size - 58 - image.height) // 2
    tile.paste(image, (x_offset, y_offset))
    draw = ImageDraw.Draw(tile)
    keypoints = read_keypoints(label_path)
    scaled = [
        (x_offset + x * image.width, y_offset + y * image.height, v)
        for x, y, v in keypoints
    ]
    for start, end in EDGES:
        x1, y1, v1 = scaled[start]
        x2, y2, v2 = scaled[end]
        if v1 > 0 and v2 > 0:
            draw.line((x1, y1, x2, y2), fill="#30d5c8", width=max(2, size // 160))
    radius = max(3, size // 80)
    for index, (x, y, visibility) in enumerate(scaled):
        if visibility == 0:
            continue
        color = "#39ff14" if visibility == 2 else "#ff9f1c"
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="black")
        draw.text((x + radius + 1, y - radius - 1), str(index), fill="white", stroke_width=2, stroke_fill="black")
    title = f"{row['source']} | kpt={row.get('present_count', '?')} | regions={row.get('covered_regions', '?')}"
    draw.text((5, 5), title, fill="white")
    short_name = label_path.stem[:38]
    draw.text((5, size - 28), short_name, fill="#dddddd")
    return tile


def category(row: dict[str, str]) -> list[str]:
    flags = set(row["flags"].split(";"))
    present = int(row.get("present_count") or 0)
    regions = int(row.get("covered_regions") or 0)
    result = []
    if present <= 2:
        result.append("sparse_00_02")
    elif present <= 4:
        result.append("sparse_03_04")
    if "semantic_no_head" in flags:
        result.append("no_head")
    if "semantic_no_axis" in flags:
        result.append("no_axis")
    if present >= 9 and regions >= 3:
        result.append("well_covered")
    return result


def make_sheet(tiles: list[Image.Image], columns: int, tile_size: int) -> Image.Image:
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "#0b0b0b")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile_size, (index // columns) * tile_size))
    return sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("audit_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=360)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.audit_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        for name in category(row):
            grouped.setdefault(name, []).append(row)
    rng = random.Random(args.seed)
    for name, candidates in sorted(grouped.items()):
        candidates = list(candidates)
        rng.shuffle(candidates)
        tiles = []
        for row in candidates:
            tile = render_tile(args.dataset, row, args.tile_size)
            if tile is not None:
                tiles.append(tile)
            if len(tiles) >= args.samples:
                break
        if tiles:
            output = args.output_dir / f"{name}.jpg"
            make_sheet(tiles, args.columns, args.tile_size).save(output, quality=92)
            print(f"{name}: {len(candidates)} candidates, rendered {len(tiles)} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
