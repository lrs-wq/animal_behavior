from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def disable_wandb() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    try:
        from ultralytics.utils import SETTINGS

        SETTINGS.update({"wandb": False})
    except Exception:
        pass


def iter_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_class_images(source_dir: Path) -> dict[str, list[Path]]:
    class_images: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        images = iter_images(class_dir)
        if images:
            class_images[class_dir.name] = images
    if not class_images:
        raise FileNotFoundError(f"No class image folders found in {source_dir}")
    return class_images


def prepare_split(
    source_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    overwrite: bool,
) -> dict:
    class_images = collect_class_images(source_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)

    rng = random.Random(seed)
    summary = {
        "source": str(source_dir.resolve()),
        "output": str(output_dir.resolve()),
        "val_ratio": val_ratio,
        "seed": seed,
        "classes": sorted(class_images),
        "source_images": {},
        "train_images": {},
        "val_images": {},
    }

    for class_name, images in class_images.items():
        shuffled = list(images)
        rng.shuffle(shuffled)
        if len(shuffled) <= 1:
            val_count = 0
        else:
            val_count = max(1, round(len(shuffled) * val_ratio))
            val_count = min(val_count, len(shuffled) - 1)

        val_images = shuffled[:val_count]
        train_images = shuffled[val_count:]

        for split, split_images in (("train", train_images), ("val", val_images)):
            split_dir = output_dir / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for image in split_images:
                shutil.copy2(image, split_dir / image.name)

        summary["source_images"][class_name] = len(images)
        summary["train_images"][class_name] = len(train_images)
        summary["val_images"][class_name] = len(val_images)

    summary["source_total"] = sum(summary["source_images"].values())
    summary["train_total"] = sum(summary["train_images"].values())
    summary["val_total"] = sum(summary["val_images"].values())
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def summarize_existing_split(dataset_dir: Path) -> dict:
    classes: set[str] = set()
    split_images: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        split_images[split] = {}
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            images = iter_images(class_dir)
            if images:
                classes.add(class_dir.name)
                split_images[split][class_dir.name] = len(images)

    if "train" not in split_images or "val" not in split_images:
        raise FileNotFoundError(f"Expected train/ and val/ class folders in {dataset_dir}")
    if not classes:
        raise FileNotFoundError(f"No class image folders found in {dataset_dir}")

    summary = {
        "source": str(dataset_dir.resolve()),
        "output": str(dataset_dir.resolve()),
        "classes": sorted(classes),
        "source_images": {
            class_name: sum(split_counts.get(class_name, 0) for split_counts in split_images.values())
            for class_name in sorted(classes)
        },
        "train_images": {
            class_name: split_images.get("train", {}).get(class_name, 0)
            for class_name in sorted(classes)
        },
        "val_images": {
            class_name: split_images.get("val", {}).get(class_name, 0)
            for class_name in sorted(classes)
        },
        "split_images": split_images,
    }
    summary["source_total"] = sum(summary["source_images"].values())
    summary["train_total"] = sum(summary["train_images"].values())
    summary["val_total"] = sum(summary["val_images"].values())
    return summary


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    repo_dir = project_dir.parent

    parser = argparse.ArgumentParser(description="Train a YOLO26 image classification model without W&B.")
    parser.add_argument("--source", default=str(repo_dir / "data" / "cls_v1"))
    parser.add_argument("--dataset", default=str(repo_dir / "data" / "cls_v1_yolo26_cls"))
    parser.add_argument("--model", default="yolo26n-cls.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--project", default=str(project_dir / "runs_cls"))
    parser.add_argument("--name", default="cls_v1_yolo26n_100e_128_b64")
    parser.add_argument("--use-existing-split", action="store_true")
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    disable_wandb()
    args = parse_args()

    source_dir = Path(args.source)
    dataset_dir = Path(args.dataset)

    if args.use_existing_split:
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        summary = summarize_existing_split(dataset_dir)
    else:
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        summary = prepare_split(
            source_dir=source_dir,
            output_dir=dataset_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            overwrite=args.overwrite_split,
        )

    print("W&B disabled:", os.environ.get("WANDB_DISABLED"), os.environ.get("WANDB_MODE"))
    print("Source:", source_dir.resolve())
    print("Dataset:", dataset_dir.resolve())
    print("Classes:", ", ".join(summary["classes"]))
    print("Counts:", Counter({"train": summary["train_total"], "val": summary["val_total"]}))
    print("Model:", args.model)
    print("Epochs:", args.epochs)
    print("Batch:", args.batch)
    print("Image size:", args.imgsz)
    print("Project:", args.project)
    print("Run name:", args.name)

    if args.dry_run:
        return

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(dataset_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        resume=args.resume,
        plots=True,
        save=True,
        val=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
