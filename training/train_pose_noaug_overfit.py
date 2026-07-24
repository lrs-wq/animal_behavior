from __future__ import annotations

import argparse
import os
from pathlib import Path


def disable_wandb() -> None:
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        from ultralytics import settings

        settings.update({"wandb": False})
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run no-augmentation YOLO pose train-overfit diagnostic.")
    parser.add_argument("--data", default=str(project_dir / "data.cat_pose_train_as_val.yaml"))
    parser.add_argument("--model", default=str(project_dir / "yolo11n-pose.pt"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--project", default=str(project_dir / "runs_pose_diag"))
    parser.add_argument("--name", default="cat_pose_trainmap_noaug_k4fix_300e")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    disable_wandb()
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model, task="pose")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        device=args.device,
        workers=args.workers,
        patience=args.epochs,
        seed=args.seed,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        plots=True,
        save=True,
        val=True,
        verbose=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        erasing=0.0,
        auto_augment=None,
    )


if __name__ == "__main__":
    main()
