from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SWEEP_PARAMETER_KEYS = {
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "fliplr",
    "mosaic",
    "mixup",
    "copy_paste",
    "pose",
    "kobj",
    "close_mosaic",
    "cos_lr",
}


def resolve_model_arg(model: str) -> str:
    model_path = Path(model)
    if model_path.exists():
        return str(model_path)
    if model_path.is_absolute():
        raise FileNotFoundError(f"Model weights/config not found: {model_path}")
    return model


def configure_wandb(project: str, entity: str | None, mode: str) -> None:
    if mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        try:
            from ultralytics import settings

            settings.update({"wandb": False})
        except Exception:
            pass
        return

    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_MODE"] = mode
    os.environ["WANDB_PROJECT"] = project
    if entity:
        os.environ["WANDB_ENTITY"] = entity

    try:
        from ultralytics import settings

        settings.update({"wandb": True})
    except Exception:
        pass


def load_data_yaml(data_yaml: Path) -> dict[str, Any]:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_yaml}")

    import yaml

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Data yaml must contain a mapping: {data_yaml}")
    return data


def resolve_data_path(data_yaml: Path, data: dict[str, Any], split: str) -> Path:
    if split not in data:
        raise KeyError(f"Missing '{split}' in {data_yaml}")

    root = Path(data.get("path") or data_yaml.parent)
    split_path = Path(str(data[split]))
    if not split_path.is_absolute():
        split_path = root / split_path
    return split_path


def expected_label_values(data: dict[str, Any]) -> int | None:
    kpt_shape = data.get("kpt_shape")
    if not isinstance(kpt_shape, (list, tuple)) or len(kpt_shape) != 2:
        return None
    keypoints, dims = int(kpt_shape[0]), int(kpt_shape[1])
    return 5 + keypoints * dims


def labels_dir_for_images_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts)
    return images_dir.parent.parent / "labels" / images_dir.name


def count_files(path: Path, extensions: set[str]) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in extensions)


def validate_pose_dataset(data_yaml: Path) -> dict[str, Any]:
    data = load_data_yaml(data_yaml)

    if "names" not in data:
        raise KeyError(f"Missing 'names' in {data_yaml}")
    if "kpt_shape" not in data:
        raise KeyError(f"Missing 'kpt_shape' in {data_yaml}; pose training requires keypoint shape.")

    expected_values = expected_label_values(data)
    if expected_values is None:
        raise ValueError(f"Invalid kpt_shape in {data_yaml}: {data.get('kpt_shape')}")

    summary: dict[str, Any] = {
        "data_yaml": str(data_yaml.resolve()),
        "kpt_shape": data.get("kpt_shape"),
        "names": data.get("names"),
        "splits": {},
    }

    for split in ("train", "val"):
        images_dir = resolve_data_path(data_yaml, data, split)
        labels_dir = labels_dir_for_images_dir(images_dir)
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing {split} images: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Missing {split} labels: {labels_dir}")

        image_count = count_files(images_dir, IMAGE_EXTENSIONS)
        label_count = count_files(labels_dir, {".txt"})
        if image_count == 0:
            raise FileNotFoundError(f"No {split} images found under: {images_dir}")
        if label_count == 0:
            raise FileNotFoundError(f"No {split} labels found under: {labels_dir}")

        summary["splits"][split] = {
            "images_dir": str(images_dir.resolve()),
            "labels_dir": str(labels_dir.resolve()),
            "images": image_count,
            "labels": label_count,
        }

    sample_label = next(labels_dir_for_images_dir(resolve_data_path(data_yaml, data, "train")).rglob("*.txt"))
    for line_number, line in enumerate(sample_label.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        value_count = len(stripped.split())
        if value_count != expected_values:
            raise ValueError(
                f"Unexpected pose label width in {sample_label}:{line_number}; "
                f"expected {expected_values} values, got {value_count}."
            )
        break

    return summary


def build_sweep_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": args.sweep_method,
        "metric": {"name": "metrics/mAP50(P)", "goal": "maximize"},
        "parameters": {
            "lr0": {"values": [0.001, 0.002, 0.005, 0.01]},
            "lrf": {"values": [0.01, 0.02, 0.05, 0.1]},
            "momentum": {"values": [0.85, 0.9, 0.937]},
            "weight_decay": {"values": [0.0001, 0.0005, 0.001]},
            "warmup_epochs": {"values": [2.0, 3.0, 5.0]},
            "hsv_h": {"values": [0.0, 0.015, 0.03]},
            "hsv_s": {"values": [0.4, 0.7, 0.9]},
            "hsv_v": {"values": [0.2, 0.4, 0.6]},
            "degrees": {"values": [0.0, 5.0, 10.0]},
            "translate": {"values": [0.05, 0.1, 0.2]},
            "scale": {"values": [0.3, 0.5, 0.7]},
            "fliplr": {"values": [0.0, 0.5]},
            "mosaic": {"values": [0.0, 0.3, 0.7]},
            "pose": {"values": [8.0, 12.0, 16.0]},
            "kobj": {"values": [1.0, 2.0, 4.0]},
        },
    }


def static_wandb_config(args: argparse.Namespace, data_yaml: Path, model_arg: str) -> dict[str, Any]:
    return {
        "task": "pose",
        "data": str(data_yaml),
        "model": model_arg,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "optimizer": args.optimizer,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "pose": args.pose,
        "kobj": args.kobj,
        "close_mosaic": args.close_mosaic,
        "cos_lr": args.cos_lr,
    }


def scalar_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_results_dict(*objects: Any) -> dict[str, float]:
    collected: dict[str, float] = {}
    for obj in objects:
        if obj is None:
            continue
        if isinstance(obj, dict):
            source = obj
        elif hasattr(obj, "results_dict"):
            source = getattr(obj, "results_dict")
        else:
            source = {}

        if callable(source):
            source = source()
        if isinstance(source, dict):
            for key, value in source.items():
                scalar = scalar_or_none(value)
                if scalar is not None:
                    collected[str(key)] = scalar

        fitness = scalar_or_none(getattr(obj, "fitness", None))
        if fitness is not None:
            collected.setdefault("fitness", fitness)

    return collected


def pick_metric(metrics: dict[str, float], candidates: tuple[str, ...]) -> float | None:
    for key in candidates:
        if key in metrics:
            return metrics[key]
    return None


def log_sweep_metrics(wandb_module: Any, train_result: Any, model: Any) -> None:
    trainer = getattr(model, "trainer", None)
    validator = getattr(trainer, "validator", None)
    metrics = collect_results_dict(
        train_result,
        getattr(trainer, "metrics", None),
        getattr(validator, "metrics", None),
    )

    pose_map = pick_metric(
        metrics,
        (
            "metrics/mAP50-95(P)",
            "metrics/mAP50-95(Pose)",
            "metrics/mAP50-95",
            "mAP50-95(P)",
        ),
    )
    fitness = pick_metric(metrics, ("fitness",)) or pose_map
    if fitness is None:
        return

    payload = {"sweep/fitness": fitness}
    if pose_map is not None:
        payload["sweep/map50_95_pose"] = pose_map
    wandb_module.log(payload)
    wandb_module.run.summary.update(payload)


def train_pose_run(
    args: argparse.Namespace,
    data_yaml: Path,
    model_arg: str,
    sweep_run: bool,
) -> None:
    import wandb
    from ultralytics import YOLO

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None)
    run.config.update(static_wandb_config(args, data_yaml, model_arg), allow_val_change=True)

    config = dict(run.config)
    run_name = args.name
    if sweep_run and run.id:
        run_name = f"{args.name}_{run.id}"

    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "optimizer": args.optimizer,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "project": args.project,
        "name": run_name,
        "exist_ok": args.exist_ok,
        "plots": not args.no_plots,
        "save": True,
        "val": True,
        "verbose": True,
    }
    for key in SWEEP_PARAMETER_KEYS:
        if key in config:
            train_kwargs[key] = config[key]

    model = YOLO(model_arg, task="pose")
    model.train(**train_kwargs)
    if wandb.run is not None:
        run.finish()


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    repo_dir = project_dir.parent

    parser = argparse.ArgumentParser(description="Train or sweep a YOLO pose model with W&B.")
    parser.add_argument("--dataset", default=str(repo_dir / "data" / "cat_pose_v1"))
    parser.add_argument("--data-yaml", default=None, help="Override dataset/data.yaml.")
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--pose", type=float, default=12.0)
    parser.add_argument("--kobj", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--project", default=str(project_dir / "runs_pose"))
    parser.add_argument("--name", default="cat_pose_v1_yolo11n_pose_sweep")
    parser.add_argument("--mode", choices=("sweep", "train"), default="sweep")
    parser.add_argument("--sweep-count", type=int, default=5)
    parser.add_argument("--sweep-method", choices=("random", "bayes", "grid"), default="bayes")
    parser.add_argument("--sweep-id", default=None, help="Resume an existing W&B sweep id.")
    parser.add_argument("--wandb-project", default="cat-pose-yolo")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset)
    data_yaml = Path(args.data_yaml) if args.data_yaml else dataset_dir / "data.yaml"
    model_arg = resolve_model_arg(args.model)

    summary = validate_pose_dataset(data_yaml)

    print("Task: pose")
    print("Mode:", args.mode)
    print("W&B:", args.wandb_mode, args.wandb_project)
    print("Data yaml:", data_yaml.resolve())
    print("Dataset summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Model:", model_arg)
    print("Project:", args.project)
    print("Run name:", args.name)
    print("Epochs:", args.epochs)
    print("Batch:", args.batch)
    print("Image size:", args.imgsz)
    print(
        "Optimization:",
        {
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "cos_lr": args.cos_lr,
            "pose": args.pose,
            "kobj": args.kobj,
        },
    )
    print(
        "Strong augmentation:",
        {
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "copy_paste": args.copy_paste,
            "degrees": args.degrees,
            "translate": args.translate,
            "scale": args.scale,
            "fliplr": args.fliplr,
            "hsv_h": args.hsv_h,
            "hsv_s": args.hsv_s,
            "hsv_v": args.hsv_v,
        },
    )

    if args.mode == "sweep":
        sweep_config = build_sweep_config(args)
        print("Sweep count:", args.sweep_count)
        print("Sweep method:", args.sweep_method)
        print("Sweep metric:", sweep_config["metric"])
        if args.dry_run:
            print("Sweep config:")
            print(json.dumps(sweep_config, ensure_ascii=False, indent=2))
            return
        if args.wandb_mode != "online":
            raise ValueError("W&B sweep creation requires --wandb-mode online.")

        import wandb

        configure_wandb(args.wandb_project, args.wandb_entity, args.wandb_mode)
        sweep_id = args.sweep_id or wandb.sweep(
            sweep=sweep_config,
            project=args.wandb_project,
            entity=args.wandb_entity or None,
        )
        print("Sweep id:", sweep_id)
        wandb.agent(
            sweep_id,
            function=lambda: train_pose_run(args, data_yaml, model_arg, sweep_run=True),
            count=args.sweep_count,
            project=args.wandb_project,
            entity=args.wandb_entity or None,
        )
        return

    if args.dry_run:
        return

    configure_wandb(args.wandb_project, args.wandb_entity, args.wandb_mode)
    train_pose_run(args, data_yaml, model_arg, sweep_run=False)


if __name__ == "__main__":
    main()
