import argparse
import os
from pathlib import Path


def configure_wandb(enable: bool, project: str) -> None:
    if enable:
        os.environ.pop("WANDB_DISABLED", None)
        os.environ["WANDB_MODE"] = "online"
        os.environ["WANDB_PROJECT"] = project
        try:
            from ultralytics import settings

            settings.update({"wandb": True})
        except Exception:
            # Ultralytics is imported later; env vars above still request online W&B.
            pass
        return

    # Keep W&B fully disabled even if wandb is installed in the environment.
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    try:
        from ultralytics import settings

        settings.update({"wandb": False})
    except Exception:
        # Ultralytics is imported later; env vars above still keep external W&B disabled.
        pass


def read_dataset_classes(dataset_dir: Path) -> list[str]:
    classes_path = dataset_dir / "classes.txt"
    if classes_path.exists():
        classes = [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if classes:
            return classes

    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.exists():
        try:
            from ultralytics.utils import yaml_load

            data = yaml_load(str(data_yaml))
            names = data.get("names", [])
            if isinstance(names, dict):
                return [names[idx] for idx in sorted(names)]
            if isinstance(names, list):
                return [str(name) for name in names]
        except Exception as exc:
            raise ValueError(f"Failed to read class names from {data_yaml}: {exc}") from exc

    raise FileNotFoundError(f"Missing class names: expected {classes_path} or {data_yaml}")


def write_generated_data_yaml(dataset_dir: Path, output_path: Path) -> None:
    dataset_dir = dataset_dir.resolve()
    classes = read_dataset_classes(dataset_dir)
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(classes))
    test_text = "test: images/test\n" if (dataset_dir / "images" / "test").exists() else ""
    yaml_text = (
        f"path: {dataset_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"{test_text}"
        f"nc: {len(classes)}\n"
        "names:\n"
        f"{names}\n"
    )
    output_path.write_text(yaml_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    repo_dir = project_dir.parent

    parser = argparse.ArgumentParser(description="Train YOLO cat-head detector.")
    parser.add_argument(
        "--dataset",
        default=str(repo_dir / "data" / "prelabel_yolo26"),
        help="YOLO dataset directory containing images/train, images/val, labels/train, labels/val.",
    )
    parser.add_argument(
        "--model",
        default=str(repo_dir / "yolo11n.pt"),
        help="Base YOLO weights, for example yolo11n.pt.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="Use 0 for GPU, cpu for CPU.")
    parser.add_argument("--workers", type=int, default=0, help="Use 0 on Windows for fewer dataloader issues.")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--name", default="cat_head_feeding_head_yolo11n")
    parser.add_argument("--project", default=str(project_dir / "runs"))
    parser.add_argument("--enable-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cat-head-feeding-yolo")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot/image generation and W&B media logging; metrics and checkpoints are still saved.",
    )
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only validate config and print planned training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_wandb(args.enable_wandb, args.wandb_project)
    project_dir = Path(__file__).resolve().parent
    dataset_dir = Path(args.dataset)
    model_path = Path(args.model)
    data_yaml = project_dir / "data.generated.yaml"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not (dataset_dir / "images" / "train").exists():
        raise FileNotFoundError(f"Missing train images: {dataset_dir / 'images' / 'train'}")
    if not (dataset_dir / "labels" / "train").exists():
        raise FileNotFoundError(f"Missing train labels: {dataset_dir / 'labels' / 'train'}")
    if not (dataset_dir / "images" / "val").exists():
        raise FileNotFoundError(f"Missing val images: {dataset_dir / 'images' / 'val'}")
    if not (dataset_dir / "labels" / "val").exists():
        raise FileNotFoundError(f"Missing val labels: {dataset_dir / 'labels' / 'val'}")

    write_generated_data_yaml(dataset_dir, data_yaml)

    print(
        "W&B:",
        "enabled" if args.enable_wandb else "disabled",
        os.environ.get("WANDB_MODE"),
        os.environ.get("WANDB_PROJECT", ""),
    )
    print("Dataset:", dataset_dir.resolve())
    print("Generated data yaml:", data_yaml)
    print("Model:", model_path)
    print("Project:", args.project)
    print("Run name:", args.name)
    print("Plots:", not args.no_plots)

    if args.dry_run:
        return

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        plots=not args.no_plots,
        save=True,
        val=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
