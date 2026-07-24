import argparse
import os
from pathlib import Path


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


def resolve_model_arg(model: str) -> str:
    model_path = Path(model)
    if model_path.exists():
        return str(model_path)
    if model_path.is_absolute():
        raise FileNotFoundError(f"Model weights/config not found: {model_path}")
    return model


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    repo_dir = project_dir.parent

    parser = argparse.ArgumentParser(description="Tune YOLO hyperparameters with W&B enabled.")
    parser.add_argument(
        "--dataset",
        default=str(repo_dir / "data" / "v1_yolo26"),
        help="YOLO dataset directory containing images/train, images/val, and labels.",
    )
    parser.add_argument(
        "--model",
        default=str(project_dir / "yolo26n.pt"),
        help="Base YOLO weights.",
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument(
        "--project",
        default="cat-head-feeding-yolo",
        help="Ultralytics project name. Keep this W&B-safe; do not use a Windows path when W&B is enabled.",
    )
    parser.add_argument("--name", default="cat_head_feeding_head_v1_tune5_200e_b32")
    parser.add_argument("--wandb-project", default="cat-head-feeding-yolo")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    dataset_dir = Path(args.dataset)
    model_arg = resolve_model_arg(args.model)
    data_yaml = project_dir / "data.v1.generated.yaml"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    for rel in ("images/train", "labels/train", "images/val", "labels/val"):
        path = dataset_dir / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset path: {path}")

    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_PROJECT"] = args.wandb_project

    from ultralytics import YOLO, settings

    settings.update({"wandb": True})
    write_generated_data_yaml(dataset_dir, data_yaml)
    os.chdir(project_dir)

    print("W&B enabled:", settings["wandb"], os.environ.get("WANDB_MODE"), os.environ.get("WANDB_PROJECT"))
    print("Dataset:", dataset_dir.resolve())
    print("Generated data yaml:", data_yaml)
    print("Model:", model_arg)
    print("Iterations:", args.iterations)
    print("Epochs per iteration:", args.epochs)
    print("Batch:", args.batch)
    print("Optimizer:", args.optimizer)
    print("Project:", args.project)
    print("Tune name:", args.name)

    if args.dry_run:
        return

    model = YOLO(model_arg)
    model.tune(
        use_ray=False,
        iterations=args.iterations,
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        project=args.project,
        name=args.name,
        plots=True,
        save=True,
        val=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
