import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    default_weights = project_dir / "runs" / "cat_head_feeding_head_yolo11n" / "weights" / "best.pt"
    default_source = project_dir.parent / "data" / "prelabel_yolo26" / "images" / "val"

    parser = argparse.ArgumentParser(description="Run YOLO prediction for cat-head detector.")
    parser.add_argument("--weights", default=str(default_weights))
    parser.add_argument("--source", default=str(default_source))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str(project_dir / "predictions"))
    parser.add_argument("--name", default="val_preview")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"

    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        save=True,
        save_txt=True,
        save_conf=True,
    )


if __name__ == "__main__":
    main()
