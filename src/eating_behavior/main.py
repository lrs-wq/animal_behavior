from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import PROJECT_ROOT, load_yaml
from .pipeline import RealtimePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run realtime cat detection, coat routing, identity confirmation, "
            "and eating-event announcement recording on video or stream input."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Video file, video directory, camera index such as 0, or RTSP/HTTP stream URL.",
    )
    parser.add_argument(
        "--models-config",
        default=str(PROJECT_ROOT / "configs" / "models.yaml"),
    )
    parser.add_argument(
        "--pipeline-config",
        default=str(PROJECT_ROOT / "configs" / "pipeline.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "realtime"),
    )
    parser.add_argument("--device", default=None, help="YOLO device, for example cuda:0, 0, or cpu.")
    parser.add_argument("--vid-stride", type=int, default=1, help="Run inference every N frames.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional debug frame limit.")
    parser.add_argument("--show", action="store_true", help="Show live frames while processing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.vid_stride < 1:
        print("--vid-stride must be >= 1", file=sys.stderr)
        return 2
    models_config = load_yaml(args.models_config)
    pipeline_config = load_yaml(args.pipeline_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = RealtimePipeline.from_configs(
        models_config=models_config,
        pipeline_config=pipeline_config,
        output_dir=output_dir,
        device=args.device,
        show=args.show,
    )
    pipeline.run_source(args.source, vid_stride=args.vid_stride, max_frames=args.max_frames)
    print(f"Announcement JSONL: {output_dir / 'announcement_events.jsonl'}")
    print(f"Announcement CSV: {output_dir / 'announcement_events.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

