"""Command-line entrypoint.

Two modes are supported:
1. Debug mode: --source plus optional --source-id for a single input.
2. Worker mode: no --source; run configured worker.source_ids.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import PROJECT_ROOT, load_yaml
from .media import is_network_source, source_id_list
from .pipeline import RealtimePipeline


def parse_source_ids(values: list[str] | None) -> tuple[str, ...]:
    """Parse repeated or comma-separated --source-id values."""
    source_ids: list[str] = []
    for value in values or []:
        source_ids.extend(source_id_list(value))
    return tuple(source_ids)


def parse_args() -> argparse.Namespace:
    """Define realtime inference CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run realtime cat detection, coat routing, identity confirmation, "
            "and eating-event announcement recording on video or stream input."
        )
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video file, video directory, camera index such as 0, or RTSP/HTTP stream URL.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help=(
            "Stable configured source_id/camera_id to run. Can be repeated or comma-separated. "
            "When paired with --source, tags that raw single source with this stable id."
        ),
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
    """Load configs, create the pipeline, and run the selected mode."""
    args = parse_args()
    if args.vid_stride < 1:
        print("--vid-stride must be >= 1", file=sys.stderr)
        return 2
    try:
        source_ids = parse_source_ids(args.source_id)
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.source and len(source_ids) > 1:
        print("--source can only be paired with at most one --source-id", file=sys.stderr)
        return 2
    if args.source and is_network_source(args.source) and not source_ids:
        print("Network stream --source requires an explicit stable --source-id", file=sys.stderr)
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
    if args.source:
        # Debug mode: bind a raw source URI to one stable source_id.
        pipeline.run_source(
            args.source,
            source_id=source_ids[0] if source_ids else None,
            vid_stride=args.vid_stride,
            max_frames=args.max_frames,
        )
    else:
        # Worker mode: run assigned source IDs from the local config snapshot.
        pipeline.run_configured_sources(
            source_ids=source_ids or None,
            vid_stride=args.vid_stride,
            max_frames=args.max_frames,
        )
    print(f"Announcement JSONL: {output_dir / 'announcement_events.jsonl'}")
    print(f"Announcement CSV: {output_dir / 'announcement_events.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
