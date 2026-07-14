#!/usr/bin/env python3
"""Run the deployed pipeline over a bounded video batch and save annotated MP4s."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from eating_behavior.association import associate_feeding_to_cat
from eating_behavior.config import load_yaml
from eating_behavior.identity import IdentityDecision
from eating_behavior.media import SourceSpec, iter_frames
from eating_behavior.models import CoatPrediction, Detection
from eating_behavior.pipeline import RealtimePipeline, event_cat_name, is_named_class
from eating_behavior.pose import PosePrediction, draw_pose_predictions


@dataclass(frozen=True)
class VideoProbe:
    """Stable media properties needed by the writer and batch report."""

    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


class FfmpegAnnotatedWriter:
    """Stream annotated BGR frames to H.264, then copy the source audio."""

    def __init__(
        self,
        *,
        ffmpeg: str,
        destination: Path,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        self.ffmpeg = ffmpeg
        self.destination = destination
        self.width = width
        self.height = height
        self.fps = fps
        self.frames_written = 0
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.video_only = destination.with_name(
            destination.stem + ".video-only.partial.mp4"
        )
        self.mux_partial = destination.with_name(destination.stem + ".mux.partial.mp4")
        for path in (self.destination, self.video_only, self.mux_partial):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite output: {path}")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.9f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.video_only),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self.process.stdin is None or self.process.stderr is None:
            self.abort()
            raise RuntimeError("Could not open ffmpeg pipes")

    def write(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Unexpected frame layout: {frame.shape} {frame.dtype}; "
                f"expected {(self.height, self.width, 3)} uint8"
            )
        contiguous = np.ascontiguousarray(frame)
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(memoryview(contiguous).cast("B"))
        except BrokenPipeError as exc:
            error = self._read_error()
            raise RuntimeError(f"ffmpeg encoder stopped early: {error}") from exc
        self.frames_written += 1

    def close(self, source_video: Path) -> None:
        if self.frames_written <= 0:
            self.abort()
            raise RuntimeError("Cannot finalize an empty annotated video")
        assert self.process.stdin is not None
        self.process.stdin.close()
        return_code = self.process.wait()
        error = self._read_error()
        if return_code != 0:
            self._remove_partials()
            raise RuntimeError(f"ffmpeg encode failed ({return_code}): {error}")
        mux_command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(self.video_only),
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            # Preserve every rendered frame.  Source audio may be shorter or
            # longer than the decoded video stream, so cap the container at
            # the exact rendered duration instead of using -shortest (which
            # would truncate video when audio ends first).
            "-t",
            f"{self.frames_written / self.fps:.9f}",
            "-movflags",
            "+faststart",
            str(self.mux_partial),
        ]
        completed = subprocess.run(
            mux_command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            self._remove_partials()
            raise RuntimeError(
                f"ffmpeg audio mux failed ({completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
        os.replace(self.mux_partial, self.destination)
        self.video_only.unlink(missing_ok=True)

    def abort(self) -> None:
        if self.process.poll() is None:
            if self.process.stdin is not None:
                try:
                    self.process.stdin.close()
                except OSError:
                    pass
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self._remove_partials()

    def _read_error(self) -> str:
        if self.process.stderr is None:
            return ""
        return self.process.stderr.read().decode("utf-8", errors="replace").strip()

    def _remove_partials(self) -> None:
        self.video_only.unlink(missing_ok=True)
        self.mux_partial.unlink(missing_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--models-config", required=True, type=Path)
    parser.add_argument("--pipeline-config", required=True, type=Path)
    parser.add_argument("--limit", type=positive_int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--progress-every", type=positive_int, default=200)
    parser.add_argument("--max-frames", type=positive_int, default=None)
    return parser.parse_args()


def select_videos(input_root: Path, limit: int) -> list[Path]:
    root = input_root.resolve(strict=True)
    videos = sorted(
        path.resolve(strict=True)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    selected = videos[:limit]
    if len(selected) != limit:
        raise ValueError(f"Expected at least {limit} MP4 files, found {len(videos)}")
    return selected


def probe_video(path: Path) -> VideoProbe:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError("Video cannot be opened")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("Video has no decodable first frame")
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("Video has invalid dimensions")
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError("Video has invalid FPS")
    if frame_count <= 0:
        raise RuntimeError("Video has invalid frame count")
    return VideoProbe(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=frame_count / fps,
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def draw_text(
    frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(
        frame, (x, y - height - baseline - 6), (x + width + 8, y + 3), color, -1
    )
    cv2.putText(
        frame,
        text,
        (x + 4, y - 3),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def render_frame(
    frame: np.ndarray,
    detections: tuple[Detection, ...],
    cat_decisions: tuple[tuple[Detection, CoatPrediction, IdentityDecision], ...],
    poses: tuple[PosePrediction, ...],
    *,
    source_id: str,
    frame_index: int,
    timestamp_seconds: float,
    eating_state: str,
    cat_classes: set[str],
    feeding_classes: set[str],
    association_max_distance_ratio: float,
    association_ambiguity_margin: float,
    pose_keypoint_confidence: float,
) -> np.ndarray:
    rendered = draw_pose_predictions(
        frame,
        poses,
        keypoint_threshold=pose_keypoint_confidence,
    )
    cat_by_detection = {decision[0]: decision for decision in cat_decisions}
    for detection in detections:
        if is_named_class(detection, cat_classes):
            color = (40, 190, 40)
            _, coat, identity = cat_by_detection[detection]
            confirmed_name = event_cat_name(identity) or "unknown"
            margin = coat.confidence - coat.second_confidence
            label = (
                f"cat {detection.confidence:.2f} coat={coat.class_name} "
                f"{coat.confidence:.2f}/{margin:.2f} id={confirmed_name}"
            )
        elif is_named_class(detection, feeding_classes):
            color = (20, 90, 230)
            association = associate_feeding_to_cat(
                detection,
                list(cat_decisions),
                max_distance_ratio=association_max_distance_ratio,
                ambiguity_margin=association_ambiguity_margin,
            )
            if association is None:
                association_label = "unassociated"
            else:
                association_label = (
                    event_cat_name(association.identity) or association.coat.class_name
                )
                fx1, fy1, fx2, fy2 = detection.xyxy
                cx1, cy1, cx2, cy2 = association.cat_detection.xyxy
                cv2.line(
                    rendered,
                    ((fx1 + fx2) // 2, (fy1 + fy2) // 2),
                    ((cx1 + cx2) // 2, (cy1 + cy2) // 2),
                    color,
                    2,
                    cv2.LINE_AA,
                )
            label = f"feeding_head {detection.confidence:.2f} -> {association_label}"
        else:
            color = (220, 160, 30)
            label = f"{detection.class_name} {detection.confidence:.2f}"
        x1, y1, x2, y2 = detection.xyxy
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 3)
        label_y = max(24, y1)
        draw_text(rendered, label, (x1, label_y), color)
    header = f"source={source_id} frame={frame_index} t={timestamp_seconds:.2f}s state={eating_state}"
    draw_text(rendered, header, (12, 30), (35, 35, 35))
    return rendered


def safe_output_path(input_root: Path, output_root: Path, source_video: Path) -> Path:
    relative = source_video.resolve().relative_to(input_root.resolve())
    destination = (
        output_root / "videos" / relative.parent / f"{relative.stem}_annotated.mp4"
    )
    resolved_parent = destination.parent.resolve(strict=False)
    video_root = (output_root / "videos").resolve(strict=False)
    if not resolved_parent.is_relative_to(video_root):
        raise ValueError("Output path escaped the video root")
    return destination


def build_effective_config(
    base_config: dict[str, Any], videos: list[Path]
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    source_ids = [path.parent.name for path in videos]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Selected videos have duplicate source IDs")
    config["sources"] = [
        {"source_id": source_id, "stream_url": str(path)}
        for source_id, path in zip(source_ids, videos)
    ]
    config["worker"] = {
        "source_ids": source_ids,
        "require_assignments": True,
        "allow_debug_source": False,
    }
    config.setdefault("output", {})["save_event_jsonl"] = True
    config.setdefault("output", {})["save_event_csv"] = True
    return config


def verify_output(
    path: Path, expected: VideoProbe, processed_frames: int
) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("Annotated output is missing or empty")
    actual = probe_video(path)
    if actual.width != expected.width or actual.height != expected.height:
        raise RuntimeError("Annotated output dimensions changed")
    if not math.isclose(actual.fps, expected.fps, rel_tol=0.01, abs_tol=0.05):
        raise RuntimeError("Annotated output FPS changed")
    if actual.frame_count < max(1, processed_frames - 2):
        raise RuntimeError(
            f"Annotated output lost frames: expected {processed_frames}, got {actual.frame_count}"
        )
    return {"bytes": path.stat().st_size, "probe": asdict(actual)}


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve(strict=True)
    output_root = args.output_root.resolve(strict=False)
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {output_root}")
    output_root.mkdir(parents=True)
    selected = select_videos(input_root, args.limit)
    probes = {path: probe_video(path) for path in selected}
    relative_paths = [str(path.relative_to(input_root)) for path in selected]

    # Resolve CLI paths at the process boundary.  Installed wheels do not
    # contain the repository-level configs directory, so handing a relative
    # path to the package helper would incorrectly anchor it inside site-packages.
    models_config = load_yaml(args.models_config.resolve(strict=True))
    base_pipeline_config = load_yaml(args.pipeline_config.resolve(strict=True))
    effective_pipeline_config = build_effective_config(base_pipeline_config, selected)
    (output_root / "effective_pipeline.yaml").write_text(
        yaml.safe_dump(effective_pipeline_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "schema_version": 2,
        "status": "initializing",
        "started_at": utc_now(),
        "finished_at": None,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "device": args.device,
        "selected_count": len(selected),
        "selected_videos": relative_paths,
        "results": [],
    }
    atomic_write_json(output_root / "batch_summary.json", summary)

    pipeline = RealtimePipeline.from_configs(
        models_config=models_config,
        pipeline_config=effective_pipeline_config,
        output_dir=output_root / "events",
        device=args.device,
        show=False,
    )
    source_specs = pipeline.select_source_specs()
    summary["pose_enabled"] = pipeline.config.pose_enabled
    summary["status"] = "running"
    atomic_write_json(output_root / "batch_summary.json", summary)

    batch_started = time.monotonic()
    for video_number, (source_spec, source_video) in enumerate(
        zip(source_specs, selected), 1
    ):
        probe = probes[source_video]
        destination = safe_output_path(input_root, output_root, source_video)
        result: dict[str, Any] = {
            "index": video_number,
            "source_id": source_spec.source_id,
            "input": str(source_video.relative_to(input_root)),
            "output": str(destination.relative_to(output_root)),
            "status": "running",
            "input_probe": asdict(probe),
            "processed_frames": 0,
            "pose_frames": 0,
            "pose_instances": 0,
            "pose_visible_keypoints": 0,
            "pose_error_frames": 0,
            "pose_error_codes": {},
            "started_at": utc_now(),
            "finished_at": None,
        }
        summary["results"].append(result)
        atomic_write_json(output_root / "batch_summary.json", summary)
        print(
            f"[{video_number:02d}/{len(selected):02d}] START {result['input']} "
            f"frames={probe.frame_count} fps={probe.fps:.3f}",
            flush=True,
        )

        writer = FfmpegAnnotatedWriter(
            ffmpeg=args.ffmpeg,
            destination=destination,
            width=probe.width,
            height=probe.height,
            fps=probe.fps,
        )
        source_started = time.monotonic()
        last_timestamp: float | None = None
        end_reason = "eof"
        try:
            for packet in iter_frames(
                SourceSpec(source_id=source_spec.source_id, uri=str(source_video)),
                vid_stride=1,
                max_frames=args.max_frames,
            ):
                last_timestamp = packet.timestamp_seconds
                analysis = pipeline.process_frame(
                    packet.source,
                    packet.timestamp_seconds,
                    packet.frame,
                )
                if analysis.poses:
                    result["pose_frames"] += 1
                    result["pose_instances"] += len(analysis.poses)
                    result["pose_visible_keypoints"] += sum(
                        pose.visible_keypoint_count(
                            pipeline.config.pose_keypoint_confidence
                        )
                        for pose in analysis.poses
                    )
                if analysis.pose_error_code:
                    result["pose_error_frames"] += 1
                    error_codes = result["pose_error_codes"]
                    error_codes[analysis.pose_error_code] = (
                        error_codes.get(analysis.pose_error_code, 0) + 1
                    )
                rendered = render_frame(
                    packet.frame,
                    analysis.detections,
                    analysis.cat_decisions,
                    analysis.poses,
                    source_id=packet.source,
                    frame_index=packet.frame_index,
                    timestamp_seconds=packet.timestamp_seconds,
                    eating_state=analysis.eating_state,
                    cat_classes=pipeline.config.cat_classes,
                    feeding_classes=pipeline.config.feeding_classes,
                    association_max_distance_ratio=(
                        pipeline.config.feeding_association_max_distance_ratio
                    ),
                    association_ambiguity_margin=(
                        pipeline.config.feeding_association_ambiguity_margin
                    ),
                    pose_keypoint_confidence=(pipeline.config.pose_keypoint_confidence),
                )
                writer.write(rendered)
                result["processed_frames"] += 1
                if result["processed_frames"] % args.progress_every == 0:
                    elapsed = time.monotonic() - source_started
                    rate = result["processed_frames"] / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[{video_number:02d}/{len(selected):02d}] "
                        f"frames={result['processed_frames']}/{probe.frame_count} rate={rate:.2f} fps",
                        flush=True,
                    )
                    result["processing_seconds"] = round(elapsed, 3)
                    atomic_write_json(output_root / "batch_summary.json", summary)
            if result["processed_frames"] == 0:
                raise RuntimeError("No frames were processed")
            if last_timestamp is not None:
                pipeline.finish_source(source_spec.source_id, last_timestamp, "eof")
            writer.close(source_video)
            output_check = verify_output(
                destination,
                probe,
                int(result["processed_frames"]),
            )
            result.update(output_check)
            result["status"] = "completed"
        except Exception as exc:  # keep the rest of the bounded test batch running
            end_reason = "error"
            writer.abort()
            if last_timestamp is not None:
                try:
                    pipeline.finish_source(
                        source_spec.source_id, last_timestamp, end_reason
                    )
                except Exception as finish_exc:
                    result["finish_error"] = (
                        f"{type(finish_exc).__name__}: {finish_exc}"
                    )
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"[{video_number:02d}/{len(selected):02d}] FAILED {result['error']}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            elapsed = time.monotonic() - source_started
            result["processing_seconds"] = round(elapsed, 3)
            result["finished_at"] = utc_now()
            atomic_write_json(output_root / "batch_summary.json", summary)
        print(
            f"[{video_number:02d}/{len(selected):02d}] {result['status'].upper()} "
            f"frames={result['processed_frames']} elapsed={result['processing_seconds']}s",
            flush=True,
        )

    pipeline.recorder.flush_csv()
    completed = sum(item["status"] == "completed" for item in summary["results"])
    failed = sum(item["status"] == "failed" for item in summary["results"])
    summary["completed_count"] = completed
    summary["failed_count"] = failed
    summary["processing_seconds"] = round(time.monotonic() - batch_started, 3)
    summary["finished_at"] = utc_now()
    summary["status"] = "completed" if failed == 0 else "completed_with_failures"
    atomic_write_json(output_root / "batch_summary.json", summary)
    print(
        f"BATCH {summary['status']} completed={completed} failed={failed} "
        f"elapsed={summary['processing_seconds']}s output={output_root}",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
