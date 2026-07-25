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
from array import array
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

from eating_behavior.association import associate_feeding_to_cat
from eating_behavior.config import load_yaml
from eating_behavior.identity import IdentityDecision
from eating_behavior.media import SourceSpec, iter_frames
from eating_behavior.models import CoatPrediction, Detection
from eating_behavior.pipeline import RealtimePipeline, is_named_class
from eating_behavior.pose import PosePrediction, draw_pose_predictions


@dataclass(frozen=True)
class VideoProbe:
    """Stable media properties needed by the writer and batch report."""

    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


class PoseStabilityDiagnostics:
    """Measure bbox-local keypoint motion for raw and filtered track poses."""

    def __init__(self, visibility_threshold: float) -> None:
        self.visibility_threshold = visibility_threshold
        self._previous: dict[
            tuple[str, int, str], dict[int, tuple[float, float]]
        ] = {}
        # Compact C doubles keep an all-video run bounded in practice without
        # the per-value object overhead of Python float lists.
        self._steps: dict[str, array[float]] = {
            "raw": array("d"),
            "filtered": array("d"),
        }

    def observe(
        self,
        *,
        source_id: str,
        track_id: int | None,
        raw_pose: PosePrediction | None,
        filtered_pose: PosePrediction | None,
    ) -> None:
        """Record adjacent-frame local steps without joining untracked cats."""
        if track_id is None:
            return
        self._observe_pose(source_id, track_id, "raw", raw_pose)
        self._observe_pose(source_id, track_id, "filtered", filtered_pose)

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-safe distribution for each pose stream."""
        return {
            "coordinate_space": "bbox_local_xy",
            "visibility_threshold": self.visibility_threshold,
            "large_step_threshold": 0.2,
            "raw": summarize_pose_steps(self._steps["raw"]),
            "filtered": summarize_pose_steps(self._steps["filtered"]),
        }

    def _observe_pose(
        self,
        source_id: str,
        track_id: int,
        stream: str,
        pose: PosePrediction | None,
    ) -> None:
        key = (source_id, track_id, stream)
        current = self._local_visible_points(pose)
        previous = self._previous.get(key, {})
        for index in current.keys() & previous.keys():
            current_x, current_y = current[index]
            previous_x, previous_y = previous[index]
            self._steps[stream].append(
                math.hypot(current_x - previous_x, current_y - previous_y)
            )
        # Replacing the whole snapshot deliberately breaks continuity across a
        # missing/low-confidence point instead of measuring a multi-frame jump.
        self._previous[key] = current

    def _local_visible_points(
        self, pose: PosePrediction | None
    ) -> dict[int, tuple[float, float]]:
        if pose is None:
            return {}
        x1, y1, x2, y2 = pose.cat_xyxy
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return {}
        return {
            point.index: ((point.x - x1) / width, (point.y - y1) / height)
            for point in pose.keypoints
            if point.confidence >= self.visibility_threshold
            and math.isfinite(point.x)
            and math.isfinite(point.y)
        }


def summarize_pose_steps(values: Iterable[float]) -> dict[str, Any]:
    """Summarize normalized pose steps while keeping empty runs explicit."""
    if isinstance(values, array) and values.typecode == "d":
        samples = np.frombuffer(values, dtype=np.float64)
    else:
        samples = np.fromiter(values, dtype=np.float64)
    if samples.size == 0:
        return {
            "step_count": 0,
            "mean_local_step": None,
            "p50_local_step": None,
            "p95_local_step": None,
            "p99_local_step": None,
            "max_local_step": None,
            "large_step_count": 0,
            "large_step_rate": None,
        }
    large_step_count = int(np.count_nonzero(samples > 0.2))
    return {
        "step_count": int(samples.size),
        "mean_local_step": round(float(np.mean(samples)), 6),
        "p50_local_step": round(float(np.percentile(samples, 50)), 6),
        "p95_local_step": round(float(np.percentile(samples, 95)), 6),
        "p99_local_step": round(float(np.percentile(samples, 99)), 6),
        "max_local_step": round(float(np.max(samples)), 6),
        "large_step_count": large_step_count,
        "large_step_rate": round(large_step_count / int(samples.size), 6),
    }


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


def cat_pose_label(cat: Any, visibility_threshold: float) -> str:
    """Build the user-facing posture label without operational diagnostics."""
    del visibility_threshold
    posture = (
        getattr(cat, "posture", "")
        or getattr(cat, "observed_posture", "")
        or "unknown"
    )
    return f"posture={posture}"


def accumulate_pose_frame(
    result: dict[str, Any],
    cats: Iterable[Any],
    *,
    source_id: str,
    visibility_threshold: float,
    stability: PoseStabilityDiagnostics,
    batch_stability: PoseStabilityDiagnostics | None = None,
    track_ids: set[int] | None = None,
) -> dict[Detection, str]:
    """Update additive pose diagnostics and return per-cat render labels."""
    cats = tuple(cats)
    raw_frame = False
    filtered_frame = False
    labels: dict[Detection, str] = {}
    for cat in cats:
        filtered_pose = getattr(cat, "pose", None)
        raw_pose = (
            getattr(cat, "raw_pose")
            if hasattr(cat, "raw_pose")
            else filtered_pose
        )
        track_id = getattr(cat, "track_id", None)
        if track_id is None:
            result["untracked_cat_instances"] += 1
        else:
            result["tracked_cat_instances"] += 1
            if track_ids is not None:
                track_ids.add(track_id)

        if raw_pose is not None:
            raw_frame = True
            result["raw_pose_instances"] += 1
            result["raw_pose_visible_keypoints"] += raw_pose.visible_keypoint_count(
                visibility_threshold
            )
        if filtered_pose is not None:
            filtered_frame = True
            visible = filtered_pose.visible_keypoint_count(visibility_threshold)
            result["filtered_pose_instances"] += 1
            result["filtered_pose_visible_keypoints"] += visible
            # Preserve the pre-existing summary fields and their filtered-pose
            # meaning for downstream consumers.
            result["pose_instances"] += 1
            result["pose_visible_keypoints"] += visible

        status = getattr(cat, "pose_status", "") or (
            "raw" if raw_pose is not None else "missing"
        )
        status_counts = result["pose_status_counts"]
        status_counts[status] = status_counts.get(status, 0) + 1
        rejection = getattr(cat, "pose_rejection_code", "")
        if rejection:
            rejection_codes = result["pose_rejection_codes"]
            rejection_codes[rejection] = rejection_codes.get(rejection, 0) + 1
        result["pose_held_keypoints"] += int(
            getattr(cat, "pose_held_keypoints", 0)
        )
        result["pose_dropped_keypoints"] += int(
            getattr(cat, "pose_dropped_keypoints", 0)
        )
        observed_posture = getattr(cat, "observed_posture", "")
        if observed_posture:
            observation_counts = result.setdefault(
                "posture_observation_counts",
                {},
            )
            observation_counts[observed_posture] = (
                observation_counts.get(observed_posture, 0) + 1
            )
        posture = getattr(cat, "posture", "")
        if posture:
            state_counts = result.setdefault("posture_state_counts", {})
            state_counts[posture] = state_counts.get(posture, 0) + 1

        stability.observe(
            source_id=source_id,
            track_id=track_id,
            raw_pose=raw_pose,
            filtered_pose=filtered_pose,
        )
        if batch_stability is not None:
            batch_stability.observe(
                source_id=source_id,
                track_id=track_id,
                raw_pose=raw_pose,
                filtered_pose=filtered_pose,
            )
        labels[cat.detection] = cat_pose_label(cat, visibility_threshold)

    if raw_frame:
        result["raw_pose_frames"] += 1
    if filtered_frame:
        result["filtered_pose_frames"] += 1
        result["pose_frames"] += 1
    return labels


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
    cat_pose_labels: dict[Detection, str] | None = None,
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
            _, coat, _ = cat_by_detection[detection]
            posture_label = (
                cat_pose_labels.get(detection, "posture=unknown")
                if cat_pose_labels is not None
                else "posture=unknown"
            )
            label = f"coat={coat.class_name} {posture_label}"
        elif is_named_class(detection, feeding_classes):
            color = (20, 90, 230)
            association = associate_feeding_to_cat(
                detection,
                list(cat_decisions),
                max_distance_ratio=association_max_distance_ratio,
                ambiguity_margin=association_ambiguity_margin,
            )
            if association is not None:
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
            label = detection.class_name
        else:
            color = (220, 160, 30)
            label = detection.class_name
        x1, y1, x2, y2 = detection.xyxy
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 3)
        label_y = max(24, y1)
        draw_text(rendered, label, (x1, label_y), color)
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


def merge_count_maps(
    results: Iterable[dict[str, Any]], field_name: str
) -> dict[str, int]:
    """Merge per-video categorical counters into one deterministic map."""
    merged: dict[str, int] = {}
    for result in results:
        for key, count in result.get(field_name, {}).items():
            merged[key] = merged.get(key, 0) + int(count)
    return dict(sorted(merged.items()))


def batch_pose_diagnostics(
    results: Iterable[dict[str, Any]], stability: PoseStabilityDiagnostics
) -> dict[str, Any]:
    """Aggregate additive counters while retaining every legacy result field."""
    results = tuple(results)
    summed_fields = (
        "raw_pose_frames",
        "raw_pose_instances",
        "raw_pose_visible_keypoints",
        "filtered_pose_frames",
        "filtered_pose_instances",
        "filtered_pose_visible_keypoints",
        "pose_held_keypoints",
        "pose_dropped_keypoints",
        "tracked_cat_instances",
        "untracked_cat_instances",
    )
    summary = {
        field_name: sum(int(result.get(field_name, 0)) for result in results)
        for field_name in summed_fields
    }
    summary.update(
        {
            "unique_source_track_count": sum(
                len(result.get("track_ids", [])) for result in results
            ),
            "pose_status_counts": merge_count_maps(results, "pose_status_counts"),
            "pose_rejection_codes": merge_count_maps(
                results, "pose_rejection_codes"
            ),
            "posture_observation_counts": merge_count_maps(
                results, "posture_observation_counts"
            ),
            "posture_state_counts": merge_count_maps(
                results, "posture_state_counts"
            ),
            "stability": stability.summary(),
        }
    )
    return summary


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
        "schema_version": 3,
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
    summary["posture_enabled"] = (
        pipeline.config.pose_enabled and pipeline.config.posture.enabled
    )
    summary["status"] = "running"
    atomic_write_json(output_root / "batch_summary.json", summary)

    batch_started = time.monotonic()
    batch_stability = PoseStabilityDiagnostics(
        pipeline.config.pose_keypoint_confidence
    )
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
            "raw_pose_frames": 0,
            "raw_pose_instances": 0,
            "raw_pose_visible_keypoints": 0,
            "filtered_pose_frames": 0,
            "filtered_pose_instances": 0,
            "filtered_pose_visible_keypoints": 0,
            "pose_status_counts": {},
            "pose_rejection_codes": {},
            "posture_observation_counts": {},
            "posture_state_counts": {},
            "pose_held_keypoints": 0,
            "pose_dropped_keypoints": 0,
            "tracked_cat_instances": 0,
            "untracked_cat_instances": 0,
            "track_ids": [],
            "started_at": utc_now(),
            "finished_at": None,
        }
        pose_stability = PoseStabilityDiagnostics(
            pipeline.config.pose_keypoint_confidence
        )
        track_ids: set[int] = set()
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
                cat_pose_labels = accumulate_pose_frame(
                    result,
                    analysis.cats,
                    source_id=packet.source,
                    visibility_threshold=(pipeline.config.pose_keypoint_confidence),
                    stability=pose_stability,
                    batch_stability=batch_stability,
                    track_ids=track_ids,
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
                    cat_pose_labels=cat_pose_labels,
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
                    result["track_ids"] = sorted(track_ids)
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
            result["track_ids"] = sorted(track_ids)
            result["pose_stability"] = pose_stability.summary()
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
    summary["pose_diagnostics"] = batch_pose_diagnostics(
        summary["results"], batch_stability
    )
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
