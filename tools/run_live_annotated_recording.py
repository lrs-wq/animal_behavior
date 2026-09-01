#!/usr/bin/env python3
"""Continuously analyze one live stream and output annotated video."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from eating_behavior.config import load_yaml
from eating_behavior.pipeline import RealtimePipeline
from tools.run_batch_test import cat_pose_label, render_frame


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and > 0")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def inference_due(
    session_frame_index: int,
    inference_interval: int,
    *,
    has_cached_analysis: bool,
) -> bool:
    """Return whether this frame needs fresh inference results."""

    return (
        not has_cached_analysis
        or session_frame_index % inference_interval == 0
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--models-config", required=True, type=Path)
    parser.add_argument("--pipeline-config", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--segment-seconds", type=positive_float, default=120.0)
    parser.add_argument("--detection-confidence", type=probability, default=0.60)
    parser.add_argument("--min-free-gb", type=positive_float, default=20.0)
    parser.add_argument("--reconnect-delay-seconds", type=positive_float, default=5.0)
    parser.add_argument("--disk-recheck-seconds", type=positive_float, default=30.0)
    parser.add_argument("--open-timeout-seconds", type=positive_float, default=10.0)
    parser.add_argument("--read-timeout-seconds", type=positive_float, default=10.0)
    parser.add_argument("--fallback-fps", type=positive_float, default=25.0)
    parser.add_argument(
        "--inference-interval",
        type=positive_int,
        default=1,
        help=(
            "Run model inference once every N input frames and reuse the latest "
            "analysis for intervening frames"
        ),
    )
    parser.add_argument("--progress-every", type=positive_int, default=250)
    parser.add_argument("--max-frames", type=positive_int, default=None)
    parser.add_argument(
        "--publish-url",
        help="Optional RTMP URL that receives the annotated live stream",
    )
    parser.add_argument(
        "--no-recording",
        action="store_true",
        help="Disable MP4 segments; requires --publish-url",
    )
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    return parser.parse_args()


def log_event(event: str, **fields: Any) -> None:
    payload = {"timestamp": utc_now(), "event": event, **fields}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def effective_pipeline_config(
    base: dict[str, Any],
    *,
    source_id: str,
    source: str,
    detection_confidence: float,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config.setdefault("detection", {})["confidence"] = detection_confidence
    config["sources"] = [{"source_id": source_id, "stream_url": source}]
    config["worker"] = {
        "source_ids": [source_id],
        "require_assignments": True,
        "allow_debug_source": False,
    }
    config.setdefault("output", {})["save_event_jsonl"] = True
    config.setdefault("output", {})["save_event_csv"] = True
    return config


def open_capture(
    source: str,
    *,
    open_timeout_seconds: float,
    read_timeout_seconds: float,
) -> tuple[cv2.VideoCapture, np.ndarray, float]:
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        int(open_timeout_seconds * 1000),
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        int(read_timeout_seconds * 1000),
    ]
    try:
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
    except (TypeError, cv2.error):
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("stream could not be opened")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise RuntimeError("stream opened but did not provide a frame")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    return capture, frame, fps


class FfmpegSegmentWriter:
    """Write one raw-video segment and publish it only after ffmpeg succeeds."""

    def __init__(
        self,
        *,
        ffmpeg: str,
        video_root: Path,
        source_id: str,
        width: int,
        height: int,
        fps: float,
        target_frames: int,
    ) -> None:
        started_local = datetime.now().astimezone()
        day_dir = video_root / started_local.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{source_id}_{started_local.strftime('%Y%m%d_%H%M%S')}_annotated"
        destination = day_dir / f"{stem}.mp4"
        suffix = 1
        while destination.exists() or destination.with_name(
            f"{destination.stem}.partial.mp4"
        ).exists():
            destination = day_dir / f"{stem}_{suffix:02d}.mp4"
            suffix += 1
        partial = destination.with_name(f"{destination.stem}.partial.mp4")

        self.destination = destination
        self.partial = partial
        self.width = width
        self.height = height
        self.fps = fps
        self.target_frames = target_frames
        self.frames = 0
        self.started_at = utc_now()
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
            f"{fps:.6f}",
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
            str(partial),
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
            raise RuntimeError("ffmpeg pipes were not created")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"frame dimensions changed from {self.width}x{self.height} "
                f"to {frame.shape[1]}x{frame.shape[0]}"
            )
        if frame.dtype != np.uint8:
            raise ValueError("annotated frame must use uint8 pixels")
        contiguous = np.ascontiguousarray(frame)
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(contiguous.tobytes())
        except BrokenPipeError as exc:
            error = self._stderr_text()
            raise RuntimeError(f"ffmpeg stopped while writing: {error}") from exc
        self.frames += 1

    def close(self) -> dict[str, Any]:
        if self.frames == 0:
            self.abort()
            self.partial.unlink(missing_ok=True)
            raise RuntimeError("refusing to publish an empty segment")
        assert self.process.stdin is not None
        self.process.stdin.close()
        try:
            return_code = self.process.wait(timeout=60)
        except subprocess.TimeoutExpired as exc:
            self.process.kill()
            self.process.wait(timeout=10)
            raise RuntimeError("ffmpeg did not finish the segment in 60 seconds") from exc
        error = self._stderr_text()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg failed with exit code {return_code}: {error or 'no details'}"
            )
        if not self.partial.is_file() or self.partial.stat().st_size <= 0:
            raise RuntimeError("ffmpeg produced no usable segment")
        os.replace(self.partial, self.destination)
        return {
            "path": str(self.destination),
            "started_at": self.started_at,
            "finished_at": utc_now(),
            "frames": self.frames,
            "fps": self.fps,
            "duration_seconds": self.frames / self.fps,
            "width": self.width,
            "height": self.height,
            "bytes": self.destination.stat().st_size,
            "complete_duration": self.frames == self.target_frames,
        }

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

    def _stderr_text(self) -> str:
        if self.process.stderr is None:
            return ""
        return self.process.stderr.read().decode("utf-8", errors="replace").strip()


class FfmpegRtmpPublisher:
    """Publish annotated BGR frames as a low-latency H.264 RTMP stream."""

    def __init__(
        self,
        *,
        ffmpeg: str,
        publish_url: str,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        self.publish_url = publish_url
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = 0
        keyframe_interval = max(1, round(fps * 2))
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-re",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(keyframe_interval),
            "-keyint_min",
            str(keyframe_interval),
            "-sc_threshold",
            "0",
            "-bf",
            "0",
            "-f",
            "flv",
            publish_url,
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
            raise RuntimeError("RTMP publisher pipes were not created")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError("RTMP publisher frame dimensions changed")
        if frame.dtype != np.uint8:
            raise ValueError("RTMP publisher frame must use uint8 pixels")
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            error = self._stderr_text()
            raise RuntimeError(
                f"RTMP publisher stopped while writing: {error or 'no details'}"
            ) from exc
        self.frames += 1

    def close(self) -> None:
        if self.process.poll() is not None:
            error = self._stderr_text()
            if self.process.returncode != 0:
                raise RuntimeError(
                    f"RTMP publisher exited with code {self.process.returncode}: "
                    f"{error or 'no details'}"
                )
            return
        assert self.process.stdin is not None
        self.process.stdin.close()
        try:
            return_code = self.process.wait(timeout=15)
        except subprocess.TimeoutExpired as exc:
            self.process.kill()
            self.process.wait(timeout=5)
            raise RuntimeError("RTMP publisher did not stop in 15 seconds") from exc
        error = self._stderr_text()
        if return_code != 0:
            raise RuntimeError(
                f"RTMP publisher exited with code {return_code}: "
                f"{error or 'no details'}"
            )

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

    def _stderr_text(self) -> str:
        if self.process.stderr is None:
            return ""
        return self.process.stderr.read().decode("utf-8", errors="replace").strip()


def status_payload(
    stats: dict[str, Any],
    *,
    state: str,
    output_root: Path,
    writer: FfmpegSegmentWriter | None,
    last_error: str = "",
) -> dict[str, Any]:
    free_bytes = shutil.disk_usage(output_root).free
    payload = {
        **stats,
        "state": state,
        "updated_at": utc_now(),
        "free_disk_bytes": free_bytes,
        "free_disk_gb": round(free_bytes / 1_000_000_000, 3),
        "last_error": last_error,
        "current_segment": str(writer.destination) if writer is not None else None,
        "current_segment_frames": writer.frames if writer is not None else 0,
        "current_segment_target_frames": (
            writer.target_frames if writer is not None else 0
        ),
    }
    return payload


def save_status(
    path: Path,
    stats: dict[str, Any],
    *,
    state: str,
    output_root: Path,
    writer: FfmpegSegmentWriter | None,
    last_error: str = "",
) -> None:
    atomic_write_json(
        path,
        status_payload(
            stats,
            state=state,
            output_root=output_root,
            writer=writer,
            last_error=last_error,
        ),
    )


def wait_for_free_space(
    *,
    output_root: Path,
    minimum_free_bytes: int,
    recheck_seconds: float,
    stop_event: threading.Event,
    status_path: Path,
    stats: dict[str, Any],
) -> bool:
    while not stop_event.is_set():
        free_bytes = shutil.disk_usage(output_root).free
        if free_bytes >= minimum_free_bytes:
            return True
        save_status(
            status_path,
            stats,
            state="paused_low_disk",
            output_root=output_root,
            writer=None,
            last_error=(
                f"free disk {free_bytes / 1_000_000_000:.2f} GB is below "
                f"the {minimum_free_bytes / 1_000_000_000:.2f} GB guard"
            ),
        )
        log_event(
            "low_disk_pause",
            free_disk_gb=round(free_bytes / 1_000_000_000, 3),
            minimum_free_gb=round(minimum_free_bytes / 1_000_000_000, 3),
        )
        stop_event.wait(recheck_seconds)
    return False


def publish_segment(
    writer: FfmpegSegmentWriter,
    *,
    manifest_path: Path,
    reason: str,
    source_id: str,
    detection_confidence: float,
    pose_enabled: bool,
    segment_counts: dict[str, int],
) -> dict[str, Any]:
    record = writer.close()
    record.update(
        {
            "schema_version": 1,
            "source_id": source_id,
            "close_reason": reason,
            "detection_confidence": detection_confidence,
            "pose_enabled": pose_enabled,
            **segment_counts,
        }
    )
    append_jsonl(manifest_path, record)
    log_event("segment_published", **record)
    return record


def run(args: argparse.Namespace) -> int:
    if args.no_recording and not args.publish_url:
        raise ValueError("--no-recording requires --publish-url")
    recording_enabled = not args.no_recording
    publish_enabled = bool(args.publish_url)
    output_root = args.output_root.resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    manifest_path = output_root / "segments.jsonl"
    stop_event = threading.Event()

    def request_stop(signum: int, _frame: Any) -> None:
        log_event("stop_requested", signal=signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    models_config = load_yaml(args.models_config.resolve(strict=True))
    base_pipeline_config = load_yaml(args.pipeline_config.resolve(strict=True))
    pipeline_config = effective_pipeline_config(
        base_pipeline_config,
        source_id=args.source_id,
        source=args.source,
        detection_confidence=args.detection_confidence,
    )
    atomic_write_text(
        output_root / "effective_pipeline.yaml",
        yaml.safe_dump(pipeline_config, sort_keys=False, allow_unicode=True),
    )

    stats: dict[str, Any] = {
        "schema_version": 1,
        "pid": os.getpid(),
        "source_id": args.source_id,
        "source": args.source,
        "output_root": str(output_root),
        "started_at": utc_now(),
        "detection_confidence": args.detection_confidence,
        "recording_enabled": recording_enabled,
        "publish_enabled": publish_enabled,
        "segment_seconds": args.segment_seconds,
        "minimum_free_gb": args.min_free_gb,
        "pose_enabled": None,
        "posture_enabled": None,
        "inference_interval": args.inference_interval,
        "total_frames": 0,
        "total_inference_frames": 0,
        "total_detections": 0,
        "total_cat_instances": 0,
        "total_pose_instances": 0,
        "completed_segments": 0,
        "short_segments": 0,
        "reconnects": 0,
        "stream_sessions": 0,
        "last_frame_at": None,
        "last_inference_at": None,
        "last_segment": None,
        "current_cat_instances": 0,
        "current_pose_instances": 0,
        "current_eating_state": "",
        "current_postures": [],
        "last_pose_error_code": "",
    }
    save_status(
        status_path,
        stats,
        state="loading_models",
        output_root=output_root,
        writer=None,
    )
    pipeline = RealtimePipeline.from_configs(
        models_config=models_config,
        pipeline_config=pipeline_config,
        output_dir=output_root / "events",
        device=args.device,
        show=False,
    )
    stats["pose_enabled"] = pipeline.config.pose_enabled
    stats["posture_enabled"] = (
        pipeline.config.pose_enabled and pipeline.config.posture.enabled
    )
    minimum_free_bytes = int(args.min_free_gb * 1_000_000_000)
    service_started = time.monotonic()
    writer: FfmpegSegmentWriter | None = None
    exit_reason = "shutdown"
    last_error = ""
    log_event(
        "recorder_started",
        source_id=args.source_id,
        detection_confidence=args.detection_confidence,
        segment_seconds=args.segment_seconds,
        pose_enabled=pipeline.config.pose_enabled,
        inference_interval=args.inference_interval,
        recording_enabled=recording_enabled,
        publish_enabled=publish_enabled,
        output_root=str(output_root),
    )

    try:
        while not stop_event.is_set():
            if args.max_frames is not None and stats["total_frames"] >= args.max_frames:
                exit_reason = "max_frames"
                break
            save_status(
                status_path,
                stats,
                state="connecting",
                output_root=output_root,
                writer=writer,
                last_error=last_error,
            )
            capture: cv2.VideoCapture | None = None
            publisher: FfmpegRtmpPublisher | None = None
            last_timestamp: float | None = None
            session_reason = "source_disconnect"
            try:
                capture, frame, reported_fps = open_capture(
                    args.source,
                    open_timeout_seconds=args.open_timeout_seconds,
                    read_timeout_seconds=args.read_timeout_seconds,
                )
                fps = (
                    reported_fps
                    if math.isfinite(reported_fps) and reported_fps > 0
                    else args.fallback_fps
                )
                height, width = frame.shape[:2]
                target_frames = max(1, round(args.segment_seconds * fps))
                stats["stream_sessions"] += 1
                stats["stream_fps"] = fps
                stats["stream_width"] = width
                stats["stream_height"] = height
                stats["segment_target_frames"] = target_frames
                last_error = ""
                log_event(
                    "stream_connected",
                    source_id=args.source_id,
                    session=stats["stream_sessions"],
                    fps=fps,
                    width=width,
                    height=height,
                    segment_target_frames=target_frames,
                )
                segment_counts = {
                    "detections": 0,
                    "cat_instances": 0,
                    "pose_instances": 0,
                }
                analysis: Any | None = None
                session_frame_index = 0
                if args.publish_url:
                    publisher = FfmpegRtmpPublisher(
                        ffmpeg=args.ffmpeg,
                        publish_url=args.publish_url,
                        width=width,
                        height=height,
                        fps=fps,
                    )
                    log_event(
                        "publisher_started",
                        source_id=args.source_id,
                        width=width,
                        height=height,
                        fps=fps,
                    )

                while not stop_event.is_set():
                    if (
                        args.max_frames is not None
                        and stats["total_frames"] >= args.max_frames
                    ):
                        session_reason = "max_frames"
                        exit_reason = "max_frames"
                        stop_event.set()
                        break
                    if recording_enabled and writer is None:
                        if not wait_for_free_space(
                            output_root=output_root,
                            minimum_free_bytes=minimum_free_bytes,
                            recheck_seconds=args.disk_recheck_seconds,
                            stop_event=stop_event,
                            status_path=status_path,
                            stats=stats,
                        ):
                            session_reason = "shutdown"
                            break
                        writer = FfmpegSegmentWriter(
                            ffmpeg=args.ffmpeg,
                            video_root=output_root / "videos",
                            source_id=args.source_id,
                            width=width,
                            height=height,
                            fps=fps,
                            target_frames=target_frames,
                        )
                        segment_counts = {
                            "detections": 0,
                            "cat_instances": 0,
                            "pose_instances": 0,
                        }
                        log_event(
                            "segment_started",
                            path=str(writer.destination),
                            target_frames=target_frames,
                            fps=fps,
                        )

                    timestamp_seconds = time.monotonic() - service_started
                    last_timestamp = timestamp_seconds
                    inference_frame = inference_due(
                        session_frame_index,
                        args.inference_interval,
                        has_cached_analysis=analysis is not None,
                    )
                    if inference_frame:
                        analysis = pipeline.process_frame(
                            args.source_id,
                            timestamp_seconds,
                            frame,
                        )
                        stats["total_inference_frames"] += 1
                        stats["last_inference_at"] = utc_now()
                    assert analysis is not None
                    cat_pose_labels = {
                        cat.detection: (
                            f"conf={cat.detection.confidence:.2f} "
                            + cat_pose_label(
                                cat,
                                pipeline.config.pose_keypoint_confidence,
                            )
                        )
                        for cat in analysis.cats
                    }
                    rendered = render_frame(
                        frame,
                        analysis.detections,
                        analysis.cat_decisions,
                        analysis.poses,
                        source_id=args.source_id,
                        frame_index=stats["total_frames"],
                        timestamp_seconds=timestamp_seconds,
                        eating_state=analysis.eating_state,
                        cat_classes=pipeline.config.cat_classes,
                        feeding_classes=pipeline.config.feeding_classes,
                        association_max_distance_ratio=(
                            pipeline.config.feeding_association_max_distance_ratio
                        ),
                        association_ambiguity_margin=(
                            pipeline.config.feeding_association_ambiguity_margin
                        ),
                        pose_keypoint_confidence=(
                            pipeline.config.pose_keypoint_confidence
                        ),
                        cat_pose_labels=cat_pose_labels,
                    )
                    if writer is not None:
                        writer.write(rendered)
                    if publisher is not None:
                        publisher.write(rendered)
                    detections = len(analysis.detections)
                    cats = len(analysis.cats)
                    poses = len(analysis.poses)
                    stats["total_frames"] += 1
                    stats["total_detections"] += detections
                    stats["total_cat_instances"] += cats
                    stats["total_pose_instances"] += poses
                    stats["last_frame_at"] = utc_now()
                    stats["current_analysis_age_frames"] = (
                        session_frame_index % args.inference_interval
                    )
                    stats["current_cat_instances"] = cats
                    stats["current_pose_instances"] = poses
                    stats["current_eating_state"] = analysis.eating_state
                    stats["current_postures"] = sorted(
                        {
                            posture
                            for cat in analysis.cats
                            if (
                                posture := (
                                    getattr(cat, "posture", "")
                                    or getattr(cat, "observed_posture", "")
                                )
                            )
                        }
                    )
                    stats["last_pose_error_code"] = analysis.pose_error_code
                    segment_counts["detections"] += detections
                    segment_counts["cat_instances"] += cats
                    segment_counts["pose_instances"] += poses

                    if writer is not None and writer.frames >= writer.target_frames:
                        record = publish_segment(
                            writer,
                            manifest_path=manifest_path,
                            reason="full_segment",
                            source_id=args.source_id,
                            detection_confidence=args.detection_confidence,
                            pose_enabled=pipeline.config.pose_enabled,
                            segment_counts=segment_counts,
                        )
                        stats["completed_segments"] += 1
                        stats["last_segment"] = record["path"]
                        writer = None
                        pipeline.recorder.flush_csv()

                    if stats["total_frames"] % args.progress_every == 0:
                        save_status(
                            status_path,
                            stats,
                            state=(
                                "recording_and_streaming"
                                if publisher is not None
                                else "recording"
                            )
                            if recording_enabled
                            else "streaming",
                            output_root=output_root,
                            writer=writer,
                        )
                        log_event(
                            "progress",
                            total_frames=stats["total_frames"],
                            total_inference_frames=stats["total_inference_frames"],
                            inference_interval=args.inference_interval,
                            completed_segments=stats["completed_segments"],
                            current_segment_frames=(writer.frames if writer else 0),
                            publish_enabled=publish_enabled,
                        )

                    ok, next_frame = capture.read()
                    if not ok or next_frame is None:
                        session_reason = "source_disconnect"
                        last_error = "stream read failed"
                        break
                    if next_frame.shape[:2] != (height, width):
                        session_reason = "stream_dimensions_changed"
                        last_error = (
                            f"stream dimensions changed from {width}x{height} to "
                            f"{next_frame.shape[1]}x{next_frame.shape[0]}"
                        )
                        break
                    frame = next_frame
                    session_frame_index += 1
            except Exception as exc:
                session_reason = "processing_error"
                last_error = f"{type(exc).__name__}: {exc}"
                log_event("session_error", error=last_error)
            finally:
                if capture is not None:
                    capture.release()
                if publisher is not None:
                    try:
                        publisher.close()
                    except Exception as exc:
                        publisher.abort()
                        if not stop_event.is_set():
                            last_error = f"{type(exc).__name__}: {exc}"
                            log_event("publisher_stop_error", error=last_error)
                    finally:
                        publisher = None
                if last_timestamp is not None:
                    try:
                        pipeline_finish_reason = {
                            "source_disconnect": "disconnect",
                            "stream_dimensions_changed": "disconnect",
                            "processing_error": "error",
                            "max_frames": "eof",
                            "shutdown": "manual_stop",
                        }.get(session_reason, "error")
                        pipeline.finish_source(
                            args.source_id,
                            last_timestamp,
                            pipeline_finish_reason,
                        )
                    except Exception as exc:
                        log_event(
                            "finish_source_error",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                if writer is not None:
                    try:
                        record = publish_segment(
                            writer,
                            manifest_path=manifest_path,
                            reason=session_reason,
                            source_id=args.source_id,
                            detection_confidence=args.detection_confidence,
                            pose_enabled=pipeline.config.pose_enabled,
                            segment_counts=segment_counts,
                        )
                        stats["short_segments"] += int(
                            not record["complete_duration"]
                        )
                        stats["completed_segments"] += int(
                            record["complete_duration"]
                        )
                        stats["last_segment"] = record["path"]
                    except Exception as exc:
                        writer.abort()
                        last_error = f"{type(exc).__name__}: {exc}"
                        log_event("segment_publish_error", error=last_error)
                    finally:
                        writer = None
                pipeline.recorder.flush_csv()

            if stop_event.is_set():
                break
            stats["reconnects"] += 1
            save_status(
                status_path,
                stats,
                state="reconnecting",
                output_root=output_root,
                writer=None,
                last_error=last_error,
            )
            log_event(
                "reconnect_wait",
                reconnects=stats["reconnects"],
                delay_seconds=args.reconnect_delay_seconds,
                reason=session_reason,
                error=last_error,
            )
            stop_event.wait(args.reconnect_delay_seconds)
    finally:
        if writer is not None:
            writer.abort()
        pipeline.recorder.flush_csv()
        stats["stopped_at"] = utc_now()
        stats["stop_reason"] = exit_reason
        save_status(
            status_path,
            stats,
            state="stopped",
            output_root=output_root,
            writer=None,
            last_error=last_error,
        )
        log_event(
            "recorder_stopped",
            reason=exit_reason,
            total_frames=stats["total_frames"],
            total_inference_frames=stats["total_inference_frames"],
            inference_interval=args.inference_interval,
            completed_segments=stats["completed_segments"],
            short_segments=stats["short_segments"],
        )
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        log_event("fatal_error", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
