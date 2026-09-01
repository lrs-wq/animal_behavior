#!/usr/bin/env python3
"""Extract one JPEG for every N source frames from a video tree.

The output for ``<source_root>/<group>/<name>.<ext>`` is written to
``<output_root>/<group>/<name>__<ext>/frame_<source_frame>.jpg``.
Each video is first written to a private work directory and published with an
atomic rename, so an interrupted run can be resumed safely.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
MANIFEST_NAME = "_extraction_manifest.json"
MARKER_NAME = ".complete.json"


@dataclass(frozen=True)
class VideoInfo:
    source: str
    relative_path: str
    destination: str
    size_bytes: int
    mtime_ns: int
    width: int
    height: int
    fps: float
    duration_seconds: float
    estimated_frames: int
    estimated_images: int
    frame_count_basis: str
    sample_jpeg_bytes: int | None = None


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_fraction(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def positive_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed > 0 else 0.0


def destination_for(source: Path, source_root: Path, output_root: Path) -> Path:
    relative = source.relative_to(source_root)
    extension = relative.suffix.lower().lstrip(".") or "video"
    return output_root / relative.parent / f"{relative.stem}__{extension}"


def list_videos(source_root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )


def run_json_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise RuntimeError(error)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON output: {exc}") from exc


def probe_video(
    source: Path,
    source_root: Path,
    output_root: Path,
    interval: int,
    ffprobe: str,
    exact_frame_count: bool,
) -> VideoInfo:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if exact_frame_count:
        command.append("-count_frames")
    command.extend(
        [
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration:format=duration",
            "-of",
            "json",
            str(source),
        ]
    )
    payload = run_json_command(command)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("no video stream")
    stream = streams[0]
    width = positive_int(stream.get("width"))
    height = positive_int(stream.get("height"))
    fps = parse_fraction(stream.get("avg_frame_rate")) or parse_fraction(
        stream.get("r_frame_rate")
    )
    duration = positive_float(stream.get("duration")) or positive_float(
        (payload.get("format") or {}).get("duration")
    )
    read_frames = positive_int(stream.get("nb_read_frames"))
    metadata_frames = positive_int(stream.get("nb_frames"))
    if exact_frame_count and read_frames:
        estimated_frames = read_frames
        basis = "stream.nb_read_frames"
    elif metadata_frames:
        estimated_frames = metadata_frames
        basis = "stream.nb_frames"
    elif duration and fps:
        estimated_frames = max(1, round(duration * fps))
        basis = "duration_x_fps"
    else:
        estimated_frames = 0
        basis = "unknown"
    estimated_images = (
        (estimated_frames + interval - 1) // interval if estimated_frames else 0
    )
    stat = source.stat()
    return VideoInfo(
        source=str(source),
        relative_path=source.relative_to(source_root).as_posix(),
        destination=str(destination_for(source, source_root, output_root)),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        width=width,
        height=height,
        fps=fps,
        duration_seconds=duration,
        estimated_frames=estimated_frames,
        estimated_images=estimated_images,
        frame_count_basis=basis,
    )


def sample_jpeg_size(info: VideoInfo, ffmpeg: str, qscale: int) -> int:
    seek_seconds = info.duration_seconds / 2 if info.duration_seconds else 0.0
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if seek_seconds:
        command.extend(["-ss", f"{seek_seconds:.6f}"])
    command.extend(
        [
            "-i",
            info.source,
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-q:v",
            str(qscale),
            "-c:v",
            "mjpeg",
            "-f",
            "image2pipe",
            "pipe:1",
        ]
    )
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(error or "could not generate sample JPEG")
    return len(completed.stdout)


def probe_all(
    videos: list[Path],
    source_root: Path,
    output_root: Path,
    interval: int,
    ffprobe: str,
    workers: int,
    exact_frame_count: bool,
) -> tuple[list[VideoInfo], list[dict[str, str]]]:
    infos: list[VideoInfo] = []
    errors: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers * 2, 16)) as pool:
        future_to_path = {
            pool.submit(
                probe_video,
                path,
                source_root,
                output_root,
                interval,
                ffprobe,
                exact_frame_count,
            ): path
            for path in videos
        }
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                infos.append(future.result())
            except Exception as exc:  # noqa: BLE001 - each bad file must be reported
                errors.append(
                    {
                        "source": path.relative_to(source_root).as_posix(),
                        "error": str(exc),
                    }
                )
    infos.sort(key=lambda info: info.relative_path)
    errors.sort(key=lambda error: error["source"])
    return infos, errors


def add_sample_estimates(
    infos: list[VideoInfo], ffmpeg: str, qscale: int, workers: int
) -> tuple[list[VideoInfo], list[dict[str, str]]]:
    sizes: dict[str, int] = {}
    errors: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, 8)) as pool:
        future_to_info = {
            pool.submit(sample_jpeg_size, info, ffmpeg, qscale): info for info in infos
        }
        for future in concurrent.futures.as_completed(future_to_info):
            info = future_to_info[future]
            try:
                sizes[info.relative_path] = future.result()
            except Exception as exc:  # noqa: BLE001
                errors.append({"source": info.relative_path, "error": str(exc)})
    enriched = [
        VideoInfo(**{**asdict(info), "sample_jpeg_bytes": sizes.get(info.relative_path)})
        for info in infos
    ]
    errors.sort(key=lambda error: error["source"])
    return enriched, errors


def human_bytes(value: int | float) -> str:
    size = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if abs(size) < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def plan_summary(infos: list[VideoInfo]) -> dict[str, Any]:
    resolution_counts: dict[str, int] = {}
    for info in infos:
        key = f"{info.width}x{info.height}"
        resolution_counts[key] = resolution_counts.get(key, 0) + 1
    sample_estimated_bytes = sum(
        (info.sample_jpeg_bytes or 0) * info.estimated_images for info in infos
    )
    sampled_videos = sum(info.sample_jpeg_bytes is not None for info in infos)
    return {
        "videos": len(infos),
        "source_bytes": sum(info.size_bytes for info in infos),
        "duration_seconds": sum(info.duration_seconds for info in infos),
        "estimated_frames": sum(info.estimated_frames for info in infos),
        "estimated_images": sum(info.estimated_images for info in infos),
        "unknown_frame_count_videos": sum(
            info.frame_count_basis == "unknown" for info in infos
        ),
        "sampled_videos": sampled_videos,
        "sample_estimated_output_bytes": sample_estimated_bytes
        if sampled_videos == len(infos)
        else None,
        "resolutions": dict(sorted(resolution_counts.items())),
    }


def print_plan(summary: dict[str, Any], output_root: Path) -> None:
    print(
        "PLAN "
        f"videos={summary['videos']} "
        f"duration_hours={summary['duration_seconds'] / 3600:.3f} "
        f"estimated_frames={summary['estimated_frames']} "
        f"estimated_images={summary['estimated_images']}",
        flush=True,
    )
    print(f"PLAN source_size={human_bytes(summary['source_bytes'])}", flush=True)
    if summary["sample_estimated_output_bytes"] is not None:
        estimate = summary["sample_estimated_output_bytes"]
        print(
            f"PLAN sampled_jpeg_output_estimate={human_bytes(estimate)} "
            "(one middle-frame sample per video)",
            flush=True,
        )
    print(f"PLAN resolutions={json.dumps(summary['resolutions'], ensure_ascii=False)}")
    disk_base = output_root if output_root.exists() else output_root.parent
    usage = shutil.disk_usage(disk_base)
    print(f"PLAN disk_free={human_bytes(usage.free)} at {disk_base}", flush=True)


def marker_matches(marker: dict[str, Any], task: dict[str, Any], jpg_count: int) -> bool:
    return (
        marker.get("source") == task["relative_path"]
        and marker.get("source_size_bytes") == task["size_bytes"]
        and marker.get("source_mtime_ns") == task["mtime_ns"]
        and marker.get("frame_interval") == task["interval"]
        and marker.get("jpeg_qscale") == task["qscale"]
        and marker.get("images_written") == jpg_count
        and jpg_count > 0
    )


def read_marker(marker_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def extract_one(task: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    source = Path(task["source"])
    destination = Path(task["destination"])
    marker_path = destination / MARKER_NAME
    if destination.exists():
        jpg_count = sum(1 for _ in destination.glob("frame_*.jpg"))
        marker = read_marker(marker_path)
        if marker and marker_matches(marker, task, jpg_count):
            return {
                "source": task["relative_path"],
                "destination": str(destination),
                "status": "skipped",
                "images_written": jpg_count,
                "output_bytes": marker.get("output_bytes", 0),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        return {
            "source": task["relative_path"],
            "destination": str(destination),
            "status": "conflict",
            "error": "destination exists without a matching completion marker",
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    digest = hashlib.sha256(task["relative_path"].encode("utf-8")).hexdigest()[:16]
    work_dir = Path(task["output_root"]) / ".work" / digest
    try:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=False)
        selected_pattern = work_dir / "selected_%08d.jpg"
        command = [
            task["ffmpeg"],
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-threads",
            str(task["threads_per_job"]),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            f"select=not(mod(n\\,{task['interval']}))",
            "-fps_mode",
            "vfr",
            "-q:v",
            str(task["qscale"]),
            "-start_number",
            "0",
            str(selected_pattern),
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            error = completed.stderr.strip() or f"ffmpeg exit code {completed.returncode}"
            raise RuntimeError(error)

        selected = sorted(work_dir.glob("selected_*.jpg"))
        if not selected:
            raise RuntimeError("ffmpeg produced no JPEG images")
        for sequence, old_path in enumerate(selected):
            frame_number = sequence * task["interval"]
            old_path.rename(work_dir / f"frame_{frame_number:08d}.jpg")

        images = sorted(work_dir.glob("frame_*.jpg"))
        output_bytes = 0
        for image_path in images:
            output_bytes += image_path.stat().st_size
            with image_path.open("rb") as handle:
                if handle.read(2) != b"\xff\xd8":
                    raise RuntimeError(f"invalid JPEG header: {image_path.name}")
                handle.seek(-2, os.SEEK_END)
                if handle.read(2) != b"\xff\xd9":
                    raise RuntimeError(f"invalid JPEG trailer: {image_path.name}")

        marker = {
            "schema_version": 1,
            "completed_at": now_iso(),
            "source": task["relative_path"],
            "source_absolute": str(source),
            "source_size_bytes": task["size_bytes"],
            "source_mtime_ns": task["mtime_ns"],
            "frame_interval": task["interval"],
            "first_source_frame": 0,
            "jpeg_qscale": task["qscale"],
            "images_written": len(images),
            "last_source_frame": (len(images) - 1) * task["interval"],
            "output_bytes": output_bytes,
            "ffmpeg_command": command,
        }
        (work_dir / MARKER_NAME).write_text(
            json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(work_dir, destination)
        return {
            "source": task["relative_path"],
            "destination": str(destination),
            "status": "processed",
            "images_written": len(images),
            "output_bytes": output_bytes,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001 - return a per-video failure to the parent
        return {
            "source": task["relative_path"],
            "destination": str(destination),
            "status": "failed",
            "error": str(exc),
            "work_dir": str(work_dir),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }


def task_for(info: VideoInfo, args: argparse.Namespace) -> dict[str, Any]:
    return {
        **asdict(info),
        "interval": args.frame_interval,
        "qscale": args.jpeg_qscale,
        "output_root": str(args.output_root),
        "ffmpeg": args.ffmpeg,
        "threads_per_job": args.threads_per_job,
    }


def write_manifest(
    output_root: Path,
    args: argparse.Namespace,
    summary: dict[str, Any],
    infos: list[VideoInfo],
    results: list[dict[str, Any]],
) -> None:
    payload = {
        "schema_version": 1,
        "updated_at": now_iso(),
        "source_root": str(args.source_root),
        "output_root": str(output_root),
        "frame_interval": args.frame_interval,
        "first_source_frame": 0,
        "jpeg_qscale": args.jpeg_qscale,
        "plan_summary": summary,
        "videos": [asdict(info) for info in infos],
        "results": sorted(results, key=lambda result: result["source"]),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{MANIFEST_NAME}.tmp-{os.getpid()}"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_root / MANIFEST_NAME)


def verify_one(task: dict[str, Any], decode_images: bool) -> dict[str, Any]:
    destination = Path(task["destination"])
    marker = read_marker(destination / MARKER_NAME)
    if not marker:
        return {"source": task["relative_path"], "error": "completion marker missing"}
    images = sorted(destination.glob("frame_*.jpg"))
    if not marker_matches(marker, task, len(images)):
        return {
            "source": task["relative_path"],
            "error": "completion marker or JPEG count does not match",
        }
    if (
        task["frame_count_basis"] == "stream.nb_read_frames"
        and len(images) != task["estimated_images"]
    ):
        return {
            "source": task["relative_path"],
            "error": (
                f"expected {task['estimated_images']} images from "
                f"{task['estimated_frames']} decoded source frames, found {len(images)}"
            ),
        }
    expected_names = [
        f"frame_{sequence * task['interval']:08d}.jpg" for sequence in range(len(images))
    ]
    actual_names = [path.name for path in images]
    if actual_names != expected_names:
        return {"source": task["relative_path"], "error": "frame filenames are not contiguous"}
    try:
        cv2 = None
        if decode_images:
            import cv2 as imported_cv2

            cv2 = imported_cv2
        for image_path in images:
            with image_path.open("rb") as handle:
                if handle.read(2) != b"\xff\xd8":
                    raise RuntimeError(f"bad JPEG header: {image_path.name}")
                handle.seek(-2, os.SEEK_END)
                if handle.read(2) != b"\xff\xd9":
                    raise RuntimeError(f"bad JPEG trailer: {image_path.name}")
            if cv2 is not None and cv2.imread(str(image_path)) is None:
                raise RuntimeError(f"OpenCV could not decode: {image_path.name}")
    except Exception as exc:  # noqa: BLE001
        return {"source": task["relative_path"], "error": str(exc)}
    return {
        "source": task["relative_path"],
        "images": len(images),
        "bytes": sum(path.stat().st_size for path in images),
    }


def verify_all(
    infos: list[VideoInfo], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers * 2, 16)) as pool:
        future_to_info = {
            pool.submit(verify_one, task_for(info, args), args.decode_verify): info
            for info in infos
        }
        for future in concurrent.futures.as_completed(future_to_info):
            result = future.result()
            if "error" in result:
                invalid.append(result)
            else:
                valid.append(result)
    valid.sort(key=lambda result: result["source"])
    invalid.sort(key=lambda result: result["source"])
    return valid, invalid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-interval", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-job", type=int, default=2)
    parser.add_argument("--jpeg-qscale", type=int, default=2)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--sample-estimate", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--decode-verify", action="store_true")
    parser.add_argument(
        "--exact-source-count",
        action="store_true",
        help="Decode/count every source frame during probing; intended for final verification.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N videos (useful for a pilot run).",
    )
    args = parser.parse_args()
    if args.frame_interval < 1:
        parser.error("--frame-interval must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.threads_per_job < 1:
        parser.error("--threads-per-job must be at least 1")
    if not 1 <= args.jpeg_qscale <= 31:
        parser.error("--jpeg-qscale must be between 1 and 31")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    return args


def main() -> int:
    args = parse_args()
    if not args.source_root.is_dir():
        print(f"ERROR source directory does not exist: {args.source_root}", file=sys.stderr)
        return 2
    if args.output_root == args.source_root or args.source_root in args.output_root.parents:
        print("ERROR output root must not be inside source root", file=sys.stderr)
        return 2
    if shutil.which(args.ffmpeg) is None or shutil.which(args.ffprobe) is None:
        print("ERROR ffmpeg and ffprobe must both be available", file=sys.stderr)
        return 2

    videos = list_videos(args.source_root)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print("ERROR no supported video files found", file=sys.stderr)
        return 2
    print(f"DISCOVERED videos={len(videos)}", flush=True)

    infos, probe_errors = probe_all(
        videos,
        args.source_root,
        args.output_root,
        args.frame_interval,
        args.ffprobe,
        args.workers,
        args.exact_source_count,
    )
    if probe_errors:
        for error in probe_errors:
            print(f"PROBE_FAILED {error['source']}: {error['error']}", file=sys.stderr)
        print(f"ERROR probe_failures={len(probe_errors)}", file=sys.stderr)
        return 1

    destinations = [info.destination for info in infos]
    if len(destinations) != len(set(destinations)):
        print("ERROR multiple sources map to the same output directory", file=sys.stderr)
        return 2

    sample_errors: list[dict[str, str]] = []
    if args.sample_estimate:
        infos, sample_errors = add_sample_estimates(
            infos, args.ffmpeg, args.jpeg_qscale, args.workers
        )
        for error in sample_errors:
            print(f"SAMPLE_FAILED {error['source']}: {error['error']}", file=sys.stderr)
    summary = plan_summary(infos)
    print_plan(summary, args.output_root)
    if args.plan_only:
        print(
            "PLAN_JSON " + json.dumps(summary, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        return 1 if sample_errors else 0

    if args.verify_only:
        valid, invalid = verify_all(infos, args)
        for error in invalid:
            print(f"VERIFY_FAILED {error['source']}: {error['error']}", file=sys.stderr)
        print(
            f"VERIFY_SUMMARY videos_ok={len(valid)} videos_failed={len(invalid)} "
            f"images={sum(item['images'] for item in valid)} "
            f"bytes={sum(item['bytes'] for item in valid)}",
            flush=True,
        )
        return 1 if invalid else 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / ".work").mkdir(parents=True, exist_ok=True)
    tasks = [task_for(info, args) for info in infos]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_to_task = {pool.submit(extract_one, task): task for task in tasks}
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_task):
            result = future.result()
            results.append(result)
            completed_count += 1
            status = result["status"].upper()
            detail = (
                f"images={result.get('images_written', 0)} "
                f"elapsed={result.get('elapsed_seconds', 0):.1f}s"
            )
            if "error" in result:
                detail += f" error={result['error']}"
            print(
                f"[{completed_count}/{len(tasks)}] {status} {result['source']} {detail}",
                flush=True,
            )

    write_manifest(args.output_root, args, summary, infos, results)
    try:
        (args.output_root / ".work").rmdir()
    except OSError:
        pass
    successful = [result for result in results if result["status"] in {"processed", "skipped"}]
    failed = [result for result in results if result["status"] not in {"processed", "skipped"}]
    print(
        f"EXTRACTION_SUMMARY videos_ok={len(successful)} videos_failed={len(failed)} "
        f"images={sum(result.get('images_written', 0) for result in successful)} "
        f"bytes={sum(result.get('output_bytes', 0) for result in successful)}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
