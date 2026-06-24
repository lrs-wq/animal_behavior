from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".wmv",
    ".flv",
    ".webm",
}
STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")


@dataclass(frozen=True)
class FramePacket:
    source: str
    frame_index: int
    timestamp_seconds: float
    frame: object


def is_stream_source(source: str) -> bool:
    value = source.strip().lower()
    return value.isdigit() or value.startswith(STREAM_PREFIXES)


def resolve_sources(source: str) -> list[str]:
    if is_stream_source(source):
        return [source]
    path = Path(source)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        videos = [
            str(item)
            for item in sorted(path.rglob("*"))
            if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if not videos:
            raise FileNotFoundError(f"No video files found under: {path}")
        return videos
    raise FileNotFoundError(f"Source does not exist and is not a stream: {source}")


def open_capture(source: str) -> cv2.VideoCapture:
    if source.strip().isdigit():
        return cv2.VideoCapture(int(source.strip()))
    return cv2.VideoCapture(source)


def iter_frames(source: str, vid_stride: int = 1, max_frames: int | None = None) -> Iterator[FramePacket]:
    capture = open_capture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    live = is_stream_source(source)
    start_time = time.monotonic()
    frame_index = 0
    yielded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % max(1, vid_stride) != 0:
                frame_index += 1
                continue
            timestamp = time.monotonic() - start_time if live or fps <= 0 else frame_index / fps
            yield FramePacket(
                source=source,
                frame_index=frame_index,
                timestamp_seconds=float(timestamp),
                frame=frame,
            )
            yielded += 1
            frame_index += 1
            if max_frames is not None and yielded >= max_frames:
                break
    finally:
        capture.release()

