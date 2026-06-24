"""Video source resolution and frame iteration.

uri/stream_url is used only for opening video. source_id/camera_id is the stable
business key passed to identity routing, event output, and throttling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

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
class SourceSpec:
    """One video source in a worker-local snapshot."""

    source_id: str
    uri: str


@dataclass(frozen=True)
class FramePacket:
    """One decoded frame.

    source is always the stable source_id. stream_url is kept only for optional
    diagnostics and should not be used as an identity key.
    """

    source: str
    frame_index: int
    timestamp_seconds: float
    frame: object
    stream_url: str = ""


def is_stream_source(source: str) -> bool:
    """Return whether a source is a camera index or network stream."""
    value = source.strip().lower()
    return value.isdigit() or value.startswith(STREAM_PREFIXES)


def resolve_sources(source: str) -> list[str]:
    """Resolve CLI --source into one or more concrete video URIs."""
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


def normalize_source_id(value: Any) -> str:
    """Normalize source_id/camera_id values."""
    return str(value).strip()


def source_id_list(values: Any) -> tuple[str, ...]:
    """Parse worker.source_ids or CLI --source-id values."""
    if values is None:
        return ()
    if isinstance(values, str):
        return tuple(item.strip() for item in values.split(",") if item.strip())
    if isinstance(values, (list, tuple, set)):
        return tuple(normalize_source_id(item) for item in values if normalize_source_id(item))
    raise TypeError(f"source ids must be a string or list of strings, got {type(values).__name__}")


def configured_source_specs(
    sources_config: Any,
    selected_source_ids: tuple[str, ...] | list[str] | set[str] | None = None,
) -> list[SourceSpec]:
    """Build SourceSpec objects from worker-local source config."""
    selected = set(source_id_list(selected_source_ids))
    specs: list[SourceSpec] = []
    seen: set[str] = set()
    for item in sources_config or []:
        if isinstance(item, str):
            source_id = normalize_source_id(item)
            uri = item
        elif isinstance(item, dict):
            # Accept several field names to ease DB/config-service integration.
            source_id = normalize_source_id(
                item.get("source_id")
                or item.get("camera_id")
                or item.get("id")
                or item.get("name")
                or ""
            )
            uri = str(
                item.get("stream_url")
                or item.get("url")
                or item.get("uri")
                or item.get("source")
                or ""
            ).strip()
        else:
            raise TypeError(f"source config items must be strings or mappings, got {type(item).__name__}")
        if not source_id:
            raise ValueError("source config item is missing source_id/camera_id")
        if not uri:
            raise ValueError(f"source config item is missing stream_url/url: {source_id}")
        if source_id in seen:
            raise ValueError(f"Duplicate configured source_id: {source_id}")
        seen.add(source_id)
        if selected and source_id not in selected:
            continue
        specs.append(SourceSpec(source_id=source_id, uri=uri))
    if selected:
        found = {spec.source_id for spec in specs}
        missing = sorted(selected - found)
        if missing:
            raise ValueError(f"Configured source_id not found: {', '.join(missing)}")
    return specs


def resolve_source_specs(source: str, source_id: str | None = None) -> list[SourceSpec]:
    """Convert debug --source input into SourceSpec objects."""
    uris = resolve_sources(source)
    if source_id is not None and len(uris) > 1:
        raise ValueError("--source-id can only be paired with --source when it resolves to a single source")
    return [SourceSpec(source_id=normalize_source_id(source_id) if source_id else uri, uri=uri) for uri in uris]


def open_capture(source: str) -> cv2.VideoCapture:
    """Open an OpenCV capture; digit-only strings are local camera indexes."""
    if source.strip().isdigit():
        return cv2.VideoCapture(int(source.strip()))
    return cv2.VideoCapture(source)


def iter_frames(source: str | SourceSpec, vid_stride: int = 1, max_frames: int | None = None) -> Iterator[FramePacket]:
    """Iterate frames from a video source.

    FramePacket.source is the stable source_id even when the capture is opened
    from a raw RTSP/HTTP URL.
    """
    spec = source if isinstance(source, SourceSpec) else SourceSpec(source_id=str(source), uri=str(source))
    capture = open_capture(spec.uri)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {spec.source_id}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    live = is_stream_source(spec.uri)
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
            # Live streams often do not expose reliable FPS, so use monotonic time.
            timestamp = time.monotonic() - start_time if live or fps <= 0 else frame_index / fps
            yield FramePacket(
                source=spec.source_id,
                frame_index=frame_index,
                timestamp_seconds=float(timestamp),
                frame=frame,
                stream_url=spec.uri,
            )
            yielded += 1
            frame_index += 1
            if max_frames is not None and yielded >= max_frames:
                break
    finally:
        capture.release()
