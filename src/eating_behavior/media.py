"""Video source resolution and frame iteration.

uri/stream_url is used only for opening video. source_id/camera_id is the stable
business key passed to identity routing, event output, and throttling.
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

import cv2
import numpy as np


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
MAX_SOURCE_ID_LENGTH = 128
_URL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


@dataclass(frozen=True)
class SourceSpec:
    """One video source in a worker-local snapshot."""

    source_id: str
    # A stream URI may contain user-info or bearer tokens. It must never be
    # included in the dataclass representation used by logs and tracebacks.
    uri: str = field(repr=False)

    def __post_init__(self) -> None:
        source_id = normalize_source_id(self.source_id)
        uri = str(self.uri).strip()
        if not source_id:
            raise ValueError("source_id/camera_id must not be empty")
        if not uri:
            raise ValueError(f"source URI must not be empty for source_id: {source_id}")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "uri", uri)


@dataclass(frozen=True)
class FramePacket:
    """One decoded frame.

    source is always the stable source_id. stream_url is kept only for optional
    diagnostics and should not be used as an identity key.
    """

    source: str
    frame_index: int
    timestamp_seconds: float
    frame: np.ndarray = field(repr=False)
    stream_url: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        source_id = normalize_source_id(self.source)
        if not source_id:
            raise ValueError("FramePacket.source must be a non-empty source_id")
        object.__setattr__(self, "source", source_id)


def _short_source_hash(value: str) -> str:
    """Return a stable, non-reversible label suitable for logs and IDs."""
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()[:16]


def redact_uri(uri: Any) -> str:
    """Return a credential-safe source label for diagnostics.

    Network output contains only the scheme and host. User-info, path, query,
    and fragment data are deliberately omitted because credentials can appear
    in any of them. Local paths are represented only by a stable short hash.
    """
    value = "" if uri is None else str(uri).strip()
    if not value:
        return "<empty-source>"
    if _URL_PATTERN.match(value):
        try:
            parsed = urlsplit(value)
            host = parsed.hostname or "<unknown-host>"
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            try:
                port = f":{parsed.port}" if parsed.port is not None else ""
            except ValueError:
                port = ""
            return f"{parsed.scheme.lower()}://{host}{port}/<redacted>"
        except ValueError:
            return "<network-source:redacted>"
    if value.isdigit():
        return f"<camera:{value}>"
    return f"<local-source:{_short_source_hash(value)}>"


def local_source_id(uri: str) -> str:
    """Create a stable ID for a local debug source without exposing its path."""
    value = str(uri).strip()
    if value.isdigit():
        canonical = f"camera:{value}"
    else:
        path = Path(value).expanduser().resolve(strict=False)
        canonical = str(path).casefold()
    return f"local_{_short_source_hash(canonical)}"


def is_network_source(source: str) -> bool:
    """Return whether a source is a supported network stream URL."""
    return str(source).strip().lower().startswith(STREAM_PREFIXES)


def is_stream_source(source: str) -> bool:
    """Return whether a source is a camera index or network stream."""
    value = source.strip().lower()
    return value.isdigit() or is_network_source(value)


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
            raise FileNotFoundError(f"No video files found under: {redact_uri(path)}")
        return videos
    raise FileNotFoundError(f"Source does not exist and is not a stream: {redact_uri(source)}")


def normalize_source_id(value: Any) -> str:
    """Normalize and validate a source_id/camera_id business key.

    Empty values are returned as an empty string so callers can provide a
    context-specific "missing source_id" error. Non-empty IDs must be safe to
    place in events and logs and must never be stream URLs.
    """
    raw_source_id = "" if value is None else str(value)
    if any(unicodedata.category(char) == "Cc" for char in raw_source_id):
        raise ValueError("source_id/camera_id must not contain control characters")
    source_id = raw_source_id.strip()
    if not source_id:
        return ""
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id/camera_id must be at most {MAX_SOURCE_ID_LENGTH} characters")
    if _URL_PATTERN.match(source_id):
        raise ValueError("source_id/camera_id must be a stable business ID, not a URL")
    return source_id


def source_id_list(values: Any) -> tuple[str, ...]:
    """Parse worker.source_ids or CLI --source-id values."""
    if values is None:
        return ()
    if isinstance(values, str):
        normalized = (normalize_source_id(item) for item in values.split(","))
        return tuple(item for item in normalized if item)
    if isinstance(values, (list, tuple, set)):
        return tuple(normalize_source_id(item) for item in values if normalize_source_id(item))
    raise TypeError(f"source ids must be a string or list of strings, got {type(values).__name__}")


def configured_source_specs(
    sources_config: Any,
    selected_source_ids: tuple[str, ...] | list[str] | set[str] | None = None,
) -> list[SourceSpec]:
    """Build SourceSpec objects from worker-local source config."""
    if sources_config is not None and not isinstance(sources_config, (list, tuple)):
        raise ValueError("sources must be a list of source mappings")
    selected = set(source_id_list(selected_source_ids))
    specs: list[SourceSpec] = []
    seen: set[str] = set()
    for item in sources_config or []:
        if isinstance(item, str):
            uri = item.strip()
            if _URL_PATTERN.match(uri):
                raise ValueError(
                    "Network sources must use a mapping with separate source_id/camera_id and stream_url/url"
                )
            source_id = local_source_id(uri)
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
    normalized_source_id = normalize_source_id(source_id)
    if any(is_network_source(uri) for uri in uris) and not normalized_source_id:
        raise ValueError("Network stream --source requires an explicit stable --source-id")
    return [
        SourceSpec(source_id=normalized_source_id or local_source_id(uri), uri=uri)
        for uri in uris
    ]


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
    if isinstance(source, SourceSpec):
        spec = source
    else:
        specs = resolve_source_specs(str(source))
        if len(specs) != 1:
            raise ValueError("iter_frames requires one source; pass a SourceSpec for resolved source sets")
        spec = specs[0]
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
