"""Announcement event recording.

The current project writes structured events to files. Actual notification
channels such as app push, speech, or WebSocket can consume these files later.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnnouncementEvent:
    """Structured event emitted by the realtime pipeline.

    source is the stable source_id/camera_id, not the raw stream URL.
    household_id is top-level so downstream systems can group by customer home.
    """

    event_type: str
    source: str
    timestamp_seconds: float
    cat_name: str
    coat_class: str
    confidence: float
    message_key: str
    household_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    created_at_unix: float = field(default_factory=time.time)


class AnnouncementRecorder:
    """Stream events to disk while retaining only a bounded diagnostic tail."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        save_jsonl: bool = True,
        save_csv: bool = True,
        max_recent_events: int = 1000,
    ) -> None:
        if max_recent_events < 0:
            raise ValueError("max_recent_events must be >= 0")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "announcement_events.jsonl"
        self.csv_path = self.output_dir / "announcement_events.csv"
        self.save_jsonl = save_jsonl
        self.save_csv = save_csv
        self._events: deque[AnnouncementEvent] = deque(maxlen=max_recent_events)
        self._lock = threading.Lock()

    @staticmethod
    def _csv_fieldnames() -> list[str]:
        return [
            "event_type",
            "source",
            "timestamp_seconds",
            "cat_name",
            "coat_class",
            "confidence",
            "message_key",
            "household_id",
            "details_json",
            "created_at_unix",
        ]

    @staticmethod
    def _spreadsheet_safe(value: Any) -> Any:
        """Prevent user-controlled text from becoming a spreadsheet formula."""
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        return value

    def _csv_row(self, event: AnnouncementEvent) -> dict[str, Any]:
        row = asdict(event)
        details = row.pop("details")
        row["details_json"] = json.dumps(details, ensure_ascii=False)
        return {key: self._spreadsheet_safe(value) for key, value in row.items()}

    def record(self, event: AnnouncementEvent) -> None:
        """Append one event to each enabled sink without unbounded buffering."""
        with self._lock:
            self._events.append(event)
            if self.save_jsonl:
                with self.jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
            if self.save_csv:
                needs_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
                with self.csv_path.open("a", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames())
                    if needs_header:
                        writer.writeheader()
                    writer.writerow(self._csv_row(event))

    def flush_csv(self) -> None:
        """Compatibility no-op: CSV rows are persisted by ``record``."""


class EventThrottler:
    """Simple per-key event rate limiter.

    Keys should include source_id so different customers do not suppress each
    other's events.
    """

    def __init__(self) -> None:
        self.last_event_time: dict[tuple[str, ...], float] = {}

    def allow(self, key: tuple[str, ...], timestamp_seconds: float, interval_seconds: float) -> bool:
        """Return whether an event is allowed at timestamp_seconds."""
        last = self.last_event_time.get(key)
        if last is not None and timestamp_seconds - last < interval_seconds:
            return False
        self.last_event_time[key] = timestamp_seconds
        return True

    def clear_source(self, source: str) -> None:
        """Drop media-time state for one source before a new stream session."""
        self.last_event_time = {
            key: value
            for key, value in self.last_event_time.items()
            if len(key) < 2 or key[1] != source
        }
