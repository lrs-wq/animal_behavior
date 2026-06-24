"""Announcement event recording.

The current project writes structured events to files. Actual notification
channels such as app push, speech, or WebSocket can consume these files later.
"""

from __future__ import annotations

import csv
import json
import time
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
    """Append JSONL immediately and flush CSV at the end of a run."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "announcement_events.jsonl"
        self.csv_path = self.output_dir / "announcement_events.csv"
        self._events: list[AnnouncementEvent] = []

    def record(self, event: AnnouncementEvent) -> None:
        """Record one event in memory and append it to JSONL."""
        self._events.append(event)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def flush_csv(self) -> None:
        """Write all events seen by this process to a CSV file."""
        fieldnames = [
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
        with self.csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for event in self._events:
                row = asdict(event)
                details = row.pop("details")
                row["details_json"] = json.dumps(details, ensure_ascii=False)
                writer.writerow(row)


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
