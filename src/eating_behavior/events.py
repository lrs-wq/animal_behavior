from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnnouncementEvent:
    event_type: str
    source: str
    timestamp_seconds: float
    cat_name: str
    coat_class: str
    confidence: float
    message_key: str
    details: dict[str, Any] = field(default_factory=dict)
    created_at_unix: float = field(default_factory=time.time)


class AnnouncementRecorder:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "announcement_events.jsonl"
        self.csv_path = self.output_dir / "announcement_events.csv"
        self._events: list[AnnouncementEvent] = []

    def record(self, event: AnnouncementEvent) -> None:
        self._events.append(event)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def flush_csv(self) -> None:
        fieldnames = [
            "event_type",
            "source",
            "timestamp_seconds",
            "cat_name",
            "coat_class",
            "confidence",
            "message_key",
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
    def __init__(self) -> None:
        self.last_event_time: dict[tuple[str, str], float] = {}

    def allow(self, key: tuple[str, str], timestamp_seconds: float, interval_seconds: float) -> bool:
        last = self.last_event_time.get(key)
        if last is not None and timestamp_seconds - last < interval_seconds:
            return False
        self.last_event_time[key] = timestamp_seconds
        return True

