"""Announcement event recording.

The current project writes structured events to files. Actual notification
channels such as app push, speech, or WebSocket can consume these files later.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_ROLLED_BACK_ATTRIBUTE = "_announcement_sink_write_rolled_back"


@contextmanager
def _exclusive_path_lock(path: Path, timeout_seconds: float = 30.0):
    """Hold a small cross-process lock file for one append transaction."""
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            deadline = time.monotonic() + timeout_seconds
            while True:
                lock_file.seek(0)
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out acquiring event sink lock: {path.name}"
                        )
                    time.sleep(0.01)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            deadline = time.monotonic() + timeout_seconds
            while True:
                try:
                    fcntl.flock(
                        lock_file.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out acquiring event sink lock: {path.name}"
                        )
                    time.sleep(0.01)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


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
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise ValueError("record_id must be a non-blank string")
        self.record_id = self.record_id.strip()


@dataclass
class _DeliveryState:
    """Per-record progress across the bounded diagnostic and file sinks."""

    fingerprint: str
    serialized_event: str
    csv_row: tuple[tuple[str, Any], ...]
    recent_event: AnnouncementEvent
    memory_written: bool = False
    jsonl_written: bool = False
    csv_written: bool = False


class AnnouncementRecorder:
    """Stream events to disk while retaining only a bounded diagnostic tail."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        save_jsonl: bool = True,
        save_csv: bool = True,
        max_recent_events: int = 1000,
        max_pending_events: int = 1000,
    ) -> None:
        if (
            isinstance(max_recent_events, bool)
            or not isinstance(max_recent_events, int)
            or max_recent_events < 0
        ):
            raise ValueError("max_recent_events must be >= 0")
        if (
            isinstance(max_pending_events, bool)
            or not isinstance(max_pending_events, int)
            or max_pending_events < 1
        ):
            raise ValueError("max_pending_events must be an integer >= 1")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "announcement_events.jsonl"
        self.csv_path = self.output_dir / "announcement_events.csv"
        self.save_jsonl = save_jsonl
        self.save_csv = save_csv
        self._events: deque[AnnouncementEvent] = deque(maxlen=max_recent_events)
        self._pending_deliveries: dict[str, _DeliveryState] = {}
        self._completed_fingerprints: OrderedDict[str, str] = OrderedDict()
        self._max_completed_fingerprints = max(1, max_recent_events)
        self._max_pending_events = max_pending_events
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

    def _csv_row_from_snapshot(self, row: dict[str, Any]) -> dict[str, Any]:
        details = row.pop("details")
        # Keep the established CSV columns compatible with existing files.
        # The generic delivery id lives inside details_json for CSV consumers.
        record_id = row.pop("record_id")
        if "_record_id" in details and details["_record_id"] != record_id:
            raise ValueError("event details contains a conflicting _record_id")
        details["_record_id"] = record_id
        row["details_json"] = json.dumps(details, ensure_ascii=False)
        return {key: self._spreadsheet_safe(value) for key, value in row.items()}

    def _freeze_delivery(self, event: AnnouncementEvent) -> _DeliveryState:
        """Derive every sink payload from one defensive event snapshot."""
        snapshot = asdict(event)
        serialized_event = json.dumps(snapshot, ensure_ascii=False)
        csv_row = self._csv_row_from_snapshot(deepcopy(snapshot))
        fingerprint = hashlib.sha256(serialized_event.encode("utf-8")).hexdigest()
        return _DeliveryState(
            fingerprint=fingerprint,
            serialized_event=serialized_event,
            csv_row=tuple(
                (field_name, csv_row[field_name])
                for field_name in self._csv_fieldnames()
            ),
            recent_event=AnnouncementEvent(**snapshot),
        )

    def _write_jsonl(self, serialized_event: str) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(serialized_event + "\n")

    def _write_csv(self, csv_row: tuple[tuple[str, Any], ...]) -> None:
        needs_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames())
            if needs_header:
                writer.writeheader()
            writer.writerow(dict(csv_row))

    @staticmethod
    def _sink_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0

    @staticmethod
    def _truncate_sink(path: Path, original_size: int) -> None:
        try:
            with path.open("r+b") as f:
                f.truncate(original_size)
        except FileNotFoundError:
            if original_size != 0:
                raise

    def _append_jsonl(self, serialized_event: str) -> None:
        transaction_started = False
        try:
            with _exclusive_path_lock(self.jsonl_path):
                transaction_started = True
                original_size = self._sink_size(self.jsonl_path)
                try:
                    self._write_jsonl(serialized_event)
                except Exception as exc:
                    self._truncate_sink(self.jsonl_path, original_size)
                    try:
                        setattr(exc, _ROLLED_BACK_ATTRIBUTE, True)
                    except Exception:
                        pass
                    raise
        except Exception as exc:
            if not transaction_started:
                try:
                    setattr(exc, _ROLLED_BACK_ATTRIBUTE, True)
                except Exception:
                    pass
            raise

    def _append_csv(self, csv_row: tuple[tuple[str, Any], ...]) -> None:
        transaction_started = False
        try:
            with _exclusive_path_lock(self.csv_path):
                transaction_started = True
                original_size = self._sink_size(self.csv_path)
                try:
                    self._write_csv(csv_row)
                except Exception as exc:
                    self._truncate_sink(self.csv_path, original_size)
                    try:
                        setattr(exc, _ROLLED_BACK_ATTRIBUTE, True)
                    except Exception:
                        pass
                    raise
        except Exception as exc:
            if not transaction_started:
                try:
                    setattr(exc, _ROLLED_BACK_ATTRIBUTE, True)
                except Exception:
                    pass
            raise

    def _jsonl_contains(self, serialized_event: str) -> bool:
        """Resolve uncertain completion without truncating concurrent writers."""
        try:
            with self.jsonl_path.open("r", encoding="utf-8") as f:
                return any(line.rstrip("\r\n") == serialized_event for line in f)
        except FileNotFoundError:
            return False

    def _csv_contains(self, csv_row: tuple[tuple[str, Any], ...]) -> bool:
        """Return whether the exact frozen CSV row is already present."""
        expected = {key: "" if value is None else str(value) for key, value in csv_row}
        try:
            with self.csv_path.open(
                newline="",
                encoding="utf-8-sig",
            ) as f:
                return any(
                    all(row.get(key) == value for key, value in expected.items())
                    for row in csv.DictReader(f)
                )
        except FileNotFoundError:
            return False

    def _remember_completed(self, record_id: str, fingerprint: str) -> None:
        self._completed_fingerprints.pop(record_id, None)
        self._completed_fingerprints[record_id] = fingerprint
        while len(self._completed_fingerprints) > self._max_completed_fingerprints:
            self._completed_fingerprints.popitem(last=False)

    def record(self, event: AnnouncementEvent) -> None:
        """Idempotently append one frozen record snapshot to each enabled sink.

        Successful sinks are not written again if a later sink fails.  A retry
        must reuse the same ``record_id`` and payload; conflicting reuse fails
        closed instead of producing divergent JSONL and CSV records.
        """
        if not isinstance(event, AnnouncementEvent):
            raise ValueError("event must be an AnnouncementEvent")
        if not isinstance(event.record_id, str) or not event.record_id.strip():
            raise ValueError("event.record_id must be a non-blank string")
        candidate = self._freeze_delivery(event)
        with self._lock:
            completed = self._completed_fingerprints.get(event.record_id)
            if completed is not None:
                if completed != candidate.fingerprint:
                    raise ValueError("record_id was reused with a different payload")
                self._completed_fingerprints.move_to_end(event.record_id)
                return

            delivery = self._pending_deliveries.get(event.record_id)
            if delivery is None:
                if len(self._pending_deliveries) >= self._max_pending_events:
                    raise BufferError("announcement pending-delivery limit reached")
                delivery = candidate
                self._pending_deliveries[event.record_id] = delivery
            elif delivery.fingerprint != candidate.fingerprint:
                raise ValueError("record_id was reused with a different payload")

            if not delivery.memory_written:
                self._events.append(deepcopy(delivery.recent_event))
                delivery.memory_written = True
            if self.save_jsonl and not delivery.jsonl_written:
                try:
                    self._append_jsonl(delivery.serialized_event)
                except Exception as exc:
                    if getattr(exc, _ROLLED_BACK_ATTRIBUTE, False) or not (
                        self._jsonl_contains(delivery.serialized_event)
                    ):
                        raise
                delivery.jsonl_written = True
            if self.save_csv and not delivery.csv_written:
                try:
                    self._append_csv(delivery.csv_row)
                except Exception as exc:
                    if getattr(exc, _ROLLED_BACK_ATTRIBUTE, False) or not (
                        self._csv_contains(delivery.csv_row)
                    ):
                        raise
                delivery.csv_written = True

            if (delivery.jsonl_written or not self.save_jsonl) and (
                delivery.csv_written or not self.save_csv
            ):
                self._pending_deliveries.pop(event.record_id, None)
                self._remember_completed(event.record_id, delivery.fingerprint)

    def discard_pending(self, record_id: str) -> bool:
        """Release one explicitly abandoned partial delivery from memory."""
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be a non-blank string")
        with self._lock:
            return self._pending_deliveries.pop(record_id.strip(), None) is not None

    def flush_csv(self) -> None:
        """Compatibility no-op: CSV rows are persisted by ``record``."""


class EventThrottler:
    """Simple per-key event rate limiter.

    Keys should include source_id so different customers do not suppress each
    other's events.
    """

    def __init__(self) -> None:
        self.last_event_time: dict[tuple[str, ...], float] = {}

    def allow(
        self, key: tuple[str, ...], timestamp_seconds: float, interval_seconds: float
    ) -> bool:
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
