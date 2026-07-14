"""Streaming and bounded-memory tests for announcement persistence."""

import csv
import json

from eating_behavior.events import AnnouncementEvent, AnnouncementRecorder


def event(cat_name: str, timestamp: float) -> AnnouncementEvent:
    return AnnouncementEvent(
        event_type="cat_eating",
        source="cam01",
        timestamp_seconds=timestamp,
        cat_name=cat_name,
        coat_class="blue_cat",
        confidence=0.9,
        message_key="cat_eating",
    )


def test_recorder_streams_all_rows_but_bounds_recent_memory(tmp_path) -> None:
    recorder = AnnouncementRecorder(tmp_path, max_recent_events=2)

    recorder.record(event("first", 1.0))
    recorder.record(event("second", 2.0))
    recorder.record(event("third", 3.0))

    assert [item.cat_name for item in recorder._events] == ["second", "third"]
    assert len(recorder.jsonl_path.read_text(encoding="utf-8").splitlines()) == 3
    with recorder.csv_path.open(newline="", encoding="utf-8-sig") as f:
        assert [row["cat_name"] for row in csv.DictReader(f)] == [
            "first",
            "second",
            "third",
        ]


def test_new_recorder_appends_consistently_instead_of_overwriting_csv(tmp_path) -> None:
    AnnouncementRecorder(tmp_path).record(event("first", 1.0))
    AnnouncementRecorder(tmp_path).record(event("second", 2.0))

    jsonl_rows = [
        json.loads(line)
        for line in (tmp_path / "announcement_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    with (tmp_path / "announcement_events.csv").open(
        newline="", encoding="utf-8-sig"
    ) as f:
        csv_rows = list(csv.DictReader(f))

    assert [row["cat_name"] for row in jsonl_rows] == ["first", "second"]
    assert [row["cat_name"] for row in csv_rows] == ["first", "second"]


def test_csv_neutralizes_spreadsheet_formulas(tmp_path) -> None:
    recorder = AnnouncementRecorder(tmp_path)
    recorder.record(event("=HYPERLINK(\"https://example.test\")", 1.0))

    with recorder.csv_path.open(newline="", encoding="utf-8-sig") as f:
        row = next(csv.DictReader(f))

    assert row["cat_name"].startswith("'=")


def test_disabled_file_sinks_do_not_create_event_files(tmp_path) -> None:
    recorder = AnnouncementRecorder(tmp_path, save_jsonl=False, save_csv=False)

    recorder.record(event("blue", 1.0))

    assert not recorder.jsonl_path.exists()
    assert not recorder.csv_path.exists()
    assert len(recorder._events) == 1
