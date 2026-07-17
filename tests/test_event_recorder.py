"""Streaming and bounded-memory tests for announcement persistence."""

import csv
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

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
    recorder.record(event('=HYPERLINK("https://example.test")', 1.0))

    with recorder.csv_path.open(newline="", encoding="utf-8-sig") as f:
        row = next(csv.DictReader(f))

    assert row["cat_name"].startswith("'=")


def test_disabled_file_sinks_do_not_create_event_files(tmp_path) -> None:
    recorder = AnnouncementRecorder(tmp_path, save_jsonl=False, save_csv=False)

    recorder.record(event("blue", 1.0))

    assert not recorder.jsonl_path.exists()
    assert not recorder.csv_path.exists()
    assert len(recorder._events) == 1


def test_csv_write_then_error_is_recognized_without_duplicate_retry(tmp_path) -> None:
    recorder = AnnouncementRecorder(tmp_path)
    announcement = event("blue", 1.0)
    original_append_csv = recorder._append_csv
    attempts = 0

    def fail_csv_once(csv_row) -> None:
        nonlocal attempts
        attempts += 1
        original_append_csv(csv_row)
        if attempts == 1:
            raise OSError("csv unavailable")

    recorder._append_csv = fail_csv_once

    recorder.record(announcement)

    assert len(recorder.jsonl_path.read_text(encoding="utf-8").splitlines()) == 1
    assert list(recorder._events) == [announcement]

    recorder.record(announcement)
    recorder.record(announcement)

    assert len(recorder.jsonl_path.read_text(encoding="utf-8").splitlines()) == 1
    with recorder.csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert json.loads(rows[0]["details_json"])["_record_id"] == (announcement.record_id)
    assert len(recorder._events) == 1


def test_failure_before_csv_write_retries_only_the_unfinished_sink(tmp_path) -> None:
    recorder = AnnouncementRecorder(tmp_path)
    announcement = event("blue", 1.0)
    original_append_csv = recorder._append_csv
    attempts = 0

    def fail_before_csv_once(csv_row) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("csv unavailable")
        original_append_csv(csv_row)

    recorder._append_csv = fail_before_csv_once

    with pytest.raises(OSError, match="csv unavailable"):
        recorder.record(announcement)
    assert len(recorder.jsonl_path.read_text(encoding="utf-8").splitlines()) == 1

    recorder.record(announcement)

    assert len(recorder.jsonl_path.read_text(encoding="utf-8").splitlines()) == 1
    with recorder.csv_path.open(newline="", encoding="utf-8-sig") as f:
        assert len(list(csv.DictReader(f))) == 1


def test_jsonl_write_then_error_is_recognized_without_duplicate_retry(
    tmp_path,
) -> None:
    recorder = AnnouncementRecorder(tmp_path, save_csv=False)
    announcement = event("blue", 1.0)
    original_append_jsonl = recorder._append_jsonl
    attempts = 0

    def fail_jsonl_once(serialized_event: str) -> None:
        nonlocal attempts
        attempts += 1
        original_append_jsonl(serialized_event)
        if attempts == 1:
            raise OSError("jsonl close failed")

    recorder._append_jsonl = fail_jsonl_once

    recorder.record(announcement)

    recorder.record(announcement)

    rows = recorder.jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["record_id"] == announcement.record_id


def test_partial_jsonl_record_is_rolled_back_under_sink_lock(tmp_path) -> None:
    recorder = AnnouncementRecorder(tmp_path, save_csv=False)
    announcement = event("blue", 1.0)
    original_write_jsonl = recorder._write_jsonl

    def write_partial_then_fail(serialized_event: str) -> None:
        with recorder.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(serialized_event[:20])
            f.flush()
        raise OSError("partial jsonl write")

    recorder._write_jsonl = write_partial_then_fail
    with pytest.raises(OSError, match="partial jsonl write"):
        recorder.record(announcement)
    assert recorder.jsonl_path.stat().st_size == 0

    recorder._write_jsonl = original_write_jsonl
    recorder.record(announcement)

    rows = recorder.jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["record_id"] == announcement.record_id


def test_partial_csv_record_is_rolled_back_under_sink_lock(tmp_path) -> None:
    recorder = AnnouncementRecorder(tmp_path, save_jsonl=False)
    announcement = event("blue", 1.0)
    original_write_csv = recorder._write_csv

    def write_partial_then_fail(csv_row) -> None:
        del csv_row
        with recorder.csv_path.open("ab") as f:
            f.write(b"partial-row")
            f.flush()
        raise OSError("partial csv write")

    recorder._write_csv = write_partial_then_fail
    with pytest.raises(OSError, match="partial csv write"):
        recorder.record(announcement)
    assert recorder.csv_path.stat().st_size == 0

    recorder._write_csv = original_write_csv
    recorder.record(announcement)

    with recorder.csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["cat_name"] == "blue"


def test_uncertain_write_check_never_truncates_another_recorder(tmp_path) -> None:
    recorder_a = AnnouncementRecorder(tmp_path)
    recorder_b = AnnouncementRecorder(tmp_path)
    event_a = event("blue", 1.0)
    event_b = event("tabby", 2.0)
    original_append_csv = recorder_a._append_csv

    def append_a_then_b_then_error(csv_row) -> None:
        original_append_csv(csv_row)
        recorder_b.record(event_b)
        raise OSError("late csv error")

    recorder_a._append_csv = append_a_then_b_then_error
    recorder_a.record(event_a)

    jsonl_rows = [
        json.loads(line)
        for line in recorder_a.jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    with recorder_a.csv_path.open(newline="", encoding="utf-8-sig") as f:
        csv_rows = list(csv.DictReader(f))

    assert [row["cat_name"] for row in jsonl_rows] == ["blue", "tabby"]
    assert [row["cat_name"] for row in csv_rows] == ["blue", "tabby"]


def test_two_recorder_instances_serialize_concurrent_file_appends(tmp_path) -> None:
    recorder_a = AnnouncementRecorder(tmp_path)
    recorder_b = AnnouncementRecorder(tmp_path)
    events_a = [event(f"a-{index}", float(index)) for index in range(20)]
    events_b = [event(f"b-{index}", float(index)) for index in range(20)]

    def persist_all(recorder, announcements) -> None:
        for announcement in announcements:
            recorder.record(announcement)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(persist_all, recorder_a, events_a),
            executor.submit(persist_all, recorder_b, events_b),
        ]
        for future in futures:
            future.result()

    jsonl_rows = [
        json.loads(line)
        for line in recorder_a.jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    with recorder_a.csv_path.open(newline="", encoding="utf-8-sig") as f:
        csv_rows = list(csv.DictReader(f))

    expected_ids = {item.record_id for item in (*events_a, *events_b)}
    assert {row["record_id"] for row in jsonl_rows} == expected_ids
    assert {
        json.loads(row["details_json"])["_record_id"] for row in csv_rows
    } == expected_ids
    assert len(jsonl_rows) == 40
    assert len(csv_rows) == 40


def test_all_sink_rows_are_derived_from_one_frozen_snapshot(tmp_path) -> None:
    recorder = AnnouncementRecorder(tmp_path)
    announcement = event("blue", 1.0)
    announcement.details["evidence"] = "original"
    original_append_jsonl = recorder._append_jsonl

    def mutate_after_jsonl(serialized_event: str) -> None:
        original_append_jsonl(serialized_event)
        announcement.details["evidence"] = "mutated"

    recorder._append_jsonl = mutate_after_jsonl
    recorder.record(announcement)

    jsonl_row = json.loads(
        recorder.jsonl_path.read_text(encoding="utf-8").splitlines()[0]
    )
    with recorder.csv_path.open(newline="", encoding="utf-8-sig") as f:
        csv_row = next(csv.DictReader(f))

    assert jsonl_row["details"]["evidence"] == "original"
    assert json.loads(csv_row["details_json"])["evidence"] == "original"
    assert recorder._events[0].details["evidence"] == "original"


def test_record_id_reuse_with_different_payload_fails_closed(tmp_path) -> None:
    recorder = AnnouncementRecorder(tmp_path, save_jsonl=False, save_csv=False)
    original = event("blue", 1.0)
    recorder.record(original)
    conflicting = event("tabby", 2.0)
    conflicting.record_id = original.record_id

    with pytest.raises(ValueError, match="different payload"):
        recorder.record(conflicting)

    assert list(recorder._events) == [original]


def test_discard_pending_releases_capacity_after_bounded_abandonment(tmp_path) -> None:
    recorder = AnnouncementRecorder(
        tmp_path,
        save_csv=False,
        max_pending_events=1,
    )
    first = event("blue", 1.0)
    second = event("tabby", 2.0)
    original_append_jsonl = recorder._append_jsonl

    def unavailable(serialized_event: str) -> None:
        del serialized_event
        raise OSError("jsonl unavailable")

    recorder._append_jsonl = unavailable
    with pytest.raises(OSError, match="jsonl unavailable"):
        recorder.record(first)
    with pytest.raises(BufferError, match="pending-delivery"):
        recorder.record(second)

    assert recorder.discard_pending(first.record_id) is True
    assert recorder.discard_pending(first.record_id) is False
    recorder._append_jsonl = original_append_jsonl
    recorder.record(second)

    rows = recorder.jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["record_id"] == second.record_id


def test_new_rows_remain_compatible_with_legacy_csv_header(tmp_path) -> None:
    recorder = AnnouncementRecorder(tmp_path, save_jsonl=False)
    legacy_fields = recorder._csv_fieldnames()
    with recorder.csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=legacy_fields)
        writer.writeheader()

    recorder.record(event("blue", 1.0))

    with recorder.csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == legacy_fields
    assert len(rows) == 1
    assert None not in rows[0]
    assert "_record_id" in json.loads(rows[0]["details_json"])


@pytest.mark.parametrize("value", [True, 0, -1])
def test_pending_delivery_limit_is_strictly_validated(tmp_path, value) -> None:
    with pytest.raises(ValueError, match="max_pending_events"):
        AnnouncementRecorder(tmp_path, max_pending_events=value)
