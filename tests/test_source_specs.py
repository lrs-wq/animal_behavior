"""Tests for source snapshot parsing and worker source selection."""

import pytest

from eating_behavior.media import configured_source_specs, resolve_source_specs
from eating_behavior.pipeline import PipelineConfig, RealtimePipeline


def test_configured_source_specs_keep_stable_id_separate_from_stream_url() -> None:
    """source_id must remain separate from token-bearing stream_url."""
    specs = configured_source_specs(
        [
            {
                "source_id": "cam_customer_a_01",
                "stream_url": "rtsp://customer-a.example/live?token=secret",
            },
            {
                "camera_id": "cam_customer_b_01",
                "url": "rtsp://customer-b.example/live?token=secret",
            },
        ],
        selected_source_ids=["cam_customer_b_01"],
    )

    assert len(specs) == 1
    assert specs[0].source_id == "cam_customer_b_01"
    assert specs[0].uri == "rtsp://customer-b.example/live?token=secret"


def test_configured_source_specs_reject_duplicate_source_id() -> None:
    """Duplicate source_id would make household ownership ambiguous."""
    with pytest.raises(ValueError, match="Duplicate configured source_id"):
        configured_source_specs(
            [
                {"source_id": "cam_shared_01", "stream_url": "rtsp://a/live"},
                {"source_id": "cam_shared_01", "stream_url": "rtsp://b/live"},
            ]
        )


def test_raw_source_can_be_tagged_with_stable_source_id() -> None:
    """Debug --source can be tagged with a stable --source-id."""
    specs = resolve_source_specs("rtsp://debug.example/live", source_id="cam_debug_01")

    assert specs[0].source_id == "cam_debug_01"
    assert specs[0].uri == "rtsp://debug.example/live"


def test_pipeline_selects_worker_assigned_source_ids() -> None:
    """worker_source_ids restrict which configured sources run."""
    pipeline = RealtimePipeline(
        detector=object(),
        coat_classifier=object(),
        identity_service=object(),
        recorder=object(),
        config=PipelineConfig(
            cat_classes={"cat"},
            feeding_classes={"feeding_head"},
            coat_min_area=1,
            identity_history_seconds=8.0,
            pass_event_interval_seconds=5.0,
            eating_event_interval_seconds=2.0,
            bowl_monitor_interval_seconds=30.0,
            worker_source_ids=("cam_customer_b_01",),
        ),
        source_specs=configured_source_specs(
            [
                {"source_id": "cam_customer_a_01", "stream_url": "rtsp://a/live"},
                {"source_id": "cam_customer_b_01", "stream_url": "rtsp://b/live"},
            ]
        ),
    )

    selected = pipeline.select_source_specs()

    assert [spec.source_id for spec in selected] == ["cam_customer_b_01"]
