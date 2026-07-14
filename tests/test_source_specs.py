"""Tests for source snapshot parsing and worker source selection."""

import pytest
import numpy as np

from eating_behavior.media import (
    FramePacket,
    SourceSpec,
    configured_source_specs,
    redact_uri,
    resolve_source_specs,
    source_id_list,
)
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


def test_network_debug_source_requires_stable_source_id() -> None:
    with pytest.raises(ValueError, match="requires an explicit stable --source-id"):
        resolve_source_specs("rtsp://user:secret@debug.example/live?token=secret")


def test_string_network_source_config_requires_mapping() -> None:
    with pytest.raises(ValueError, match="must use a mapping"):
        configured_source_specs(["https://customer.example/live?token=secret"])


def test_sources_section_requires_a_list() -> None:
    with pytest.raises(ValueError, match="sources must be a list"):
        configured_source_specs(
            {"cam01": "rtsp://customer.example/live"}
        )


def test_local_debug_source_uses_stable_non_path_id(tmp_path) -> None:
    video = tmp_path / "private" / "customer-video.mp4"
    video.parent.mkdir()
    video.touch()

    first = resolve_source_specs(str(video))[0]
    second = resolve_source_specs(str(video))[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("local_")
    assert str(video) not in first.source_id
    assert first.source_id != first.uri


@pytest.mark.parametrize(
    "source_id",
    [
        "rtsp://customer.example/live",
        "custom+stream://customer.example/live",
        "\tcam_01",
        "cam_01\nforged-log-line",
        "x" * 129,
    ],
)
def test_source_id_validation_rejects_unsafe_values(source_id: str) -> None:
    with pytest.raises(ValueError, match="source_id/camera_id"):
        SourceSpec(source_id=source_id, uri="rtsp://customer.example/live")


def test_comma_separated_worker_ids_cannot_bypass_source_id_validation() -> None:
    with pytest.raises(ValueError, match="not a URL"):
        source_id_list("cam01,rtsp://user:secret@example.test/live")


def test_source_and_frame_repr_do_not_expose_uri_or_frame() -> None:
    uri = "rtsp://customer:password@example.test/private/token?secret=bearer"
    spec = SourceSpec(source_id="cam_01", uri=uri)
    packet = FramePacket(
        source="cam_01",
        frame_index=1,
        timestamp_seconds=1.5,
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        stream_url=uri,
    )

    assert uri not in repr(spec)
    assert "password" not in repr(spec)
    assert uri not in repr(packet)
    assert "array" not in repr(packet)


def test_redact_uri_removes_all_credential_bearing_components() -> None:
    uri = "rtsp://customer:password@example.test:8554/private/token?secret=bearer#fragment"

    redacted = redact_uri(uri)

    assert redacted == "rtsp://example.test:8554/<redacted>"
    assert all(secret not in redacted for secret in ("customer", "password", "private", "token", "bearer"))


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


def test_explicit_source_ids_cannot_expand_worker_assignment() -> None:
    """CLI selection may narrow an assignment but must not expand it."""
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
            worker_source_ids=("cam_customer_a_01",),
        ),
        source_specs=configured_source_specs(
            [
                {"source_id": "cam_customer_a_01", "stream_url": "rtsp://a/live"},
                {"source_id": "cam_customer_b_01", "stream_url": "rtsp://b/live"},
            ]
        ),
    )

    with pytest.raises(ValueError, match="not assigned"):
        pipeline.select_source_specs(("cam_customer_b_01",))


def test_strict_worker_requires_nonempty_assignment() -> None:
    """Production workers fail closed when their assignment is missing."""
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
            require_worker_assignments=True,
            allow_debug_source=False,
        ),
        source_specs=(),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        pipeline.select_source_specs()


def test_strict_assignments_cannot_enable_raw_debug_sources() -> None:
    with pytest.raises(ValueError, match="allow_debug_source must be false"):
        PipelineConfig(
            cat_classes={"cat"},
            feeding_classes={"feeding_head"},
            coat_min_area=1,
            identity_history_seconds=8.0,
            pass_event_interval_seconds=5.0,
            eating_event_interval_seconds=2.0,
            bowl_monitor_interval_seconds=30.0,
            require_worker_assignments=True,
            allow_debug_source=True,
        )


def test_raw_debug_source_cannot_impersonate_unassigned_camera() -> None:
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
            worker_source_ids=("cam_customer_a_01",),
            allow_debug_source=True,
        ),
    )

    with pytest.raises(PermissionError, match="not assigned"):
        pipeline.run_source(
            "rtsp://debug.example/live",
            source_id="cam_customer_b_01",
        )
