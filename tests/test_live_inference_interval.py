import sys

from tools import run_live_annotated_recording as live_recording


def test_inference_due_runs_first_and_every_third_frame() -> None:
    due_frames = [
        frame_index
        for frame_index in range(10)
        if live_recording.inference_due(
            frame_index,
            3,
            has_cached_analysis=frame_index > 0,
        )
    ]

    assert due_frames == [0, 3, 6, 9]


def test_inference_due_forces_first_frame_after_reconnect() -> None:
    assert live_recording.inference_due(
        2,
        3,
        has_cached_analysis=False,
    )


def test_inference_interval_defaults_to_every_frame(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_live_annotated_recording.py",
            "--source",
            "rtmp://127.0.0.1/live/camera02",
            "--source-id",
            "camera02",
            "--output-root",
            "/tmp/camera02",
            "--models-config",
            "/tmp/models.yaml",
            "--pipeline-config",
            "/tmp/pipeline.yaml",
        ],
    )

    assert live_recording.parse_args().inference_interval == 1


def test_inference_interval_accepts_three(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_live_annotated_recording.py",
            "--source",
            "rtmp://127.0.0.1/live/camera02",
            "--source-id",
            "camera02",
            "--output-root",
            "/tmp/camera02",
            "--models-config",
            "/tmp/models.yaml",
            "--pipeline-config",
            "/tmp/pipeline.yaml",
            "--inference-interval",
            "3",
        ],
    )

    assert live_recording.parse_args().inference_interval == 3
