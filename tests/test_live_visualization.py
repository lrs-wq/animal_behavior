from types import SimpleNamespace

import numpy as np

import eating_behavior.pipeline as pipeline_module
from eating_behavior.media import FramePacket, SourceSpec
from eating_behavior.models import Detection
from eating_behavior.pipeline import RealtimePipeline


def test_live_show_renders_classification_and_posture(monkeypatch) -> None:
    detection = Detection(0, "cat", 0.91, (10, 20, 90, 100))
    cat = SimpleNamespace(
        detection=detection,
        posture="crouching",
        observed_posture="standing",
    )
    analysis = SimpleNamespace(
        detections=(detection,),
        cat_decisions=((detection, object(), object()),),
        poses=(),
        cats=(cat,),
    )
    source_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    rendered_frame = np.ones_like(source_frame)
    packet = FramePacket(
        source="cam_rtsp_test",
        frame_index=0,
        timestamp_seconds=1.25,
        frame=source_frame,
        stream_url="rtsp://camera.example/live",
    )
    render_calls = []
    shown_frames = []
    finish_calls = []
    flush_calls = []
    destroy_calls = []

    def fake_render(*args, **kwargs):
        render_calls.append((args, kwargs))
        return rendered_frame

    monkeypatch.setattr(pipeline_module, "iter_frames", lambda *_args, **_kwargs: (packet,))
    monkeypatch.setattr(
        pipeline_module,
        "render_classification_pose_frame",
        fake_render,
    )
    monkeypatch.setattr(
        pipeline_module.cv2,
        "imshow",
        lambda _name, frame: shown_frames.append(frame),
    )
    monkeypatch.setattr(pipeline_module.cv2, "waitKey", lambda _delay: ord("q"))
    monkeypatch.setattr(
        pipeline_module.cv2,
        "destroyAllWindows",
        lambda: destroy_calls.append(True),
    )

    pipeline = object.__new__(RealtimePipeline)
    pipeline.show = True
    pipeline.config = SimpleNamespace(
        cat_classes={"cat"},
        feeding_classes={"feeding_head"},
        feeding_association_max_distance_ratio=0.5,
        feeding_association_ambiguity_margin=0.1,
        pose_keypoint_confidence=0.3,
    )
    pipeline.recorder = SimpleNamespace(flush_csv=lambda: flush_calls.append(True))
    pipeline.process_frame = lambda *_args: analysis
    pipeline.finish_source = lambda *args: finish_calls.append(args)

    pipeline._run_source_specs(
        [SourceSpec("cam_rtsp_test", "rtsp://camera.example/live")]
    )

    assert len(render_calls) == 1
    args, kwargs = render_calls[0]
    assert args[0] is source_frame
    assert args[1:] == (analysis.detections, analysis.cat_decisions, ())
    assert kwargs["cat_pose_labels"] == {detection: "posture=crouching"}
    assert kwargs["cat_classes"] == {"cat"}
    assert len(shown_frames) == 1
    assert shown_frames[0] is rendered_frame
    assert finish_calls == [("cam_rtsp_test", 1.25, "manual_stop")]
    assert flush_calls == [True]
    assert destroy_calls == [True]
