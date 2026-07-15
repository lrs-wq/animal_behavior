from __future__ import annotations

import csv
import importlib.util
import json
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_BUILDER_PATH = (
    Path(__file__).parents[1] / "tools" / "build_local_camera_rtmpose_dataset.py"
)
_SPEC = importlib.util.spec_from_file_location("rtmpose_dataset_builder", _BUILDER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_BUILDER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BUILDER
_SPEC.loader.exec_module(_BUILDER)

BuildConfig = _BUILDER.BuildConfig
Detection = _BUILDER.Detection
GateFailure = _BUILDER.GateFailure
build_dataset = _BUILDER.build_dataset
class_aware_nms = _BUILDER.class_aware_nms
one_to_one_assignment = _BUILDER.one_to_one_assignment


class FakeProposalProvider:
    def __init__(self, bad_frame: str | None = None) -> None:
        self.bad_frame = bad_frame
        self._descriptor = {
            "sha256": "a" * 64,
            "cat_class_id": 0,
            "framework": "synthetic-test-detector",
        }

    @property
    def cat_class_id(self) -> int:
        return 0

    @property
    def descriptor(self) -> dict:
        return self._descriptor

    def predict(self, images: dict[str, Path]) -> dict[str, list[Detection]]:
        output = {}
        for stem in images:
            if stem == self.bad_frame:
                output[stem] = [Detection((0.0, 0.0, 10.0, 10.0), 0.95, 0)]
            else:
                output[stem] = [
                    Detection((19.0, 19.0, 81.0, 81.0), 0.95, 0),
                    Detection((20.0, 20.0, 80.0, 80.0), 0.80, 0),
                    # Same geometry in another class must not suppress the cat.
                    Detection((19.0, 19.0, 81.0, 81.0), 0.90, 1),
                ]
        return output


def _pose_line() -> str:
    keypoints = []
    visible = {
        0: (0.50, 0.50, 2),  # nose
        1: (0.40, 0.40, 2),  # left eye
        2: (0.60, 0.40, 2),  # right eye
        3: (0.50, 0.60, 2),  # neck
    }
    for index in range(17):
        keypoints.extend(visible.get(index, (0.0, 0.0, 0)))
    values = [0, 0.5, 0.5, 1.0, 1.0, *keypoints]
    return " ".join(str(value) for value in values) + "\n"


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _make_sources(tmp_path: Path, include_generic: bool = True) -> tuple[Path, Path, Path]:
    pose_root = tmp_path / "pose"
    full_root = tmp_path / "full"
    metadata_path = tmp_path / "metadata.csv"
    manifest_rows = []
    metadata_rows = []
    for serial in range(1, 5):
        hash_part = f"{serial:08x}"
        full_stem = f"v4__{hash_part}-frame_{serial:06d}"
        pose_stem = f"{full_stem}__ann000__cls0"
        full = np.full((100, 100, 3), serial * 20, dtype=np.uint8)
        _write_image(full_root / "images" / "train" / f"{full_stem}.jpg", full)
        label_path = full_root / "labels" / "train" / f"{full_stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("0 0.5 0.5 0.6 0.6\n", encoding="utf-8")
        # The builder reconstructs the exact integer crop [20:80, 20:80].
        _write_image(
            pose_root / "images" / "train" / f"{pose_stem}.jpg",
            full[20:80, 20:80],
        )
        pose_label = pose_root / "labels" / "train" / f"{pose_stem}.txt"
        pose_label.parent.mkdir(parents=True, exist_ok=True)
        pose_label.write_text(_pose_line(), encoding="utf-8")
        manifest_rows.append(
            {
                "path": f"labels/train/{pose_stem}.txt",
                "split": "train",
                "source": "v4",
                "line": "1",
                "decision": "accepted",
            }
        )
        clip_id = f"clip_{serial}"
        metadata_rows.append(
            {
                "image": f"D:\\frames\\frame_{serial:06d}.jpg",
                "label": "",
                "source_video": f"D:\\data\\05\\{clip_id}\\0000.mp4",
                "clip_id": clip_id,
                "frame_index": str(serial),
                "video_time_sec": f"{serial / 20:.3f}",
                "fps": "20.0",
                "duration_sec": "4.0",
            }
        )

    if include_generic:
        generic_stem = "v9__generic__ann000__cls0"
        generic = np.full((64, 64, 3), 127, dtype=np.uint8)
        _write_image(pose_root / "images" / "val" / f"{generic_stem}.jpg", generic)
        generic_label = pose_root / "labels" / "val" / f"{generic_stem}.txt"
        generic_label.parent.mkdir(parents=True, exist_ok=True)
        generic_label.write_text(_pose_line(), encoding="utf-8")
        manifest_rows.append(
            {
                "path": f"labels/val/{generic_stem}.txt",
                "split": "val",
                "source": "v9",
                "line": "1",
                "decision": "accepted",
            }
        )

    pose_root.mkdir(parents=True, exist_ok=True)
    with (pose_root / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["path", "split", "source", "line", "decision"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    with metadata_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metadata_rows[0]))
        writer.writeheader()
        writer.writerows(metadata_rows)
    return pose_root, full_root, metadata_path


def test_explicit_nms_is_class_aware_and_matching_is_one_to_one() -> None:
    detections = [
        Detection((0.0, 0.0, 20.0, 20.0), 0.9, 0),
        Detection((1.0, 1.0, 19.0, 19.0), 0.8, 0),
        Detection((0.0, 0.0, 20.0, 20.0), 0.7, 1),
        Detection((40.0, 40.0, 60.0, 60.0), 0.6, 0),
    ]
    kept = class_aware_nms(detections, 0.7)
    assert [(item.score, item.class_id) for item in kept] == [
        (0.9, 0),
        (0.7, 1),
        (0.6, 0),
    ]

    assignments, ious = one_to_one_assignment(
        [(0.0, 0.0, 20.0, 20.0), (40.0, 40.0, 60.0, 60.0)],
        [kept[0], kept[2]],
    )
    assert assignments == [0, 1]
    assert len(set(assignments)) == 2
    assert np.allclose(np.diag(ious), 1.0)


def test_build_dataset_uses_human_keypoints_real_proposals_and_clip_splits(
    tmp_path: Path,
) -> None:
    pose_root, full_root, metadata = _make_sources(tmp_path)
    output = tmp_path / "output"
    report = build_dataset(
        BuildConfig(
            pose_root=pose_root,
            metadata_csv=metadata,
            full_frame_root=full_root,
            output=output,
            include_non_v4_train=True,
            image_mode="copy",
        ),
        FakeProposalProvider(),
    )

    assert report["status"] == "complete"
    assert report["counts"]["camera_manifest_instances"] == 4
    assert report["counts"]["generic_non_v4_train_instances"] == 1
    assert report["detector_counts"]["raw_all_classes"] == 12
    # Per frame: duplicate cat removed, different-class overlap retained.
    assert report["detector_counts"]["post_nms_all_classes"] == 8
    assert report["invariants"]["detector_keypoints_used_as_labels"] is False
    assert all(not overlap for overlap in report["invariants"]["camera_clip_overlap"].values())
    assert report["training_source_sampling"]["samples_per_source_per_batch"] == [16, 16]
    assert report["training_source_sampling"][
        "actual_yielded_samples_per_single_gpu_epoch"
    ] == 32
    assert (output / "provenance" / "metadata.csv").read_bytes() == metadata.read_bytes()

    all_clips: dict[str, str] = {}
    annotation_count = 0
    for split in ("train", "val", "test"):
        payload = json.loads((output / "annotations" / f"{split}.json").read_text())
        assert payload["annotations"]
        assert "bbox_role" not in payload["info"]["build"]
        assert payload["info"]["build"]["train_annotation_bbox_role"] == (
            "topdown_input_proposal_not_ground_truth"
        )
        assert payload["info"]["build"]["val_test_inference_bbox_role"] == (
            "separate_frozen_detector_bbox_file"
        )
        for image in payload["images"]:
            assert image["clip_id"] not in all_clips
            all_clips[image["clip_id"]] = split
        for annotation in payload["annotations"]:
            annotation_count += 1
            if split == "train":
                assert annotation["bbox_role"] == "topdown_input_proposal_not_ground_truth"
                assert annotation["bbox"] == [19.0, 19.0, 62.0, 62.0]
            else:
                assert annotation["bbox_role"].startswith(
                    "source_annotation_reference_bbox"
                )
                assert annotation["bbox"] == [20.0, 20.0, 60.0, 60.0]
            assert annotation["keypoint_source"] == "human_annotation_only"
            assert annotation["labeled_keypoint_coverage_in_padded_proposal"] == 1.0
            # AP-10K first point is the human left eye: 20 + 0.4 * 60 = 44.
            assert annotation["keypoints"][:3] == [44.0, 44.0, 2]
            # AP-10K third point is the human nose.
            assert annotation["keypoints"][6:9] == [50.0, 50.0, 2]
        if split != "train":
            detector_bboxes = json.loads(
                (output / "annotations" / f"{split}_detector_bboxes.json").read_text()
            )
            assert detector_bboxes == [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [19.0, 19.0, 62.0, 62.0],
                    "score": 0.95,
                }
            ]
    assert annotation_count == 4

    generic = json.loads((output / "annotations" / "generic_train.json").read_text())
    assert len(generic["annotations"]) == 1
    assert generic["info"]["camera_validation_or_test_eligible"] is False


def test_gate_failure_aborts_without_publishing_partial_dataset(tmp_path: Path) -> None:
    pose_root, full_root, metadata = _make_sources(tmp_path, include_generic=False)
    output = tmp_path / "must_not_exist"
    bad_frame = "v4__00000001-frame_000001"
    with pytest.raises(GateFailure) as captured:
        build_dataset(
            BuildConfig(
                pose_root=pose_root,
                metadata_csv=metadata,
                full_frame_root=full_root,
                output=output,
                image_mode="copy",
            ),
            FakeProposalProvider(bad_frame=bad_frame),
        )
    assert not output.exists()
    assert captured.value.rejections[0]["full_stem"] == bad_frame
    assert "source_iou_below_threshold" in captured.value.rejections[0]["reasons"]


def test_training_config_keeps_real_pipeline_switch_and_official_initialization() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "rtmpose"
        / "cat_pose_local_camera_proposals.py"
    )
    config = runpy.run_path(str(config_path))
    hook_types = [hook["type"] for hook in config["custom_hooks"]]
    assert "mmdet.PipelineSwitchHook" in hook_types
    assert "EarlyStoppingHook" not in hook_types
    assert config["load_from"].endswith("rtmpose-m_ap10k_256x256.pth")
    assert "best_coco_AP" not in config["load_from"]
    assert config["randomness"]["deterministic"] is True
    assert config["env_cfg"]["cudnn_benchmark"] is False
    assert config["train_dataloader"]["sampler"]["source_ratio"] == [1, 1]
    assert config["val_dataloader"]["dataset"]["bbox_file"].endswith(
        "val_detector_bboxes.json"
    )
    assert config["test_dataloader"]["dataset"]["bbox_file"].endswith(
        "test_detector_bboxes.json"
    )
