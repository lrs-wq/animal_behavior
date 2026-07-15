# RTMPose 姿态检测接入

姿态链路采用 top-down 方式：YOLO 输出先由项目侧显式执行同类 NMS；NMS 后的 `cat` 框在各自 `source_id` 内做短时关联和 bbox 平滑，RTMPose 再把同一帧内符合面积限制的平滑猫框组成一个批次，输出 AP-10K 的 17 个关键点。原始姿态随后经过质量/几何门控和按 `(source_id, track_id)` 隔离的关键点时序滤波。

平滑 bbox 只用于姿态 crop。毛色、身份、`feeding_head` 关联、进食状态机和事件记录仍使用 NMS 后但未平滑的检测框，也不会使用姿态分数、姿态状态或 `track_id` 作为证据，因此不会改变既有身份和事件语义。

## 依赖

服务器 A10 的已验证组合是 Python 3.10、PyTorch 2.1.0+cu121、TorchVision 0.16.0+cu121、MMCV 2.1.0、MMEngine 0.10.7、MMDetection 3.3.0、MMPose 1.3.2 和 NumPy 1.26.4。不要把这套带编译扩展的 OpenMMLab 依赖直接装入原来的 PyTorch 2.11/cu128 环境。

从全新的 Conda 环境安装：

```bash
conda create -p /opt/eating-behavior/conda-env-rtmpose python=3.10 pip -y
conda run -p /opt/eating-behavior/conda-env-rtmpose \
  pip install -r requirements/rtmpose-cu121.txt
```

不要单独使用 `pip install .[rtmpose]` 部署 GPU 环境；Python 项目元数据无法固定 PyTorch 的 CUDA 构建和 MMCV 的专用轮子源，专用 requirements 才是可复现入口。

## 配置

`configs/models.yaml` 中的配置文件和权重都必须使用普通文件、受信目录和精确 SHA256：

```yaml
pose:
  config: /opt/eating-behavior/shared/models/rtmpose/cat_pose_v2_clean516_ap10k.py
  config_sha256: ca887cff59457f72ab6f639d417f8638f65c9626ea7dd1df9e3b7a5624296bc7
  checkpoint: /opt/eating-behavior/shared/models/rtmpose/best_coco_AP_epoch_100.pth
  checkpoint_sha256: dddd0f21441a6ed000d1fac244a7a0fb7add4aade750ff066bebb6717d1602a2

security:
  allow_pickle_weights: true
  require_sha256: true
  allowed_roots:
    - /opt/eating-behavior/shared/models
```

`configs/pipeline.yaml` 中显式启用并限制单帧工作量。检测模型的 `detection.iou` 和项目侧去重的 `detection.nms_iou` 是两个独立配置：后者即使在端到端权重绕过后端 NMS 时也会执行，并且只抑制同类别框。

```yaml
detection:
  iou: 0.70
  nms_iou: 0.70

tracking:
  enabled: true
  match_iou: 0.20
  max_center_distance_ratio: 0.60
  ambiguity_margin: 0.05
  max_age_seconds: 0.75
  bbox_smoothing_seconds: 0.12
  max_tracks_per_source: 32

pose:
  enabled: true
  min_cat_area: 4096
  max_instances_per_frame: 8
  keypoint_confidence: 0.30
  quality:
    enabled: true
    min_bbox_area_enter: 8192
    min_bbox_area_keep: 6144
    min_bbox_short_side: 64
    point_enter_confidence: 0.50
    point_keep_confidence: 0.30
    enter_min_points: 5
    enter_min_edges: 3
    enter_min_anchors: 1
    keep_min_points: 3
    keep_min_edges: 1
  smoothing:
    max_bridge_gap_seconds: 0.15
    reset_gap_seconds: 0.30
    one_euro_min_cutoff: 1.0
    one_euro_beta: 0.5
    one_euro_derivative_cutoff: 1.0
    reference_dt_seconds: 0.05
    core_hold_frames: 3
    joint_hold_frames: 2
    paw_hold_frames: 2
```

`tracking` 状态按 `source_id` 隔离。匹配歧义、track 超时、视频会话结束或时间戳回退时不会继续继承旧姿态历史。bbox EMA 的时间尺度由 `bbox_smoothing_seconds` 控制，输出只写入每只猫的 `pose_detection`。

`pose.quality.enabled` 是质量门控和时序滤波的总开关。新 track 使用 `*_enter` 阈值，已有 track 使用较低的 `*_keep` 阈值以减少遮挡造成的闪烁；门控还会检查帧/bbox 边界、AP-10K 骨架连接、锚点和骨长。通过门控的关键点在 bbox 局部坐标中使用 One Euro 滤波，避免普通 bbox 平移被误判成关键点跳变。短时缺失按核心点、关节和爪部各自的 `*_hold_frames` 有限补点并降低置信度；`max_bridge_gap_seconds` 之后停止桥接，`reset_gap_seconds` 之后丢弃该 track 的滤波状态。

如果 `pose.quality.enabled` 为 `false`，`raw_pose` 会原样作为 `pose` 返回；如果 `tracking.enabled` 为 `false`，姿态使用当前 cat 框并且没有可复用的 track 历史。

`.pth` 使用 pickle 反序列化，MMEngine 也会执行 `.py` 配置；两者都属于代码执行边界。即使全局 `require_sha256` 被关闭，启用姿态时仍强制要求这两个文件各自的 SHA256 和 `allowed_roots`。

## 输出

`RealtimePipeline.process_frame()` 每帧只调用一次姿态估计器，返回：

- `FrameAnalysis.detections`：本帧全部 YOLO 检测；
- `FrameAnalysis.cats`：与 cat 检测框对齐的逐实例结果；
- `CatFrameAnalysis.detection`：NMS 后、未平滑的原始 cat 框，继续作为身份和事件证据；
- `CatFrameAnalysis.track_id`：当前 `source_id` 会话内的临时 track 编号；
- `CatFrameAnalysis.pose_detection`：仅供姿态 crop/坐标使用的平滑框；
- `CatFrameAnalysis.raw_pose`：RTMPose 未经质量门控和时序滤波的诊断输出；
- `CatFrameAnalysis.pose`：门控和滤波后的对外姿态，拒绝时为 `None`；
- `CatFrameAnalysis.pose_status`：`raw`、`missing`、`accepted`、`held` 或 `rejected`；
- `CatFrameAnalysis.pose_rejection_code`：拒绝或 hold 的有界原因码；
- `CatFrameAnalysis.pose_held_keypoints` / `pose_dropped_keypoints`：本实例补点和丢点计数；
- `FrameAnalysis.poses`：所有非空 `CatFrameAnalysis.pose` 的便捷集合；
- `FrameAnalysis.pose_error_code`：可恢复的单帧姿态错误；
- `FrameAnalysis.eating_state`：本帧更新后的进食状态。

`raw_pose` 仅用于比较模型原始输出与稳定化结果，不能用于身份、进食关联或事件判定。`finish_source()` 会同时清理该 source 的 tracker 和关键点滤波状态，防止不同媒体会话复用历史。

批量测试工具会在输出视频中画出骨架与关键点，并在 `batch_summary.json` 记录姿态帧数、实例数、可见关键点数和错误计数。
