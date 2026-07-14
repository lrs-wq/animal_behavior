# RTMPose 姿态检测接入

姿态链路采用 top-down 方式：YOLO 先检测 `cat` 框，RTMPose 再把同一帧内符合面积限制的猫框组成一个批次，输出 AP-10K 的 17 个关键点。姿态结果通过 `FrameAnalysis` 返回，供视频标注和后续行为特征使用；当前版本不会把姿态分数写入身份投票或进食状态机，因此不会改变既有事件语义。

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

`configs/pipeline.yaml` 中显式启用并限制单帧工作量：

```yaml
pose:
  enabled: true
  min_cat_area: 4096
  max_instances_per_frame: 8
  keypoint_confidence: 0.30
```

`.pth` 使用 pickle 反序列化，MMEngine 也会执行 `.py` 配置；两者都属于代码执行边界。即使全局 `require_sha256` 被关闭，启用姿态时仍强制要求这两个文件各自的 SHA256 和 `allowed_roots`。

## 输出

`RealtimePipeline.process_frame()` 每帧只调用一次姿态估计器，返回：

- `FrameAnalysis.detections`：本帧全部 YOLO 检测；
- `FrameAnalysis.cats`：检测框、毛色、身份和与该框对齐的可选姿态；
- `FrameAnalysis.poses`：成功姿态的便捷集合；
- `FrameAnalysis.pose_error_code`：可恢复的单帧姿态错误；
- `FrameAnalysis.eating_state`：本帧更新后的进食状态。

批量测试工具会在输出视频中画出骨架与关键点，并在 `batch_summary.json` 记录姿态帧数、实例数、可见关键点数和错误计数。
