# 本机位真实检测框 RTMPose 重训

这套重训链路把人工关键点还原到本机位原始全帧，并将生产检测器经显式同类 NMS 后的框作为 top-down 输入 proposal。检测器只提供框和分数，绝不生成或补全关键点标签。

## 污染边界

旧的 `cat_pose_v2_clean516` 微调权重已经见过全部 v4 标签，包括重新按 clip 切分后的 val/test。因此，本配置必须从官方 AP-10K base 初始化：

```text
/root/rtmpose/checkpoints/rtmpose-m_ap10k_256x256.pth
SHA256 896e3665d849ef7eb9b6ec0995955796cc9810f024fa0aa0bdc18acb0d68bf52
```

不要用旧 `best_coco_AP_epoch_100.pth` 初始化后宣称 val/test 指标无泄漏，也不要依据旧权重在这些 v4 clip 上的表现做模型选择。数据构建器会把全部 accepted non-v4 clean crop 只放进通用训练集；本机位 val/test 仅含按 `clip_id` 隔离的 v4 全帧。

## 1. 上传映射元数据

本机的权威映射文件必须作为构建输入上传到服务器数据盘：

```powershell
ssh root@39.106.120.253 "mkdir -p /data/Eating_behavior/rtmpose/source_metadata"
scp "D:\gogelila\Eating_behavior\outputs\training_frames_4000_20260602_05_06\metadata.csv" `
  root@39.106.120.253:/data/Eating_behavior/rtmpose/source_metadata/metadata_20260602_05_06.csv
```

期望 SHA256：

```text
6e34881ca69ef95cb3f01989d0d8594ed210b7df17474b5a84e689c14ae414aa
```

## 2. 构建并审计数据

下面的命令在 A10 服务器上执行。输出位于 `/data`，不会覆盖已有目录。它还会逐帧验证 metadata 到源视频的映射；任一 proposal 未通过源框 IoU 与人工关键点覆盖率双门控时，默认整次构建失败。

```bash
cd /opt/eating-behavior/current

/opt/eating-behavior/conda-env/bin/python \
  tools/build_local_camera_rtmpose_dataset.py \
  --pose-root /data/Eating_behavior/yolo26_pose/cat_pose_v2_train_clean_20260713 \
  --metadata-csv /data/Eating_behavior/rtmpose/source_metadata/metadata_20260602_05_06.csv \
  --full-frame-root /data/Eating_behavior/v9_yolo26 \
  --video-root /data/2026_06_02/05 \
  --verify-video-pixels \
  --max-video-mae 3.0 \
  --detector /opt/eating-behavior/shared/models/detection-best.pt \
  --detector-sha256 7726d4ca9dae3812d2c0d84c15d23ce5dbf8607f9ff14bf16f448aaede53335a \
  --raw-confidence 0.05 \
  --proposal-min-score 0.25 \
  --upstream-iou 0.99 \
  --explicit-nms-iou 0.70 \
  --min-source-iou 0.50 \
  --proposal-padding 1.25 \
  --min-keypoint-coverage 1.0 \
  --split-ratios 0.70 0.15 0.15 \
  --split-seed 21 \
  --include-non-v4-train \
  --image-mode hardlink \
  --review-report /data/Eating_behavior/rtmpose/cat_pose_local_camera_fullframe_proposals_v1.rejections.csv \
  --output /data/Eating_behavior/rtmpose/cat_pose_local_camera_fullframe_proposals_v1
```

`--upstream-iou 0.99` 尽量保留 Ultralytics 上游候选，构建器随后必定再次执行自己的 class-aware NMS。多猫帧通过 Hungarian 一对一匹配，不允许复用同一 proposal。门控条件是：

- 原始 proposal 与裁剪来源框 IoU 不低于 `0.50`；来源框只作映射锚点，不当作真值。
- `1.25x` proposal 内覆盖全部已标注人工关键点。
- proposal 分数不低于 `0.25`。

失败时只会写显式指定的 review CSV，不会发布部分数据集。只有完成了人工复核并明确接受丢弃样本时，才可额外使用 `--allow-review-rejections`；这类输出在 `build_report.json` 中标记为 `partial_review_build`。

构建产物中的关键审计文件：

- `build_report.json`：输入哈希、门控、切分、污染边界和采样重复率。
- `build_manifest.csv`：逐实例 proposal、IoU、关键点覆盖率和源文件哈希。
- `clip_splits.csv`：clip 到 train/val/test 的唯一映射。
- `provenance/metadata.csv`：构建时实际使用的完整权威映射文件副本。
- `provenance/proposals_upstream.json`：显式 NMS 前的冻结检测结果，可用于可重复重建。

val/test 的 COCO annotation 保留来源裁剪框及其 `area`，推理输入框单独写入 `val_detector_bboxes.json` 与 `test_detector_bboxes.json`，由 MMPose 的 `bbox_file` 读取。来源裁剪框本身也不是可靠检测真值：审计发现 68 个来源框中 19 个 manual、18 个 prediction-changed、31 个 prediction。因此这里的 `coco/AP` 只用于同一数据定义下的内部选模，不宣称为标准 AP-10K 基准成绩。

## 3. 配置预检与训练命令

数据构建使用同时包含 Ultralytics 与 SciPy 的应用 Conda 环境；RTMPose 配置加载和训练使用专用 OpenMMLab 环境。先分别校验依赖、官方初始化权重和配置：

```bash
/opt/eating-behavior/conda-env/bin/python -m pip check
/opt/eating-behavior/conda-env/bin/python -c \
  "import cv2, numpy, scipy, ultralytics; print('builder imports ok')"

/opt/miniconda3/envs/rtmpose/bin/python -m pip check
/opt/miniconda3/envs/rtmpose/bin/python -c \
  "import mmcv, mmengine, mmdet, mmpose, torch; print('training imports ok')"
sha256sum /root/rtmpose/checkpoints/rtmpose-m_ap10k_256x256.pth

/opt/miniconda3/envs/rtmpose/bin/python - <<'PY'
from mmengine.config import Config

cfg = Config.fromfile(
    '/opt/eating-behavior/current/configs/rtmpose/'
    'cat_pose_local_camera_proposals.py')
assert cfg.load_from.endswith('rtmpose-m_ap10k_256x256.pth')
assert any(hook.get('type') == 'mmdet.PipelineSwitchHook'
           for hook in cfg.custom_hooks)
print(cfg.work_dir)
PY
```

正式训练命令：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /opt/miniconda3/envs/rtmpose/bin/python \
  /opt/miniconda3/envs/rtmpose/lib/python3.10/site-packages/mmpose/.mim/tools/train.py \
  /opt/eating-behavior/current/configs/rtmpose/cat_pose_local_camera_proposals.py
```

`CUBLAS_WORKSPACE_CONFIG` 是 `randomness.deterministic: true` 在 CUDA/cuBLAS
下的必要启动条件；缺少它时 PyTorch 会在首个训练迭代直接拒绝执行，而不是静默退回非确定性算法。

训练配置使用 `MultiSourceSampler`，每批 16 个 generic non-v4 crop 加 16 个 camera-train proposal，因此会重复采样较小的 camera-train 集。精确估算会写入 `build_report.json`。最后 20 个 epoch 由真正注册的 `mmdet.PipelineSwitchHook` 切换到低噪声、弱几何扰动的第二阶段 pipeline。配置不启用 EarlyStopping，避免在 epoch 130 之前停止而导致第二阶段永远不执行。

## 4. 2026-07-15 执行记录

本次确定性训练完成 150 epoch，日志在 epoch 130 后明确记录
`Switch pipeline now!`。输入和产物如下：

- 数据集：`/data/Eating_behavior/rtmpose/cat_pose_local_camera_fullframe_proposals_v1`
- 数据报告 SHA256：`1acaea7b5447c90dc1704f38007ef3f9c697b9da8b265b80bd7e1bb8d2895a9c`
- 训练目录：`/data/Eating_behavior/rtmpose/runs/cat_pose_local_camera_proposals_rtmpose_m_ap10k_v1`
- 部署候选：`rtmpose_m_camera05_realbbox_stage2_epoch140-56b9fb51_20260715.pth`
- 候选 SHA256：`56b9fb5165a3a2cc298fd135bba964f45cffebe6e14a6abe416dd51f0705d3ec`

同一 camera test 定义下的内部 COCO 指标：

| 权重 | 阶段 | AP | AR | 解释 |
|---|---|---:|---:|---|
| 官方 AP-10K base | 未适配 | 0.327694 | 0.340000 | 无本机位微调 |
| 新 epoch 125 | switch 前 val-best | 0.724202 | 0.780000 | 未经过第二阶段 |
| 新 epoch 140 | switch 后 | 0.739604 | 0.790000 | 部署候选 |
| 旧 clean516 best | 历史参考 | 0.831865 | 0.880000 | 已见过全部 v4 标签，指标污染 |

epoch 140 在保留下来的 post-switch checkpoint 中具有最好的 test AP；epoch
145/150 分别为 0.714741/0.724202。由于 test 只有 10 条 annotation，且本次已经用于工程对比，后续不能再把它称为未触碰的最终盲测集。正式精度结论需要新增按 clip 隔离、从未参与训练或选模的人工标注集。
