# 精选模型权重与选型依据

发布日期：2026-09-30。四个权重文件共 68,652,651 字节，约 68.7 MB，均保留原始字节。
权重位于 [models/weights](../models/weights/README.md)，完整 SHA-256、大小与指标记录在
[manifest.json](../models/weights/manifest.json)。训练数据和其他 checkpoint 不随仓库上传。

## 本次选择

| 用途 | 文件 | 选择依据 |
| --- | --- | --- |
| 当前生产检测 | `detection_best.pt` | 最新 v9 AdamW 200 epoch 训练的 best；mAP50=0.90633、mAP50-95=0.79980 |
| v9 验证最优检测候选 | `detection_v9_map5095_best.pt` | 100 epoch auto 训练的 epoch 93；三组 v9 对比中 mAP50-95 最高：0.80395；mAP50=0.90363 |
| 毛色分类 | `coat_classifier_best.pt` | 当前生产 v2 分类器，按框架 fitness 选出的 best；checkpoint 内 top1=0.73316、top5=0.97872、fitness=0.85594 |
| RTMPose | `rtmpose_m_camera05_realbbox_stage2_epoch140-56b9fb51_20260715.pth` | 真实机位 proposal 适配后第二阶段 epoch 140；保留的 post-switch checkpoint 中内部 AP 最高：0.739604、AR=0.79 |

三个 v9 检测训练使用相同验证数据、640 输入、batch 16 和 seed 20260820。
100 epoch auto / 200 epoch auto / 200 epoch AdamW 的最佳 mAP50-95 分别为
0.80395 / 0.79942 / 0.79980。AdamW 的 mAP50 最高，因此同时提供现行版本和 mAP50-95 候选。
这些指标不能与更早的 v4/v5 等不同数据集直接排名。

分类器按框架组合 fitness 选模，所选 checkpoint 的 top1 不是逐 epoch 的最高 top1。
早期 v1 的更高数值来自不同数据与类别定义，不作为本次替换依据。

RTMPose 的 camera 内部对比只有 10 条 annotation，已用于工程选型，不能称为未触碰的最终盲测。
旧 clean516 参考的 AP 更高，但曾见过全部 v4 标签，存在指标污染；本次采用已注册的真实机位
适配权重，后续精度结论仍需新的 clip 隔离评测集。完整背景见
[RTMPose 训练记录](../training/rtmpose/README.md#4-2026-07-15-执行记录)。

## 克隆后使用

普通 `git clone` 即可获得四个完整权重。先从仓库根目录校验：

```bash
(cd models/weights && sha256sum -c SHA256SUMS)
```

macOS 可将上面的 `sha256sum` 换成 `shasum -a 256`。

两个配置均要求在仓库根目录运行，保留 allowed_roots 和完整 SHA-256 检查：

- `configs/models.github.yaml`：复现当前生产的检测、分类和姿态权重组合。
- `configs/models.github.best-map5095.yaml`：仅将检测器替换为 v9 验证 mAP50-95 最优候选。

安装推理环境后，可运行自己的输入视频：

```bash
python -m eating_behavior.main \
  --source /absolute/path/to/your/video.mp4 \
  --source-id local_demo \
  --models-config configs/models.github.yaml \
  --pipeline-config configs/pipeline.yaml \
  --output-dir outputs/model-demo \
  --show
```

服务器运行时将输出目录指向 `/data/Eating_behavior` 下的对应目录。
启用 RTMPose 需要既有 OpenMMLab 依赖，安装方法见
[RTMPose 集成说明](rtmpose_integration.md) 与 `requirements/rtmpose-cu121.txt`。

本次只分发选定模型副本；服务器 `configs/models.yaml`、实际模型内容和生产服务不自动切换。
训练产物进入生产仍须按既有评估、受控复制和 SHA-256 注册流程执行。

## 发布前验证

四个权重的大小与 SHA-256 均与阿里云原件一致，两个仓库本地配置均通过允许目录和哈希校验。
在既有推理环境中以 CPU 读取演示截图，已验证现行检测、毛色分类、RTMPose 的 17 点输出，
以及检测候选的实际推理；Python 回归测试 427 项全部通过。
