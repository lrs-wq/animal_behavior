# 猫咪进食行为识别正式项目

本项目是猫咪进食行为识别的正式推理工程，负责接入外部训练好的模型，完成视频推理、进食事件分析、猫咪身份识别、结果评估和错误诊断。

本项目**不参与模型训练**。YOLO 检测模型、品种/外观分类模型、Re-ID 或 embedding 模型的训练代码均在其他项目中维护。本项目只使用这些外部训练产物。

## 0. 当前主业务链路

当前正式项目优先面向用户视频流实时推理，主入口必须同时兼容视频文件、视频文件夹、摄像头编号和 RTSP / HTTP 视频流。

第一版主链路为：

```text
用户视频/视频流
  -> 猫检测
  -> 毛色/外观类别确认
  -> 如果该毛色在用户家庭中只有一只猫，直接确认身份并结束身份分支
  -> 如果该毛色存在多只猫，进入身份确认/Re-ID 分支
  -> 同步做 feeding_head 吃饭检测
  -> 记录可播报事件
```

当前需要记录的可播报事件：

- `cat_passed`：猫咪经过，附带猫咪身份；无法确认身份时 `cat_name` 留空。
- `cat_eating`：猫咪吃饭，附带猫咪身份；当前帧无法确认时参考最近身份时间线，仍无法确认则 `cat_name` 留空。
- `bowl_food_status`：碗内余粮监控占位，后续补充具体模型和规则。

播报方式暂不在第一版实现中固定，当前只负责把事件结构化记录到文件，后续可以接语音、App 推送、WebSocket 或其他消费端。

## 1. 项目目标

正式项目的目标是把当前原型脚本整理成稳定、可维护、可评估的推理系统：

```text
视频输入
  -> 读取外部模型权重
  -> YOLO 检测 cat / feeding_head
  -> 进食事件时间平滑
  -> 进食头部裁剪
  -> 猫咪身份图库比对
  -> 事件级身份聚合
  -> 输出 CSV / JSON / 可视化视频 / 诊断报告
```

第一阶段重点支持两个能力：

1. **进食事件检测**：输出吃饭开始时间、结束时间和持续时长。
2. **进食猫咪身份识别**：在检测到进食后，判断正在吃饭的是哪只猫；事件输出中低置信度身份留空，内部诊断和评估仍保留 `unknown` 状态。

后续可扩展：

- 路过猫咪身份识别。
- 品种/外观类型分流。
- 多猫切换事件切分。
- 评估报告自动生成。

## 2. 项目边界

### 2.1 本项目负责

- 加载外部训练好的 YOLO 检测权重。
- 加载外部训练好的分类模型或 embedding 模型。
- 构建和读取猫咪身份图库 embedding。
- 对视频、文件夹、摄像头或流媒体做推理。
- 将帧级检测结果转换成进食事件。
- 将帧级身份结果聚合成事件级猫名。
- 输出结构化结果、可视化视频、错误诊断素材。
- 对外部模型版本做离线评估和回归检查。

### 2.2 本项目不负责

- YOLO 训练。
- 分类模型训练。
- Re-ID / metric learning 训练。
- 训练集划分。
- 数据增强。
- 训练任务调度。
- 标注平台开发。
- 大规模训练数据管理。

训练项目产出模型权重后，本项目只负责接入、推理、评估和反馈问题。

## 3. 当前原型代码参考

当前 `Eating_behavior_project` 已经有一版可运行原型，正式项目会从这些脚本中迁移推理逻辑：

| 原型文件 | 当前职责 | 正式项目迁移方向 |
| --- | --- | --- |
| `detect_eating.py` | YOLO 进食检测、状态机、事件输出 | 检测模块、状态机模块、CLI |
| `detect_eating_identity.py` | 进食检测 + 身份识别 + 事件聚合 | 身份推理管线、事件身份聚合 |
| `cat_identity.py` | DINOv2 / MegaDescriptor / ResNet 特征提取和图库匹配 | identity 核心库 |
| `build_cat_gallery.py` | 从猫头图库构建 embedding | 图库构建 CLI |
| `extract_cat_heads.py` | 从图片中自动裁剪猫头 | 数据工具脚本 |
| `build_balanced_video_head_gallery.py` | 为高相似猫构建平衡图库 | 诊断/实验工具 |
| `diagnose_identity_vectors.py` | 身份错误向量诊断 | diagnostics 模块 |
| `cut_test_eating_clips.py` | 裁剪事件片段和标签点片段 | review clips 工具 |
| `run_all_videos_visual.py` | 批量可视化运行 | batch CLI |

正式项目不会直接照搬单文件脚本结构，而是拆成库模块和命令行入口。

## 4. 设计原则

### 4.1 YOLO 不直接识别猫身份

YOLO 只负责检测目标和进食状态，不负责判断是哪只猫。

检测类别建议：

| 类别 | 说明 | 阶段 |
| --- | --- | --- |
| `cat` | 整只猫或可见猫身体区域 | 必需 |
| `feeding_head` | 正在吃饭的猫头 | 必需 |
| `food_bowl` | 食盆或食物区域 | 后续建议 |
| `side_body` | 侧身花纹区域 | 后续建议 |

身份识别采用两阶段方式：

```text
feeding_head 检测框
  -> 外扩裁剪
  -> 提取 embedding
  -> 与猫咪图库做相似度比对
  -> 多帧投票
  -> 事件级猫名
```

### 4.2 猫名来自图库文件夹

身份识别必须遵守：

```text
cat_name = 图库一级文件夹名
```

示例：

```text
data/gallery/head/小蓝猫
data/gallery/head/纯银渐
data/gallery/head/长银渐
data/gallery/head/串银渐
```

输出猫名分别为：

```text
小蓝猫
纯银渐
长银渐
串银渐
```

代码中不得硬编码猫名列表。新增、删除、重命名猫，只修改图库文件夹，然后重新构建图库 embedding。

### 4.3 低置信度优先 unknown

身份识别不能强行给出猫名。

第一版默认阈值建议：

```text
identity_score >= 0.55
score_margin >= 0.05
event_confirm_ratio >= 0.60
```

如果分数不足或第一名和第二名过近，输出：

```text
cat_name = unknown
```

这里的 `unknown` 是内部识别状态，用于诊断、评估和阈值策略。对外可播报事件 `cat_passed` / `cat_eating` 的 `cat_name` 字段在无法确认身份时输出为空字符串。

高相似猫，如银渐层、蓝猫、狸花、美短虎斑，应使用更严格的 margin 和 unknown 策略。

### 4.4 头部身份和整猫身份分开

第一阶段主做吃饭身份识别：

```text
feeding_head -> head_gallery -> eating identity
```

后续扩展路过身份识别：

```text
cat / side_body -> body_gallery -> passing identity
```

不要把低头吃饭猫头、整猫侧身、背部花纹全部混进同一个图库。正式项目保留双图库设计：

```text
data/gallery/head
data/gallery/body
```

### 4.5 非纯色猫花纹辅助身份识别

对于非纯色猫，可以增加一个轻量的花纹辅助身份识别分支。该分支不负责判断猫的花色类型，花色类型由外部 YOLO 分类或其他分类模型先确定；花纹分支只在同一花色范围内辅助区分具体是哪只猫。

整体定位：

```text
YOLO 花色分类 = 分流
花纹比对 = 同花色猫之间的辅助身份确认
```

示例流程：

```text
检测到 cat / side_body / head_top
  -> YOLO 分类得到花色类型
  -> 如果是纯色猫，跳过花纹分支
  -> 如果是非纯色猫，只检索同花色图库
  -> 生成灰度增强图或边缘纹理图
  -> 做轻量花纹匹配
  -> pattern_score 作为身份分数的辅助项
```

第一版不建议人工绘制花纹走向，也不建议做复杂的全局花纹理解。更适合使用自动预处理：

```text
原图 crop
  -> 转灰度
  -> CLAHE 局部对比度增强
  -> 轻度去噪
  -> Sobel / Canny / AKAZE / ORB 提取纹理或局部特征
```

推荐图库结构：

```text
data/gallery/pattern/
  tabby/
    猫A/
    猫B/
  tricolor/
    猫C/
    猫D/
  tortoiseshell/
    猫E/
```

也可以先不新建独立图库，只在 `body_gallery` 的 metadata 中增加字段：

```text
color_pattern
view_type
is_patterned
pattern_quality_score
```

第一版融合方式建议保持简单：

```text
final_score =
  0.75 * 原图 identity embedding score
+ 0.25 * pattern_score
```

如果侧身、背部或额头纹路非常清晰，可以调整为：

```text
final_score =
  0.65 * 原图 identity embedding score
+ 0.35 * pattern_score
```

该分支适合：

- 狸花、虎斑、美短虎斑、三花、玳瑁、双色带明显花纹的猫。
- 路过身份识别。
- 侧身、背部、额头纹路清晰可见的画面。
- 同花色多猫家庭中的辅助区分。

该分支不适合：

- 纯色猫。
- 只露很小一块身体的画面。
- 严重模糊、夜视强噪声或压缩严重的视频。
- 低头进食时只露耳朵、后脑，且没有稳定花纹区域的 crop。

正式项目中，这个能力应作为 `pattern-aware identity branch` 可选开启，不替代原有 embedding 身份识别，只作为同花色非纯色猫的辅助分数来源。

## 5. 推荐项目结构

```text
Eating_behavior_formal_project/
  README.md
  pyproject.toml
  configs/
    paths.yaml
    models.yaml
    detection.yaml
    identity.yaml
    evaluation.yaml
  src/
    eating_behavior/
      __init__.py
      config/
        loader.py
        schemas.py
      detection/
        yolo_detector.py
        eating_state_machine.py
        event_postprocess.py
      identity/
        feature_extractor.py
        gallery.py
        matcher.py
        event_identity.py
      media/
        source_resolver.py
        video_reader.py
        video_writer.py
      evaluation/
        label_reader.py
        event_metrics.py
        identity_metrics.py
        reports.py
      diagnostics/
        identity_timeline.py
        vector_diagnostics.py
        review_clips.py
      visualization/
        overlay.py
      cli/
        detect_eating.py
        build_gallery.py
        detect_identity.py
        evaluate.py
        diagnose_identity.py
  scripts/
    migrate_from_prototype.py
    run_batch.py
    cut_review_clips.py
  docs/
    model_inputs.md
    data_interfaces.md
    output_formats.md
    evaluation_protocol.md
    troubleshooting.md
  tests/
    test_state_machine.py
    test_event_postprocess.py
    test_identity_matcher.py
  data/
    gallery/
      head/
      body/
    embeddings/
      head/
      body/
    evaluation/
  outputs/
    eating_detection/
    eating_identity/
    diagnostics/
    reports/
```

目录说明：

- `configs/`：统一管理路径、模型权重、阈值和输出目录。
- `src/eating_behavior/`：正式库代码。
- `src/eating_behavior/cli/`：命令行入口。
- `scripts/`：一次性迁移和批处理工具。
- `docs/`：数据接口、输出格式、模型接入和排错文档。
- `tests/`：状态机、事件合并、身份匹配等核心逻辑测试。
- `data/gallery/`：本项目使用的猫咪注册图库。
- `data/embeddings/`：图库 embedding 产物。
- `outputs/`：推理、诊断、评估结果。

## 6. 外部输入

### 6.1 模型权重

本项目通过配置文件读取外部模型：

```yaml
detection_model: D:/gogelila/Eating_behavior/yolo/runs/.../best.pt
breed_classifier_model: D:/gogelila/Eating_behavior/yolo/runs_cls/.../best.pt
feature_backend: dinov2
feature_model: facebook/dinov2-small
```

模型来源由训练项目保证，本项目只校验文件是否存在、类别是否符合要求、推理输出是否可解析。

### 6.2 视频输入

支持输入：

```text
单个视频文件
视频文件夹
摄像头编号
RTSP / HTTP 视频流
```

支持格式：

```text
.mp4 .avi .mov .mkv .m4v .wmv .flv .webm
```

### 6.3 身份图库

头部图库：

```text
data/gallery/head/
  小蓝猫/
    001.jpg
    002.jpg
  纯银渐/
    001.jpg
    002.jpg
```

整猫图库：

```text
data/gallery/body/
  小蓝猫/
  纯银渐/
```

第一阶段只要求 `head` 图库可用。

### 6.4 评估标签

评估标签建议使用结构化 CSV：

```text
source
label_time_seconds
true_cat_name
feeding_start_seconds
feeding_end_seconds
scene_type
notes
```

也可以兼容当前原型中的 `0000.txt` 标签形式，但正式项目内部应统一转换为标准表。

## 7. 核心流程

### 7.1 构建身份图库

输入：

```text
data/gallery/head/<cat_name>/*.jpg
```

输出：

```text
data/embeddings/head/cat_gallery_embeddings.npz
data/embeddings/head/cat_gallery_metadata.csv
data/embeddings/head/cat_gallery_summary.json
```

metadata 至少包含：

```text
image_path
cat_name
folder_name
embedding_index
gallery_type
feature_backend
feature_model
```

### 7.2 进食事件检测

当前状态机规则：

```text
检测到 feeding_head
  -> 连续 enter_seconds 后进入 EATING
  -> 连续 exit_seconds 无 feeding_head 后退出 EATING
  -> merge_gap_seconds 合并短暂停顿
  -> min_event_seconds 过滤过短事件
```

默认参数：

```text
enter_seconds: 2.0
exit_seconds: 5.0
merge_gap_seconds: 10.0
min_event_seconds: 2.0
```

输出字段：

```text
source
start_seconds
end_seconds
duration_seconds
start_time
end_time
duration_time
```

### 7.3 进食身份识别

帧级流程：

```text
feeding_head 检测
  -> padded crop
  -> feature embedding
  -> cosine similarity
  -> per-cat score
  -> score / margin 阈值
```

事件级流程：

```text
事件内 accepted observations
  -> weight = identity_score * feeding_head_confidence
  -> 按 cat_name 加权投票
  -> top1_ratio >= event_confirm_ratio 才确认猫名
```

事件身份输出字段：

```text
source
start_seconds
end_seconds
duration_seconds
start_time
end_time
duration_time
cat_name
event_identity_confidence
event_identity_margin
observation_count
accepted_observation_count
unknown_observation_count
```

### 7.4 身份时间线

正式项目必须补充 `identity_timeline.csv`，用于排查事件内猫名变化。

建议字段：

```text
source
timestamp_seconds
timestamp_time
event_id
feeding_confidence
cat_name
accepted
identity_score
second_cat_name
second_score
score_margin
x1
y1
x2
y2
crop_path
```

### 7.5 错误诊断

诊断输出应包含：

- query crop。
- query embedding `.npy`。
- query embedding `.csv`。
- top gallery matches。
- per-cat scores。
- contact sheet。
- 标签点附近局部窗口身份结果。
- 对应视频 review clip。

## 8. 身份打分策略

当前原型使用：

```text
topk_mean
```

正式项目需要支持参数化：

| 策略 | 说明 |
| --- | --- |
| `topk_mean` | 每只猫取 top-k 相似图库图平均，稳定但可能被相似图库拉偏 |
| `best_single` | 只取单张最近邻，局部信号强但容易抖 |
| `weighted_topk` | top1 权重更高，兼顾稳定和最近邻 |

银渐层等高相似猫建议优先对比 `best_single` 和 `weighted_topk`，并配合更严格 margin。

## 9. 配置文件建议

### 9.1 `configs/paths.yaml`

```yaml
project_root: D:/gogelila/Eating_behavior/Eating_behavior_formal_project
gallery_head_dir: data/gallery/head
gallery_body_dir: data/gallery/body
embedding_head_dir: data/embeddings/head
output_dir: outputs
```

### 9.2 `configs/models.yaml`

```yaml
detection_model: D:/gogelila/Eating_behavior/yolo/runs/cat_feeding_head/weights/best.pt
classifier_model: D:/gogelila/Eating_behavior/yolo/runs_cls/cat_breed/weights/best.pt
feature_backend: dinov2
feature_model: facebook/dinov2-small
```

### 9.3 `configs/detection.yaml`

```yaml
feeding_classes:
  - feeding_head
confidence: 0.25
iou: 0.70
imgsz: 640
enter_seconds: 2.0
exit_seconds: 5.0
merge_gap_seconds: 10.0
min_event_seconds: 2.0
vid_stride: 1
```

### 9.4 `configs/identity.yaml`

```yaml
gallery: data/embeddings/head/cat_gallery_embeddings.npz
crop_padding: 0.15
score_mode: topk_mean
top_k: 3
min_score: 0.55
min_margin: 0.05
event_confirm_ratio: 0.60
```

## 10. 命令行规划

构建头部图库：

```powershell
python -m eating_behavior.cli.build_gallery `
  --gallery-type head `
  --input data/gallery/head `
  --output data/embeddings/head `
  --config configs/models.yaml
```

检测进食事件：

```powershell
python -m eating_behavior.cli.detect_eating `
  --source D:\path\to\video_or_dir `
  --config configs/detection.yaml `
  --output outputs/eating_detection
```

检测进食身份：

```powershell
python -m eating_behavior.cli.detect_identity `
  --source D:\path\to\video_or_dir `
  --detection-config configs/detection.yaml `
  --identity-config configs/identity.yaml `
  --output outputs/eating_identity
```

评估模型结果：

```powershell
python -m eating_behavior.cli.evaluate `
  --pred outputs/eating_identity/identity_events_all.csv `
  --labels data/evaluation/labels.csv `
  --config configs/evaluation.yaml
```

身份错误诊断：

```powershell
python -m eating_behavior.cli.diagnose_identity `
  --labels data/evaluation/error_points.csv `
  --gallery data/embeddings/head/cat_gallery_embeddings.npz `
  --output outputs/diagnostics
```

## 11. 评估指标

本项目用于评估外部模型效果。

### 11.1 进食事件指标

| 指标 | 目标 |
| --- | ---: |
| 进食事件 F1 | >= 0.85 |
| 开始时间误差中位数 | <= 2 秒 |
| 结束时间误差中位数 | <= 3 秒 |
| 误报事件 | <= 1 次/小时/摄像头 |
| 漏报完整进食事件 | <= 3% |

### 11.2 身份识别指标

必须分别统计：

```text
feeding_identity_accuracy
same_breed_feeding_identity_accuracy
unknown_reject_accuracy
event_level_identity_accuracy
wrong_confirm_rate
```

高相似猫必须单独输出混淆矩阵，例如：

```text
纯银渐 vs 长银渐 vs 串银渐
```

### 11.3 诊断重点

重点检查：

- 标签点附近是否真的检测到 `feeding_head`。
- 单帧身份是否被阈值拒识。
- 事件级投票是否覆盖局部真实猫名。
- 高相似猫是否 top1/top2 margin 过小。
- 图库是否缺少低头、头顶、后脑、同摄像头样本。

## 12. 第一阶段实施计划

### 阶段 1：工程骨架

- 建立 `src/eating_behavior/`。
- 建立 `configs/`。
- 建立 `data/gallery`、`data/embeddings`、`outputs` 目录约定。
- 准备 `pyproject.toml` 或正式依赖文件。

### 阶段 2：迁移进食检测

- 从 `detect_eating.py` 迁移 YOLO 推理封装。
- 迁移 `EatingStateMachine`。
- 迁移事件合并和 CSV / JSON 输出。
- 添加状态机和事件合并单元测试。

### 阶段 3：迁移身份识别

- 从 `cat_identity.py` 迁移 feature extractor。
- 迁移 gallery builder。
- 迁移 matcher。
- 增加 `score_mode` 参数。
- 支持 `head_gallery` 第一版。

### 阶段 4：补充诊断和评估

- 输出 `identity_timeline.csv`。
- 实现标签点局部窗口评估。
- 输出 top matches、contact sheet、query embedding。
- 生成 review clips。

### 阶段 5：外部训练项目反馈闭环

- 根据评估结果整理错误样例。
- 输出需要补充的数据类型。
- 记录模型版本、图库版本、评估报告版本。
- 将问题反馈给训练项目，而不是在本项目中训练。

## 13. 输出目录约定

```text
outputs/
  eating_detection/
    events_all.csv
    events_all.json
    *_events.csv
    *_annotated.mp4
  eating_identity/
    identity_events_all.csv
    identity_events_all.json
    identity_timeline.csv
    *_identity_annotated.mp4
  diagnostics/
    identity_vector_diagnostics.csv
    query_crops/
    contact_sheets/
    top_matches/
    review_clips/
  reports/
    event_metrics.json
    identity_metrics.json
    confusion_matrix.csv
```

## 14. 最终定位

本项目最终保持为清晰的推理工程：

```text
外部训练项目
  -> 输出模型权重
  -> 本项目接入权重、图库和视频
  -> 本项目完成推理、事件、身份、评估、诊断
  -> 本项目输出错误样例和改进建议
  -> 反馈给外部训练项目继续训练
```

项目内不放训练代码，不启动训练任务，不维护训练流水线。这里要做的是把模型用好、评估清楚、问题定位准确。
