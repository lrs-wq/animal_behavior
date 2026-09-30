# 猫咪进食行为识别正式项目

本项目是猫咪进食行为识别的正式推理工程，负责接入外部训练好的模型，完成视频推理、进食事件分析、猫咪身份识别、结果评估和错误诊断。

正式推理运行时**不执行模型训练**。`src/eating_behavior/` 只加载经过校验的外部模型产物并执行推理；生产 worker 不依赖训练数据、W&B 或训练框架入口。

仓库中的 `training/` 和姿态数据工具用于离线研发、回归实验和模型候选产出，与正式推理运行时隔离。训练结果不会自动进入生产配置；候选模型必须经过评估、复制到受控模型目录、计算 SHA-256，并显式更新 `configs/models.yaml` 后才能被推理端加载。

## 后端报警与用户视频源映射

`src/eating_behavior/alerts.py` 提供 SQLite 用户/RTMP 登记和独立后台发送服务。
直播生产者通过 `--source-db` 读取绑定视频源，发送端从持久 JSONL 导入白名单事件，
事务保存游标与待发请求，使用原 `record_id` 幂等重试。数据库位于
`/data/Eating_behavior/events/events.sqlite3`，凭据由 `/etc/eating-behavior-events.env` 注入。
新增真实用户、服务安装、鉴权暂停恢复和可靠性边界见 [后端报警发送运维文档](docs/backend_event_delivery.md)。

## 服务器代码与数据目录约定

| 类别 | 实际保存位置 |
| --- | --- |
| 项目运行代码、脚本、配置模板、知识库 | /root/Eating_behavior |
| 可执行 RTMPose Python 配置 | /root/Eating_behavior/models/rtmpose |
| 受 SHA-256 校验的模型权重 | /data/Eating_behavior/models |
| 媒体、数据集、事件、持久状态、测试输出、缓存和备份 | /data/Eating_behavior 下的对应目录 |
| AI 直播输出 | /data/Eating_behavior/runtime/camera02-ai |

Viewer 的代码源为 /root/Eating_behavior/web/stream-viewer。systemd 将其只读映射到
服务私有的 /opt/joyai-stream-viewer，保留专用用户与 ProtectHome；该 /opt 路径
仅是加载入口，不另行维护部署代码副本。修改代码后进行检查并重启 Viewer 即可生效。

Python 环境 /opt/eating-behavior/conda-env、系统 Node 与 /etc 下的服务配置/凭据
属于基础设施，沿用系统目录。/root/Eating_behavior/data 是指向数据盘的目录链接。
生产或手动运行服务器脚本时必须显式将输出和缓存指向 /data，不依赖源码目录下的
相对 outputs 默认值。路径迁移不得改变模型内容或取消校验。


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
  -> RTMPose 关键点质量门控与姿态状态判定
  -> 记录可播报事件
```

当前需要记录的可播报事件：

- `cat_passed`：兼容事件名，实际语义是画面内出现任意猫以及持续在场的周期提醒；默认每 30 分钟最多提醒一次，无法确认身份时 `cat_name` 留空。
- `cat_eating`：猫咪吃饭，附带猫咪身份；当前帧无法确认时参考最近身份时间线，仍无法确认则 `cat_name` 留空。
- `cat_posture_changed`：猫咪首次确认姿态或在吃饭、趴卧、站立、蹲姿之间切换；同一姿态不重复记录，猫消失或姿态质量不足时保留上一次状态。
- `bowl_food_status`：碗内余粮监控占位，后续补充具体模型和规则。

播报方式暂不在第一版实现中固定，当前只负责把事件结构化记录到文件，后续可以接语音、App 推送、WebSocket 或其他消费端。

## 0.1 多客户 `source_id` / `camera_id` 对应逻辑

万级客户规模下，推理进程不能把所有客户配置加载到本地，也不能把原始 RTSP / HTTP 视频流地址当成长期身份主键。本项目采用以下原则：

```text
stream_url 只负责拉流
source_id / camera_id 才负责业务身份对应
```

推荐业务关系为：

```text
user_id
  -> household_id
      -> source_id / camera_id
          -> stream_url
      -> cats
          -> cat_name
          -> coat_class
```

推理 worker 本地只需要一份小快照：

```text
source_id -> stream_url
source_id -> household_id
household_id -> cats
```

中心数据库保存万级全量客户关系；每个推理进程只加载调度系统分配给自己的 20 路以上视频源数据。

### 0.1.1 Worker 本地配置快照

生产环境建议由调度服务或配置服务从中心库生成 worker 专属 `pipeline.yaml`：

```yaml
worker:
  source_ids:
    - cam_customer_a_01
    - cam_customer_b_01

sources:
  - source_id: cam_customer_a_01
    stream_url: rtsp://customer-a.example/live
  - source_id: cam_customer_b_01
    stream_url: rtsp://customer-b.example/live

identity:
  households:
    - household_id: hh_customer_a
      source_ids:
        - cam_customer_a_01
      cats:
        - name: 小蓝
          coat_class: blue_cat

    - household_id: hh_customer_b
      source_ids:
        - cam_customer_b_01
      cats:
        - name: 小蓝
          coat_class: blue_cat
```

这里两个客户都可以有名为“小蓝”的猫，也可以拥有相同 `coat_class`。系统会先用当前帧的 `source_id` 找到家庭，再只在这个家庭内按 `coat_class` 匹配猫咪，不会跨客户混淆。

### 0.1.2 启动方式

调试单路视频流：

```powershell
$env:PYTHONPATH="D:\gogelila\Eating_behavior\Eating_behavior_formal_project\src"

python -m eating_behavior.main `
  --source "rtsp://debug.example/live" `
  --source-id "cam_debug_01" `
  --pipeline-config configs/pipeline.yaml `
  --models-config configs/models.yaml `
  --output-dir outputs/realtime_debug
```

生产 worker 启动：

```powershell
$env:PYTHONPATH="D:\gogelila\Eating_behavior\Eating_behavior_formal_project\src"

python -m eating_behavior.main `
  --pipeline-config configs/worker_01.pipeline.yaml `
  --models-config configs/models.yaml `
  --output-dir outputs/worker_01
```

不传 `--source` 时，程序会读取配置中的 `worker.source_ids`，只运行本 worker 被分配的摄像头。如果 `worker.source_ids` 为空，则运行该快照中 `sources` 下的全部视频源。临时指定部分摄像头时，可以重复传入或逗号分隔：

```powershell
python -m eating_behavior.main `
  --pipeline-config configs/worker_01.pipeline.yaml `
  --source-id cam_customer_a_01 `
  --source-id cam_customer_b_01
```

### 0.1.3 帧级对应链路

每一帧的对应关系如下：

```text
SourceSpec(source_id, stream_url)
  -> 用 stream_url 打开视频流
  -> FramePacket.source = source_id
  -> 猫检测和花色分类
  -> identity_service.decide(source_id, coat_class)
  -> source_id 找 household_id
  -> household_id 内按 coat_class 找 cat_name
  -> 输出事件 source_id + household_id + cat_name
```

当前身份判断规则：

```text
同一家庭内该 coat_class 只有一只猫
  -> 直接返回这只猫的 cat_name

同一家庭内该 coat_class 有多只猫
  -> 返回内部 unknown
  -> 事件 cat_name 留空
  -> details 中保留 identity_candidates，等待后续 Re-ID 分支确认

该 source_id 没有匹配到家庭
  -> 返回内部 unregistered_source
  -> 事件 cat_name 留空
```

事件输出中：

- `source`：稳定 `source_id` / `camera_id`，不输出原始流地址。
- `household_id`：客户家庭 ID，便于下游按客户聚合。
- `cat_name`：确认身份时输出猫名；无法确认时为空字符串。
- `details.identity_method`：记录身份来源和失败原因，例如 `direct_by_single_same_coat`、`needs_identity_confirmation`、`unregistered_source`。

身份时间线和事件限频都按 `source_id` 隔离。即使 A 客户和 B 客户的猫同名、同花色，也不会互相复用历史身份或互相吞掉事件。

### 0.1.4 数据库存储建议

数据库建议保存长期关系，字段可以按以下结构设计：

```text
users
  user_id
  status
  created_at
  updated_at

households
  household_id
  user_id
  name
  status
  created_at
  updated_at

cameras
  camera_id
  household_id
  stream_url
  status
  config_version
  updated_at

cats
  cat_id
  household_id
  cat_name
  coat_class
  status
  updated_at

worker_assignments
  worker_id
  camera_id
  assigned_at
  config_version
```

推荐索引：

```sql
CREATE UNIQUE INDEX idx_cameras_camera_id ON cameras(camera_id);
CREATE INDEX idx_cameras_household_id ON cameras(household_id);
CREATE INDEX idx_cats_household_id ON cats(household_id);
CREATE INDEX idx_worker_assignments_worker_id ON worker_assignments(worker_id);
CREATE UNIQUE INDEX idx_worker_assignments_worker_camera ON worker_assignments(worker_id, camera_id);
```

`stream_url` 建议加密存储，或由流媒体/鉴权服务在 worker 启动和刷新配置时临时签发。下游事件、日志和诊断文件不应暴露原始视频流地址。

### 0.1.5 数据库性能边界

数据库不应参与逐帧推理热路径。正确方式是：

```text
worker 启动
  -> 按 worker_id 从数据库或配置服务拉取小快照
  -> 构建内存字典
  -> 推理过程中只查内存
```

推理热路径只做内存查询：

```text
source_id -> household registry
coat_class -> candidate cats
```

数据库访问只发生在：

- worker 启动加载配置。
- 定时刷新配置，例如 30 秒或 60 秒一次。
- 收到配置变更通知后增量刷新。
- worker 异常重启后重新拉取快照。

因此数据库规模可以按万级客户设计，但每个推理进程内存中只持有自己负责的几十路摄像头数据。只要不在每帧中查库，数据库不会影响推理性能。

10 万路摄像头的容量和成本估算见 [docs/camera_capacity_cost_estimate.md](docs/camera_capacity_cost_estimate.md)。该估算基于每路 0.5FPS 候选图片全量检测、审核资源和业务冗余，正式采购前必须以真实压测结果修正。

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

- 猫出现事件的整猫身份识别。
- 品种/外观类型分流。
- 多猫切换事件切分。
- 评估报告自动生成。

## 2. 项目边界

### 2.1 正式推理运行时负责

- 加载经过路径和 SHA-256 校验的 YOLO 检测权重。
- 加载外部训练好的分类模型或 embedding 模型。
- 构建和读取猫咪身份图库 embedding。
- 对视频、文件夹、摄像头或流媒体做推理。
- 将帧级检测结果转换成进食事件。
- 将帧级身份结果聚合成事件级猫名。
- 输出结构化结果、可视化视频、错误诊断素材。
- 对外部模型版本做离线评估和回归检查。

### 2.2 仓库内的离线研发工具

- `training/`：检测、分类、姿态训练，预测检查和超参数实验入口。
- `tools/` 中的数据工具：标签审计、质量筛选、审阅可视化和数据集构建。
- `docs/cat_pose_annotation_guide.md`：猫姿态数据的人工标注与质检规范。

这些工具遵守以下隔离规则：

- `src/eating_behavior/` 不导入 `training/` 或训练专用依赖。
- 生产 worker 镜像可以不包含 `training/`、训练数据和实验日志。
- 训练权重、W&B 运行目录、生成的数据 YAML 和数据集链接不提交到 Git。
- 只有经过独立评估并登记哈希的模型文件才能进入正式推理配置。

### 2.3 本仓库不负责

- 标注平台开发。
- 分布式训练集群和训练任务调度平台。
- 大规模训练数据管理。
- 模型制品仓库、长期归档和生产发布审批系统。

离线工具产出候选模型后，正式推理子系统只负责受控接入、推理、评估和反馈问题。

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

这里的 `unknown` 是内部识别状态，用于诊断、评估和阈值策略。对外可播报事件 `cat_passed` / `cat_eating` / `cat_posture_changed` 的 `cat_name` 字段在无法确认身份时输出为空字符串。

高相似猫，如银渐层、蓝猫、狸花、美短虎斑，应使用更严格的 margin 和 unknown 策略。

### 4.4 头部身份和整猫身份分开

第一阶段主做吃饭身份识别：

```text
feeding_head -> head_gallery -> eating identity
```

后续扩展出现事件的整猫身份识别：

```text
cat / side_body -> body_gallery -> presence identity
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
- 猫出现事件的整猫身份识别。
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
