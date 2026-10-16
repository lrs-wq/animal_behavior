# 猫咪进食行为识别正式项目

## 许可证与商业使用 / License and commercial use

**本仓库中维护者有权许可的原创内容采用 [PolyForm Noncommercial 1.0.0](LICENSE)，仅许可非商业用途；商业使用须另行取得书面授权。**
第三方内容、必须保留上游许可的部分及旧版本已授予的权利不受此变更覆盖。详见 [版权与第三方声明](COPYRIGHT.md)。

Original material the maintainer is entitled to license is available for noncommercial purposes under [PolyForm Noncommercial 1.0.0](LICENSE). Commercial use requires separate written permission. Third-party and previously granted rights are preserved; see [COPYRIGHT.md](COPYRIGHT.md).

本项目是猫咪进食行为识别的正式推理工程，负责接入外部训练好的模型，完成视频推理、进食事件分析、猫咪身份识别、结果评估和错误诊断。

> 当前版本：`0.1.0`
>
> 说明更新时间：`2026-09-30`
>
> 本次源码验证：`427` 项 Python 测试、`3` 项 Viewer 测试通过。

## 演示素材与代码同步

[演示说明](docs/demo/README.md) 提供一个可直接播放的标注视频和两张输出截图。
经授权的精选模型权重已收入 [models/weights](models/weights/README.md)，可随仓库克隆；其余图片、视频、数据集和运行文件不上传。

![猫咪行为标注示例](docs/demo/cat-standing.jpg)

阿里云 Codeup 继续作为默认 `origin`，GitHub 通过独立 remote 接收相同源码提交。
具体分支、上传范围及后续操作见 [源码同步约定](docs/source_sync.md)。

正式推理运行时**不执行模型训练**。`src/eating_behavior/` 只加载经过校验的外部模型产物并执行推理；生产 worker 不依赖训练数据、W&B 或训练框架入口。

仓库中的 `training/` 和姿态数据工具用于离线研发、回归实验和模型候选产出，与正式推理运行时隔离。训练结果不会自动进入生产配置；候选模型必须经过评估、复制到受控模型目录、计算 SHA-256，并显式更新 `configs/models.yaml` 后才能被推理端加载。

当前已实现的正式链路包括：视频文件/目录/摄像头/RTSP/HTTP 输入、猫和进食头检测、毛色分类、按家庭隔离的身份路由、进食状态机、猫在场提醒、可选 RTMPose 姿态推理、姿态时序过滤、实时叠加画面、批量标注视频以及 JSONL/CSV 事件输出。图库 embedding Re-ID、余粮模型和自动评估报告仍属于后续工作，配置中的相关字段只是预留接口。

## 已发布模型

本次模型包含现行检测、毛色分类、RTMPose，以及 v9 验证 mAP50-95 最优检测候选。
[模型选型与使用说明](docs/model_weights.md) 列出实测指标、选型范围和 SHA-256。
从仓库根目录运行时，使用 `configs/models.github.yaml` 读取本仓库权重；
需要对比检测候选时改用 `configs/models.github.best-map5095.yaml`。
服务器生产注册表仍为 `configs/models.yaml`。

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



## 快速开始

### 运行要求

- Python `>=3.10`；基础依赖见 `pyproject.toml` 和 `requirements.txt`。
- 服务器生产检测与毛色分类模型放在仓库外部，并在 `configs/models.yaml` 中登记绝对路径、SHA-256 和允许的模型根目录；仓库演示可用 `configs/models.github.yaml`。
- 启用 RTMPose 时，建议使用独立的 Python 3.10 + CUDA 12.1 环境，并按 `requirements/rtmpose-cu121.txt` 安装固定版本依赖。
- 批量输出带标注 MP4 时需要 `ffmpeg`。
- 当前发布包含四个精选模型权重及校验清单；数据集、运行输出和凭据不随源码分发。

安装基础环境：

```powershell
cd D:\gogelila\Eating_behavior\Eating_behavior_formal_project
python -m pip install -e .
```

运行测试：

```powershell
python -m pip install pytest
$env:PYTHONDONTWRITEBYTECODE="1"
python -B -m pytest -p no:cacheprovider -q
```

首次运行前至少检查：

1. `configs/models.yaml` 中两个模型路径、SHA-256 和 `security.allowed_roots`。
2. `configs/pipeline.yaml` 中 `sources`、`worker.source_ids` 和 `identity.households`。
3. 若启用姿态，将 `pose.enabled` 设为 `true`，并填写 RTMPose 配置与 checkpoint 的路径和 SHA-256。

调试本地视频并显示实时叠加画面：

```powershell
python -m eating_behavior.main `
  --source D:\path\to\video.mp4 `
  --source-id cam_debug_01 `
  --models-config configs/models.yaml `
  --pipeline-config configs/pipeline.yaml `
  --output-dir outputs/realtime_debug `
  --show
```

按 `q` 停止实时窗口。RTSP/HTTP 网络流必须显式传入稳定的 `--source-id`，程序不会把含凭据的流地址当作业务主键或写入事件。

生产 worker 根据配置快照运行：

```powershell
python -m eating_behavior.main `
  --models-config configs/models.yaml `
  --pipeline-config configs/worker_01.pipeline.yaml `
  --output-dir outputs/worker_01 `
  --device 0
```

批量处理 MP4 并生成标注视频与诊断汇总：

```powershell
python tools/run_batch_test.py `
  --input-root D:\path\to\videos `
  --output-root outputs\batch_20260819 `
  --models-config configs/models.yaml `
  --pipeline-config configs/pipeline.yaml `
  --limit 20 `
  --device 0
```

`--output-root` 必须是尚不存在的目录，防止旧结果与新结果混合。完整参数可通过 `python -m eating_behavior.main --help` 和 `python tools/run_batch_test.py --help` 查看。

## 0. 当前主业务链路

当前正式项目优先面向用户视频流实时推理，主入口已经兼容视频文件、视频文件夹、摄像头编号和 RTSP / HTTP 视频流。

第一版主链路为：

```text
用户视频/视频流
  -> 猫检测
  -> 毛色/外观类别确认
  -> 如果该毛色在用户家庭中只有一只猫，直接确认身份并结束身份分支
  -> 如果该毛色存在多只猫，拒绝强行确认并保留候选，等待后续 Re-ID
  -> 同步做 feeding_head 吃饭检测
  -> RTMPose 关键点质量门控与姿态状态判定
  -> 记录可播报事件
```

当前需要记录的可播报事件：

- `cat_passed`：兼容事件名，实际语义是画面内出现任意猫以及持续在场的周期提醒；默认每 30 分钟最多提醒一次，无法确认身份时 `cat_name` 留空。
- `cat_eating`：猫咪吃饭，附带猫咪身份；当前帧无法确认时参考最近身份时间线，仍无法确认则 `cat_name` 留空。
- `cat_posture_changed`：猫咪首次确认姿态或在吃饭、趴卧、站立、蹲姿之间切换；同一姿态不重复记录，猫消失或姿态质量不足时保留上一次状态。
- `bowl_food_status`：碗内余粮监控占位；当前默认配置 `emit_bowl_placeholder: false`，不会输出该事件。

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

中心数据库保存万级全量客户关系；每个推理进程只加载调度系统分配给自己的视频源小快照。当前单进程会按配置顺序处理视频源，不提供多路并发；生产并发应由外部调度多个 worker，并以真实压测确定单 worker 容量。

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

正式项目的目标是提供稳定、可维护、可评估的推理系统。下图是目标链路，其中图库 Re-ID 和自动评估/诊断尚未全部实现：

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

当前已落地的核心能力：

1. **进食事件检测**：输出吃饭开始时间、结束时间和持续时长。
2. **家庭范围身份路由**：在检测到进食后，按稳定视频源、家庭和毛色缩小候选；同毛色只有一只猫时确认身份，多只时保持 `unknown`，不强行给出猫名。

后续可扩展：

- 同毛色多猫的图库 embedding / Re-ID。
- 猫出现事件的整猫身份识别。
- 多猫切换事件切分。
- 评估报告自动生成。

## 2. 项目边界

### 2.1 正式推理运行时负责

- 加载经过路径和 SHA-256 校验的 YOLO 检测权重。
- 加载经过相同安全校验的毛色/外观分类模型。
- 对检测结果执行项目侧 class-aware NMS，避免端到端权重绕过 Ultralytics NMS 后出现重复框。
- 按稳定 `source_id` 和家庭配置完成毛色分流、单候选身份直判与同毛色多候选拒识。
- 对视频、文件夹、摄像头或流媒体做推理。
- 将帧级检测结果转换成进食事件。
- 将帧级身份结果聚合成事件级猫名。
- 可选执行 RTMPose、姿态质量门控、track 内关键点过滤和姿态状态判定。
- 输出结构化事件、实时叠加画面、批量标注视频和姿态诊断汇总。

图库 embedding Re-ID、余粮状态模型和完整自动评估报告尚未进入正式运行时，见第 12 节的待办项。

### 2.2 仓库内的离线研发工具

- `training/`：检测、分类、姿态训练，预测检查和超参数实验入口。
- `tools/` 中的数据工具：标签审计、质量筛选、审阅可视化和数据集构建。
- `docs/cat_pose_annotation_guide.md`：猫姿态数据的人工标注与质检规范。

这些工具遵守以下隔离规则：

- `src/eating_behavior/` 不导入 `training/` 或训练专用依赖。
- 生产 worker 镜像可以不包含 `training/`、训练数据和实验日志。
- 未列入 `models/weights/manifest.json` 的训练权重、W&B 目录、生成的数据 YAML 和数据集链接不提交到 Git。
- 只有经过独立评估并登记哈希的模型文件才能进入正式推理配置。

### 2.3 本仓库不负责

- 标注平台开发。
- 分布式训练集群和训练任务调度平台。
- 大规模训练数据管理。
- 模型制品仓库、长期归档和生产发布审批系统。

离线工具产出候选模型后，正式推理子系统只负责受控接入、推理、评估和反馈问题。

## 3. 当前实现

原型迁移已经完成到正式包结构，运行入口为 `eating_behavior.main:main`：

| 文件/目录 | 当前职责 |
| --- | --- |
| `src/eating_behavior/main.py` | 单路调试与多路 worker CLI，参数校验和配置加载 |
| `src/eating_behavior/pipeline.py` | 检测、毛色、身份、姿态、状态机与事件编排 |
| `src/eating_behavior/models.py` | YOLO 模型加载、制品路径/SHA-256 校验、检测结果转换和显式 NMS |
| `src/eating_behavior/identity.py` | 按 source/家庭隔离的猫注册表、身份决策和近期身份时间线 |
| `src/eating_behavior/eating_state.py` | 秒级进食进入/退出、短间隔合并和事件身份聚合 |
| `src/eating_behavior/presence_state.py` | 猫在场状态与周期提醒 |
| `src/eating_behavior/tracking.py` | source 内临时姿态 track；不作为猫身份 |
| `src/eating_behavior/pose.py`、`pose_filter.py`、`posture.py` | RTMPose 接入、质量门控、关键点时序过滤和姿态判定 |
| `src/eating_behavior/events.py` | 带 `record_id` 的 JSONL/CSV 增量记录、重试和进程内幂等 |
| `src/eating_behavior/visualization.py` | 实时与批量共用的检测框、毛色、姿态和骨架叠加 |
| `tools/run_batch_test.py` | 有界 MP4 批处理、H.264 标注视频、原音轨复用和姿态诊断汇总 |
| `web/stream-viewer/` | 独立的登录保护 HLS 观看页及 Nginx/systemd 部署样例 |

旧目录 `Eating_behavior_project` 仅作为历史原型参考，不是当前正式运行入口。

## 4. 设计原则

本节 4.2–4.5 描述后续图库 Re-ID 的设计约束。当前版本只实现毛色分类、家庭内单候选直判以及多候选拒识；尚未实现图库构建和 embedding 检索命令。

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

## 5. 当前项目结构

```text
Eating_behavior_formal_project/
  README.md
  pyproject.toml
  configs/
    models.yaml
    pipeline.yaml
    rtmpose/
  requirements/
    rtmpose-cu121.txt
  src/
    eating_behavior/
      main.py
      pipeline.py
      models.py
      identity.py
      eating_state.py
      presence_state.py
      association.py
      tracking.py
      pose.py
      pose_filter.py
      posture.py
      events.py
      media.py
      visualization.py
  tools/
    run_batch_test.py
    *pose*.py
  training/
    README.md
    rtmpose/
    train*.py
  docs/
    realtime_pipeline.md
    rtmpose_integration.md
    camera_stream_protocol_spec.md
    feeder_waterer_*.md
    cat_pose_annotation_guide.md
  tests/
    test_*.py
  web/
    stream-viewer/
  outputs/
```

目录说明：

- `configs/`：模型制品、安全边界、视频源、家庭猫信息和推理阈值。
- `src/eating_behavior/`：正式库代码。
- `tools/`：批量验证、姿态数据构建、审计和质检工具。
- `training/`：离线实验入口；不被正式运行时导入。
- `docs/`：实时链路、RTMPose、摄像头、采集和标注说明。
- `tests/`：正式运行时与离线工具的回归测试。
- `web/stream-viewer/`：独立部署的 HLS 观看端。
- `outputs/`：运行时生成目录，已被 Git 和发布压缩包排除。

## 6. 外部输入

### 6.1 模型权重

本项目通过 `configs/models.yaml` 读取外部模型。`.pt`/`.pth` 内部使用 pickle，只有在 `security.allow_pickle_weights: true`、路径位于 `security.allowed_roots` 且 SHA-256 完全匹配时才允许加载：

```yaml
detection_model: /root/Eating_behavior/models/detection_best.pt
detection_model_sha256: <64位十六进制摘要>
coat_classifier_model: /root/Eating_behavior/models/coat_classifier_best.pt
coat_classifier_model_sha256: <64位十六进制摘要>

pose:
  config: /root/Eating_behavior/models/rtmpose_config.py
  config_sha256: <64位十六进制摘要>
  checkpoint: /root/Eating_behavior/models/rtmpose_checkpoint.pth
  checkpoint_sha256: <64位十六进制摘要>

security:
  allow_pickle_weights: true
  require_sha256: true
  allowed_roots:
    - /root/Eating_behavior/models
```

`pose` 四个值在 `pose.enabled: false` 时可以留空。`identity_gallery`、`feature_backend` 和 `feature_model` 当前为未来 Re-ID 预留字段。

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

以下目录是后续 embedding Re-ID 的输入约定，当前版本尚无正式图库构建 CLI：

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

后续自动评估接口建议使用结构化 CSV：

```text
source
label_time_seconds
true_cat_name
feeding_start_seconds
feeding_end_seconds
scene_type
notes
```

后续评估导入工具也可以兼容历史原型中的 `0000.txt` 标签形式，但正式项目内部应统一转换为标准表。

### 6.5 Label Studio 视频时间轴标注项目

服务器上的 Label Studio 已建立独立视频标注项目，用于记录猫咪出现/消失和进食行为的时间区间：

| 项目 | 内容 |
| --- | --- |
| 项目名称 | `cat_behavior_video_20260602_05_20` |
| 项目 ID | `18` |
| 视频来源 | `/data/2026_06_02/05` 中按名称排序选取的 20 个 MP4 |
| 标注标签 | `presence`（出现到消失的在场区间）、`eating`、`drinking` |
| 辅助字段 | 标注完成/存在不确定片段、备注 |
| 项目入口 | `http://39.106.120.253/projects/18/data` |

视频通过 `/data/label_studio/local_files/` 下的硬链接提供给 Label Studio，不复制原始媒体；20 个任务均已通过 HTTP Range 分段读取验证。源视频是 H.265/HEVC，服务端读取正常，但标注浏览器仍需具备 HEVC 解码能力；无法播放时应先转码为 H.264。Label Studio 是外部标注环境，不是本仓库的运行依赖。

### 6.6 服务器数据盘存储

服务器 `39.106.120.253` 上的代码与大文件数据分开存放：

- Git 项目目录：`/root/Eating_behavior`
- 数据盘挂载点：`/data`
- 项目数据根目录：`/data/Eating_behavior`
- YOLO 检测数据集：`/data/Eating_behavior/v9_yolo26`

截至 2026-07-27 的扫描快照：

| 项目 | 扫描结果 |
| --- | --- |
| 块设备 | `/dev/vdb1` |
| 挂载点 / 文件系统 | `/data` / `ext4` |
| 总容量 | 1006.9 GiB |
| 已使用 | 288.2 GiB |
| 可用 | 667.4 GiB |
| 使用率 | 29% |
| `/data/Eating_behavior` | 约 29–30 GiB（目录仍可能被训练任务写入） |
| `v9_yolo26` 解压目录 | 约 12 GiB |
| `v9_yolo26.zip` 历史快照 | 约 11 GiB，2026-07-22 生成 |

`/data/Eating_behavior` 的主要内容如下：

| 路径 | 大小 / 说明 |
| --- | --- |
| `v9_yolo26/` | 约 12 GiB，当前检测训练数据集 |
| `v9_yolo26.zip` | 约 11 GiB，静态历史快照，不会随解压目录自动更新 |
| `cls_v2/` | 约 2.6 GiB |
| `test_outputs/` | 约 2.2 GiB |
| `rtmpose/` | 约 2.1 GiB |
| `yolo26_pose/` | 约 793 MiB |
| `cat_pose_annotation_seed_800/` | 约 415 MiB |
| `cat_pose_annotation_seed_230/` | 约 69 MiB |
| `cat_pose_annotation_seed_230_yolo_pose_before_export_20260629_173612/` | 约 67 MiB |
| `cat_pose_v1/` | 约 55 MiB |
| `cat_pose_v1_k4v1_to_v0/` | 约 55 MiB |
| `cat_pose_v2` | 指向 `/data/Eating_behavior/yolo26_pose/cat_pose_v2` 的符号链接 |

检测数据集采用标准 YOLO 目录结构：

```text
/data/Eating_behavior/v9_yolo26/
  data.yaml
  classes.txt
  build_summary.json
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  audits/
```

服务器端 `data.yaml` 使用绝对数据集根路径：

```yaml
path: /data/Eating_behavior/v9_yolo26
train: images/train
val: images/val
test: images/test
names:
  0: cat
  1: feeding_head
```

2026-07-28 完成三轮 RTSP211 难例差量同步后的有效文件计数为：

| split | images | labels | RTSP211 新样本 |
| --- | ---: | ---: | ---: |
| `train` | 3509 | 3509 | 19 |
| `val` | 1001 | 1001 | 4 |
| `test` | 500 | 500 | 2 |

数据盘维护约定：

- 生产使用的数据集、权重和推理产物放在 `/data/Eating_behavior`；只有经授权、逐文件列入清单的精选权重副本可在同步分支的 `models/weights/` 分发。
- 图片与标签必须按同一 stem 成对存放；空场景硬负样本也必须保留同名的 0 字节 `.txt` 标签。
- 差量同步必须携带 SHA256 清单，并在临时目录校验通过、确认目标无同名冲突后再落盘。
- 新增难例的审计图放在 `audits/`，增补清单放在数据集根目录；当前对应 `rtsp211_20260727*_manifest.json` 和 `rtsp211_20260728_round3_augmentation_manifest.json`。
- `labels/*.cache` 是 Ultralytics 生成的缓存，不是数据源；图片或标签变化后应由 Ultralytics 按内容哈希自动重建。
- `v9_yolo26.zip` 是静态历史快照。截至 2026-07-28 的差量不应被视为已自动写入该 ZIP；需要归档时必须重新生成并单独校验。

容量和文件数量会随训练与增量数据变化，可在服务器上重新检查：

```bash
findmnt -T /data/Eating_behavior
du -h --max-depth=1 /data/Eating_behavior | sort -h
for split in train val test; do
  find /data/Eating_behavior/v9_yolo26/images/$split -maxdepth 1 -type f | wc -l
  find /data/Eating_behavior/v9_yolo26/labels/$split -maxdepth 1 -type f -name '*.txt' | wc -l
done
```

## 7. 进食与 Re-ID 流程

当前已实现 7.2 的秒级进食状态机、事件身份聚合和内部近期身份时间线。7.1、7.3–7.5 是图库 Re-ID 与自动诊断的目标接口，尚不能通过独立 CLI 调用。

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

## 8. 后续 Re-ID 身份打分策略

历史原型使用：

```text
topk_mean
```

后续图库 Re-ID 需要支持参数化：

| 策略 | 说明 |
| --- | --- |
| `topk_mean` | 每只猫取 top-k 相似图库图平均，稳定但可能被相似图库拉偏 |
| `best_single` | 只取单张最近邻，局部信号强但容易抖 |
| `weighted_topk` | top1 权重更高，兼顾稳定和最近邻 |

银渐层等高相似猫建议优先对比 `best_single` 和 `weighted_topk`，并配合更严格 margin。

## 9. 当前配置文件

项目运行时只读取两个顶层配置文件：

| 文件 | 用途 |
| --- | --- |
| `configs/models.yaml` | 检测、毛色分类和可选 RTMPose 制品路径、SHA-256 与允许根目录 |
| `configs/pipeline.yaml` | 检测阈值、NMS、姿态、跟踪、关联、worker 视频源、家庭猫信息、事件和输出策略 |

常用配置块：

- `detection`：`confidence`、`iou`、项目侧 `nms_iou`、`imgsz` 和类别名。
- `coat`：分类输入尺寸、最小 crop 面积、置信度和 margin。
- `pose`：是否启用 RTMPose，以及质量门控、时序过滤和姿态规则。
- `tracking`：仅供姿态分支使用的 source 内临时 track 参数。
- `association`：`feeding_head` 与 cat 框的距离限制和歧义拒绝阈值。
- `worker` / `sources`：本 worker 允许处理的稳定摄像头 ID 与实际流地址。
- `identity.households`：家庭、摄像头与猫咪毛色注册关系。
- `events`：进食时间门控、身份历史、在场提醒和占位事件开关。
- `output`：事件文件开关与内存中保留的近期事件上限。

生产快照建议设置：

```yaml
worker:
  source_ids:
    - cam_customer_a_01
  require_assignments: true
  allow_debug_source: false
```

这样 worker 不能通过临时 CLI 参数扩大自己的视频源授权范围。原始 RTSP/HTTP 地址可能含凭据，不应提交到 Git 或打进分发压缩包；生产环境应在部署时注入 worker 专属配置快照。

## 10. 当前命令行

### 10.1 单路调试

```powershell
python -m eating_behavior.main `
  --source D:\path\to\video_or_directory `
  --source-id cam_debug_01 `
  --models-config configs/models.yaml `
  --pipeline-config configs/pipeline.yaml `
  --output-dir outputs/realtime_debug `
  --device 0 `
  --vid-stride 1 `
  --show
```

输入可以是视频文件、视频目录、摄像头编号、RTSP 或 HTTP 流。网络流必须提供 `--source-id`；一个 `--source` 最多绑定一个 ID。调试时可用 `--max-frames` 限制帧数。

### 10.2 配置驱动 worker

```powershell
python -m eating_behavior.main `
  --models-config configs/models.yaml `
  --pipeline-config configs/worker_01.pipeline.yaml `
  --output-dir outputs/worker_01
```

省略 `--source` 时读取 `worker.source_ids`。该列表为空则处理快照中全部 `sources`；也可以重复传入 `--source-id`，临时选择授权范围内的部分源。

### 10.3 批量验证

```powershell
python tools/run_batch_test.py `
  --input-root D:\path\to\mp4_root `
  --output-root outputs\batch_run_001 `
  --models-config configs/models.yaml `
  --pipeline-config configs/pipeline.yaml `
  --limit 20 `
  --device 0 `
  --progress-every 200
```

批处理递归选择排序后的前 N 个 `.mp4`，输出带框、毛色、姿态和骨架的 H.264 视频，并尽量复制原视频音轨。输出根目录已存在时会直接拒绝运行。

### 10.4 Web 观看端

`web/stream-viewer/` 是独立的 Node.js `>=22` 服务，只负责登录、会话和页面；Nginx 使用 `auth_request` 保护 MediaMTX 低延迟 HLS。部署前先执行：

```powershell
cd web\stream-viewer
npm run check
```

环境变量、scrypt 密码摘要、Nginx 和 systemd 示例见 `web/stream-viewer/README.md`。源码和部署样例不包含真实账号、密码或媒体服务凭据。

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

## 12. 实现状态与后续工作

截至 2026-08-19：

| 能力 | 状态 |
| --- | --- |
| 正式 Python 包、配置加载与 CLI | 已实现 |
| 模型路径、允许根目录和 SHA-256 校验 | 已实现 |
| 视频/目录/摄像头/RTSP/HTTP 输入 | 已实现 |
| `cat` / `feeding_head` 检测和项目侧 NMS | 已实现 |
| 毛色分类、家庭隔离和单候选身份直判 | 已实现 |
| 多候选拒识、近期同毛色身份复用和事件级聚合 | 已实现 |
| 进食状态机、猫在场提醒和姿态变化事件 | 已实现 |
| RTMPose、姿态 track、质量门控和时序过滤 | 已实现，可选启用 |
| JSONL/CSV 增量记录、`record_id` 与失败重试 | 已实现 |
| 实时叠加窗口与批量标注视频 | 已实现 |
| 登录保护的 HLS Web 观看端 | 已实现，独立部署 |
| 图库 embedding Re-ID | 待实现 |
| `bowl_food_status` 真实余粮模型 | 待实现；默认不发占位事件 |
| 自动指标、混淆矩阵和 review clip 报告 | 待实现 |
| 跨进程/重启后的持久化事件去重与待发恢复 | 待实现 |

后续实现图库 Re-ID 时，应先补齐 feature extractor、gallery builder、matcher、阈值评估和错误诊断，再把候选模型登记到受控制品配置中；不能让离线训练结果自动进入生产。

## 13. 输出目录约定

实时入口实际输出：

```text
outputs/
  realtime_debug/
    announcement_events.jsonl
    announcement_events.csv
```

批量工具实际输出：

```text
outputs/
  batch_run_001/
    effective_pipeline.yaml
    batch_summary.json
    events/
      announcement_events.jsonl
      announcement_events.csv
    videos/
      <保留输入相对目录>/<视频名>_annotated.mp4
```

事件采用增量追加方式写入，JSONL 顶层包含稳定 `record_id`，CSV 的 `details_json._record_id` 保存同一值。运行时只保留有界的近期事件尾部；下游仍应按 `record_id` 做幂等消费。`outputs/` 属于运行产物，不提交 Git，也不收入源码发布压缩包。

## 14. 分发压缩包约定

源码压缩包包含当前工作树中的代码、配置模板、测试、工具、训练入口、Web 观看端和 Markdown 文档，并保持顶层 `Eating_behavior_formal_project/` 目录。以下内容必须排除：

- `.git/`、`.codex_tmp/` 和编辑器/代理临时文件。
- `__pycache__/`、`.pytest_cache/`、`.mypy_cache/`、`.ruff_cache/`、`node_modules/`。
- `outputs/`、训练日志、预测结果和其他可再生成产物。
- 清单外的模型权重、数据集、embedding、真实流地址、账号密码、Cookie 密钥和其他凭据。

解压或克隆后可使用 `models/weights/` 中的精选权重和 `configs/models.github.yaml`。仍需安装依赖、准备自己的输入与 worker 配置，并保持校验。详见 [模型选型与使用说明](docs/model_weights.md)。

## 15. 最终定位

本项目保持为边界清晰的推理工程：

```text
外部训练项目
  -> 输出模型权重
  -> 经过校验后登记模型路径和 SHA-256
  -> 本项目接入权重、worker 配置和视频源
  -> 本项目完成推理、事件、身份路由、姿态和批量诊断
  -> 本项目输出错误样例和改进建议
  -> 反馈给外部训练项目继续训练
```

正式运行时不启动训练任务，也不依赖训练数据或 W&B。仓库内保留的 `training/` 只用于离线研发，必须与 `src/eating_behavior/` 和生产 worker 镜像隔离。这里要做的是把模型安全地接入、把事件稳定地输出、把问题准确地定位。
