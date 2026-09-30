# Eating Behavior 当前知识

## 目标

正式运行时接入视频文件、目录、摄像头编号或 RTSP/HTTP 视频流，完成猫检测、家庭范围内身份确认、进食/姿态判断和结构化事件记录。

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


## 当前主链路

```text
视频源
  -> 猫检测与跟踪
  -> 毛色/外观类别
  -> source_id 定位 household_id
  -> 家庭内唯一毛色直接确认，重复毛色进入 Re-ID 分支
  -> feeding_head 进食检测
  -> RTMPose 质量门与姿态判断
  -> 结构化事件
```

## 身份与多客户边界

- `stream_url` 只负责拉流，不作为长期业务主键。
- `source_id / camera_id` 是摄像头业务身份。
- 身份作用域是 `source_id → household_id → household cats`，不能跨家庭匹配。
- 同一家庭同一 `coat_class` 只有一只猫时直接确认；多只时暂时输出 unknown，并保留候选等待 Re-ID。
- worker 只加载调度给自己的小型配置快照，不加载全量客户关系。

## 当前事件语义

- `cat_passed`：画面出现猫以及持续在场周期提醒；身份不确定时 `cat_name` 为空。
- `cat_eating`：进食事件；当前帧身份不明时可参考最近身份时间线。
- `cat_posture_changed`：首次确定姿态或在吃饭、趴卧、站立、蹲姿之间切换。
- `bowl_food_status`：余粮监控占位，尚未形成完整生产模型和规则。

## 后端用户归属与事件发送

- `src/eating_behavior/alerts.py` 使用标准库提供 SQLite 登记、持久待发和独立 HTTP 发送进程；数据库位于 `/data/Eating_behavior/events/events.sqlite3`，运行代码位于 `/root/Eating_behavior`。
- schema v2 中 `users.user_id` 为正整数主键，独立 `username` 为管理用用户名；sources/outbox 的 user_id 同为整数。登记和外发拒绝字符串/浮点/布尔用户 ID。`users/sources` 保存真实用户与 RTMP/事件文件的关联，支持一用户多源，用户名、RTMP 和事件文件各自唯一。实际映射仅在运行数据库中保存，不进入知识库。
- 使用 `--source-db` 的直播入口读取登记的 RTMP，校验输出事件路径，并在私有状态文件报告映射版本；发送器要求生产者的源身份、地址和版本与数据库完全匹配。
- `migrate-users` 接收显式旧用户键到数字 ID/用户名的映射，从 v1 备份并事务升级到 v2；保留源版本、事件 ID 和游标，只修正未尝试或明确鉴权拒绝的活跃载荷，结果不确定的活跃事件先对账，终态原载荷保留。只修改用户名不重置绑定或队列；实际客户映射只在运行环境保存。
- 首次或变更登记从完整行末尾接入，不自动发送历史事件；相同绑定重复执行不改变游标。变更/禁用会取消旧待发并要求生产者重启，存在有效在途请求时拒绝变更。
- JSONL 写入增加 flush/fsync；发送器将读取游标和不可变三字段请求体放在同一 SQLite 事务内，内部使用原 `record_id` 作为 `event_id`，保留请求重试的用户归属。HTTP 边界将内部 `event_id/user_id/event_type` 转换为用户确认的 `eventId/userId/eventType` JSON；日常发送不重写原有队列或重新生成 ID；外发 userId 是 JSON 整数，不发送 username。
- 仅映射 `cat_passed → cat_appeared` 与四种 `cat_posture_changed`；不推送内部完整进食区间或余粮占位。
- `eating-behavior-events.service` 独立运行，读取 `/etc` 受控环境凭据，验证 HTTPS，禁止重定向；成功要求 HTTP 200/201、`success=true` 且回执 `eventId` 相同；若含业务 code，先确认整数 200/201，再校验顶层或 data 中的回执，不能仅依 HTTP 状态或业务码认定成功。具体地址及运维过程见 `docs/backend_event_delivery.md`。
- 网络异常及 HTTP 或业务 code 的 408/429/5xx 退避重试；401/403 鉴权失败或响应契约不符持久暂停 HTTP，继续导入；409/422 终止该条重试并保留记录。HTTP 200/201 中业务 code=400/404/405 或未知/非整数码触发契约暂停，避免误报成功或批量丢弃待发。租约和单进程文件锁用于重启恢复，同一来源按序发送。
- 文件替换/截断、生产者版本不匹配及活跃待发上限触发停止导入，保留游标；非法完整行记录脱敏错误位置并跳过。禁止未经对账清理事件文件或切换生产库的接口地址。
- 本实现不包含推理状态机检查点，也不保证不同事件 ID 的语义去重；已发送记录与 JSONL 的归档、容量监控、后端接收验收和长时间对账仍需完善。实际发送是否成功以实时队列和后端回执为准。

## 模型治理

- 正式 `src/eating_behavior/` 运行时不训练模型。
- `training/`、姿态数据工具和 W&B 属于离线研发。
- 训练产物不能自动进入生产；必须评估、复制到受控目录、计算 SHA-256，并显式更新 `configs/models.yaml`。
- 当前生产注册表校验检测模型，并以配置与 checkpoint 双 SHA-256 固定 RTMPose；生产权重保留受控目录；经用户授权的四个精选副本在发布分支 models/weights 中分发，其他权重不进入 Git。

## 离线帧准备

- `tools/extract_every_50_frames.py` 对视频树进行探测，默认每 50 个源帧抽取一张 JPEG，并用逐视频完成标记、原子发布和总清单支持恢复与校验。
- `tools/flatten_extracted_frames.py` 将已抽取帧以硬链接或复制方式发布到扁平目录，同时生成可回溯源相对路径的 CSV 映射和清单。
- 原始视频、抽取帧、工作目录以及包含源文件名或绝对路径的清单都是本地数据产物，必须位于 Git 工作树外或受忽略目录中。

## 配置与查看器

- `configs/pipeline.yaml` 控制视频源、worker 分配、身份映射和运行参数。
- `configs/models.yaml` 指向受控模型资产及其校验信息。
- `web/stream-viewer/` 提供带认证的多摄像头实时查看能力。
- camera02 的 AI 分析入口默认关闭；只有部署显式开启功能开关时，Viewer 才返回分析流元数据和状态接口。
- 开始任务时必须查看 `kg context` 的实时文件列表，避免与正在进行的配置、部署、推理或 viewer 修改冲突。

## camera02 实时分析链路

- `tools/run_live_annotated_recording.py` 复用正式 `RealtimePipeline`，可同时输出分段录像和 RTMP 标注流。
- `eating-behavior-camera02-ai-live.service` 只发布 `live/camera02-ai`，不保存视频分段；原始 `live/camera02` 保持独立。
- AI 直播单元每 3 个输入帧运行一次新推理，连接或重连后的首帧强制推理；中间帧复用最近一次检测、姿态和事件分析结果进行渲染。
- 因为状态机只在新推理时调用，推理间隔会同时改变检测刷新和事件状态推进频率，不能当作纯显示参数。
- AI 直播的状态、有效配置和事件文件保存于独立数据盘目录 /data/Eating_behavior/runtime/camera02-ai/；启动依赖 /data 挂载，以 root:joyai-viewer、0750 创建目录，停止服务不删除输出。文件持久化由事件发送器补充 SQLite 待发恢复；推理状态机仍不提供跨进程恢复。
- Viewer 只读取 `/data/Eating_behavior/runtime/camera02-ai/status.json`，对外返回字段白名单；源地址、事件内容和其他运行明细不进入 HTTP 响应。
- `eating-behavior-camera02-annotated.service` 是独立的连续分段录像单元，默认逐帧推理，不与无录像 AI 直播单元混用。

## 运行时与训练边界

- 生产 worker 不依赖训练数据、W&B 或训练入口。
- 原始视频、客户流地址、模型权重和生产事件内容不能进入知识图谱。
- 推理间隔、事件节流和状态保持变化属于语义变化，必须更新图谱或当前知识。

实时分支、HEAD、未提交文件和知识漂移状态由 `kg context` 动态生成。

## 源码分发与演示

- 阿里云 Codeup 保持默认 `origin` 与既有分支 upstream；GitHub 使用独立 `github` remote。
- 2026-09-30 同步分支汇总服务器正式版本与本地统一可视化及采集文档，向 Codeup 同名分支与 GitHub `main` 发布同一提交；运行服务的源码目录保持 `/root/Eating_behavior`。
- 仓库媒体采用逐文件白名单，仅分发 `docs/demo/` 的少量标注演示结果；原始媒体、训练数据、清单外权重、运行事件、用户映射及凭据保留在仓库外。操作依据见 `docs/source_sync.md` 与 ADR 0006。
- `src/eating_behavior/visualization.py` 统一实时 `--show` 与批量视频的检测框、毛色/姿态标签、骨架及进食头关联线。标签优先使用确认姿态，其次使用当前观察；无法判断时显示 unknown。渲染不推进事件状态机。
- 当前本机 Git SSH 通过已授权的阿里云服务器转发，使用服务器既有 Git 认证；默认推送仍为 origin，本机研发分支与同步分支均跟踪 Codeup。转发配置保存在本机 .git，不随源码分发；运行时推理没有新增依赖。

## 精选模型分发

- 按用户于 2026-09-30 上传精选训练权重的要求，四个副本进入发布仓库；models/weights/manifest.json 与 SHA256SUMS 固定身份、大小和哈希。二进制不进入知识文件。
- 分发当前生产检测、v9 验证 mAP50-95 最优检测候选、现行框架 best 毛色分类器，以及真实机位适配 RTMPose 第二阶段 epoch 140；指标和评测局限见 docs/model_weights.md。不能跨数据集凭单个数值宣称全局最优。
- configs/models.github.yaml 在仓库根目录加载现行组合；configs/models.github.best-map5095.yaml 只替换检测候选。两者保持显式允许目录、配置与 checkpoint SHA-256。
- 生产 models.yaml 与 /data 中实际资产不自动改变；训练推广仍遵循评估、受控复制和显式注册。Codeup 跟踪与同提交 GitHub 同步规则保持。
