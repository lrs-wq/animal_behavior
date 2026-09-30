# 后端报警发送与用户视频源登记

实现入口为 `src/eating_behavior/alerts.py`，使用 Python 标准库，不加载推理模型。
正式运行代码在 `/root/Eating_behavior`；SQLite、事件和运行状态都在 `/data/Eating_behavior`。

## 发送链路

```text
SQLite sources：source_id → user_id + RTMP + event_file + revision
   ↓ 直播启动时读取
run_live_annotated_recording.py --source-db ... --source-id ...
   ↓ EventRecorder 写入、flush、fsync
runtime/<实例>/events/announcement_events.jsonl
   ↓ 独立发送进程读取完整行，校验生产者 source / RTMP / revision
SQLite：文件游标与不可变请求体在同一事务中保存
   ↓ 固定 event_id 重试
HTTPS POST → 后端按 event_id 幂等接收
```

默认数据库为 `/data/Eating_behavior/events/events.sqlite3`，目录权限 `0700`、数据库权限 `0600`。
`users` 保存 `user_id INTEGER PRIMARY KEY`、`username TEXT NOT NULL UNIQUE` 和创建时间；
`user_id` 是后端真实数字 ID，`username` 仅用于管理展示，不能互相替代。`sources.user_id` 与 `outbox.user_id` 也保存整数，`sources` 保存视频源归属和拉流地址。一个用户可以对应多路视频源；
同一个 RTMP 地址或事件文件只能登记一次。未登记或已禁用的源不会自动发送。
该映射用于后端用户归属，不替代现有 `source_id → household_id → cats` 猫身份配置。

其余表为 `cursors`（JSONL 读取位置）、`outbox`（待发与发送结果）、
`settings`（接口地址及持久暂停状态）、`ingest_errors`（脱敏导入错误位置）。
SQLite 使用 WAL、FULL 同步、外键、写事务和发送租约；同一数据库仅允许一个发送服务持有文件锁。
数据库、WAL/SHM、真实映射、凭据、事件及备份均不进入 Git 或知识库。

## 接口契约

实际接口为 `POST https://app.hzzootech.com/api/open/eatingBehavior/events`，
认证头为 `X-API-Key`，内容类型为 `application/json`。2026-09-07 按用户确认的实际契约，外发字段使用驼峰命名，仅发送以下三个字段：

```json
{"eventId":"<原 record_id>","userId":20001,"eventType":"cat_appeared"}
```

`eventId` 和 `eventType` 是字符串，`userId` 必须是正整数 JSON number，例如 `20001`，不能是 `"20001"` 或用户名。
API 仅发送这三个字段，不发送 `username`。登记和发送端均拒绝字符串、浮点数、布尔值、非正数及超出 SQLite 整数范围的用户 ID。

| 内部事件 | 后端 eventType |
| --- | --- |
| `cat_passed` | `cat_appeared` |
| `cat_posture_changed`，`posture=eating` | `cat_eating` |
| `cat_posture_changed`，`posture=lying` | `cat_lying` |
| `cat_posture_changed`，`posture=standing` | `cat_standing` |
| `cat_posture_changed`，`posture=crouching` | `cat_crouching` |

完整进食区间 `cat_eating`、余粮占位和其他事件不推送。猫名、流地址、图片、视频和内部时间戳不外发。
只有 HTTP `200/201` 且回执中 `success` 严格为 `true`、`eventId` 与请求一致，才标记成功。
如响应含业务 `code`，须先检查其为整数 `200/201`，再校验顶层或 `data` 中的成功回执；仅有业务码成功或 `data=null` 不足以确认接收。

SQLite 中的不可变事件载荷继续使用 `event_id/user_id/event_type`。发送时分别转换为
`eventId/userId/eventType`，日常重试不重写载荷、不重新生成 ID。请求体不随新用户映射改写；同一 ID 的冲突内容会记录为导入错误。
显式的 v1 → v2 用户语义迁移是例外：只修正未尝试或明确鉴权拒绝的活跃事件，见下方迁移说明。

## 部署与检查

先初始化数据库并登记视频源，再启动使用 `--source-db` 的直播生产者与发送服务。
在服务器执行以下命令，后续示例默认已设置这些环境变量：

```bash
cd /root/Eating_behavior
export PYTHONPATH=/root/Eating_behavior/src:/root/Eating_behavior
export PYTHONDONTWRITEBYTECODE=1
EVENT_PYTHON=/opt/eating-behavior/conda-env/bin/python
install -d -m 0700 /data/Eating_behavior/events
"$EVENT_PYTHON" -B -m eating_behavior.alerts init
```

凭据位于 `/etc/eating-behavior-events.env`，所有者 root、权限 `0600`。
通过受控方式填写，禁止将真实密钥写入源码、命令行参数、测试或日志：

```ini
EATING_BEHAVIOR_API_KEY=<实际密钥>
EATING_BEHAVIOR_EVENTS_DB=/data/Eating_behavior/events/events.sqlite3
EATING_BEHAVIOR_EVENTS_URL=https://app.hzzootech.com/api/open/eatingBehavior/events
```

完成源登记后安装发送单元：

```bash
install -m 0644 deploy/systemd/eating-behavior-events.service /etc/systemd/system/
systemd-analyze verify /etc/systemd/system/eating-behavior-events.service
systemctl daemon-reload
systemctl enable --now eating-behavior-events.service
"$EVENT_PYTHON" -B -m eating_behavior.alerts list
"$EVENT_PYTHON" -B -m eating_behavior.alerts status
systemctl status eating-behavior-events.service
```

`list` 显示数字用户 ID、用户名、启用状态和版本，不显示 RTMP 地址。`status` 显示各队列状态数量、
暂停原因、最旧待发年龄和导入错误数。发送日志只记录 ID、状态及脱敏错误分类，不记录请求头、
原始响应或完整事件。需要将 `dead`、暂停、导入失败和队列增长纳入运维监控。

## 添加和调整用户对应关系

下列 `CAMERA_ID`、`NUMERIC_USER_ID`、`USER_NAME`、`INSTANCE` 均需替换为实际值；`NUMERIC_USER_ID` 填正整数。
使用 Bash 的静默输入，避免将带凭据的流地址写入 shell 历史或外部进程参数：

```bash
read -r -s -p 'RTMP URL: ' EVENT_RTMP
printf '%s\n' "$EVENT_RTMP" | "$EVENT_PYTHON" -B -m eating_behavior.alerts bind \
  --source-id CAMERA_ID --user-id NUMERIC_USER_ID --username USER_NAME --rtmp-url-stdin \
  --event-file /data/Eating_behavior/runtime/INSTANCE/events/announcement_events.jsonl
unset EVENT_RTMP
```

对应直播实例使用 `--source-db /data/Eating_behavior/events/events.sqlite3 --source-id CAMERA_ID`
替代原来的 `--source` 参数，并将 `--output-root` 指向 `/data/Eating_behavior/runtime/INSTANCE`。
其他模型、推理、发布参数沿用该实例配置。数据库检查会拒绝输出目录不匹配的启动。
登记用户不会自动创建额外推理进程，多路服务仍需单独配置并检查 GPU 容量。

首次登记从事件文件最后一个完整行的末尾开始，不自动补发历史事件。
完全相同的重复登记不清空队列、不重置游标。只更新同一数字用户 ID 的用户名，也不会改变源版本或待发载荷。
改变用户、RTMP 或事件文件会增加映射版本、取消该源旧的待发记录，并重新从文件末尾接入。
变更时应先停止对应生产者和发送服务，待在途请求结束后登记，再启动生产者和发送服务。
新生产者在 `status.json` 报告相同映射版本后才允许导入；旧版本生产者的事件不会被归给新用户。
生产者停止到变更生效之间尚未导入的旧事件不会自动补发，变更前应检查积压。

禁用来源使用以下命令；存在有效在途租约时会拒绝变更，应等待请求结束或租约到期：

```bash
"$EVENT_PYTHON" -B -m eating_behavior.alerts disable --source-id CAMERA_ID
```

## v1 用户字段迁移到 v2

v1 曾将登录名保存在字符串 `user_id` 中；v2 将数字 ID 和用户名分开，数据库 `PRAGMA user_version=2`。
旧库不会凭用户名自动推断数字 ID。运行 `init` 或普通命令遇到 v1 时会要求显式迁移。

先停止发送器。现有 AI 生产者可继续写 JSONL，但如需重新启动 AI，应等数据库迁移完成后再启动。
从后端确认所有旧用户键对应的数字 ID 和用户名，通过标准输入提供映射；不要将真实映射写进仓库或知识库。
以下是占位示例：

```bash
systemctl stop eating-behavior-events.service
"$EVENT_PYTHON" -B -m eating_behavior.alerts migrate-users <<'JSON'
{"LEGACY_USER_KEY":{"user_id":20001,"username":"USER_NAME"}}
JSON
systemctl start eating-behavior-events.service
"$EVENT_PYTHON" -B -m eating_behavior.alerts list
"$EVENT_PYTHON" -B -m eating_behavior.alerts status
```

迁移要求映射恰好覆盖旧库所有用户，数字 ID 和用户名各自唯一；持有发送器文件锁时会拒绝迁移。
迁移在数据库写事务下生成 SQLite 一致性备份 `events.v1-backup-<唯一标识>.sqlite3`，再重建三个用户相关表、校验外键并提交。
失败会回滚，备份仍可用于核对；备份包含真实客户数据，权限 `0600`，保存在数据目录。

源、RTMP、事件文件、源版本、文件游标、事件 ID、顺序和暂停状态均保留。
只对 `pending` 且未尝试、或已明确收到鉴权拒绝的 `blocked_auth` 活跃事件纠正数字用户 ID；
`retry/inflight/blocked_contract` 等结果不确定的活跃记录必须先对账，迁移不会猜测其是否已被后端接收。
已完成、已取消和已隔离事件的原始载荷保留为历史证据，不自动重新投递。
仅修改用户名不需要运行本迁移，使用同一数字 ID 重新 `bind --username ...` 即可。

迁移不会自动恢复 HTTP。先用正确数字用户 ID 核实后端回执，再检查积压并按故障恢复步骤执行 `resume`。
回退 schema v2 时必须同时恢复相匹配的代码与数据库，并保留迁移后新增数据用于对账，不能直接用旧代码写入 v2。

## 故障恢复

| 情况 | 行为与处理 |
| --- | --- |
| 网络异常、HTTP 408/429/5xx，或 HTTP 200/201 中业务 code=408/429/5xx | 保留同一 ID 自动退避重试；支持有上限的 `Retry-After` |
| HTTP 401/403，或 HTTP 200/201 中业务 code=401/403 | 全局持久暂停 HTTP；修复鉴权后显式恢复 |
| 404、405、重定向、非契约成功响应 | 全局持久暂停；确认 URL、网关与响应契约后恢复 |
| HTTP 或业务 code 为 409/422 | 该事件标记 `dead`，保留记录；后续事件可继续，需要人工核对 |
| HTTP 200/201 中业务 code=400/404/405、其他未知业务码或非整数业务码 | 契约暂停，核对参数、路由和业务含义；不因 HTTP 200 而标记成功 |
| 发送进程中断 | 启动后恢复未完成队列，过期在途租约使用原 ID 重试 |
| 映射或生产者版本不一致 | 拒绝导入、保留游标，核对登记并重启对应生产者 |
| JSONL 最后一行不完整 | 等待生产者完成；不提前推进游标 |
| 完整行非法或 ID 冲突 | 记录脱敏错误位置并跳过；排查原始 JSONL，不把内容复制到知识库 |
| 文件替换、截断、游标锚点变化 | 停止导入并保留游标，不猜测轮转位置 |
| 活跃待发达到 100000 条 | 暂停导入并保留游标；排查积压及磁盘容量 |

发送暂停时继续导入新事件，重启服务不会自动解除暂停。凭据修复如涉及环境文件，
先重启发送服务，再检查积压年龄，最后恢复：

```bash
systemctl restart eating-behavior-events.service
"$EVENT_PYTHON" -B -m eating_behavior.alerts status
"$EVENT_PYTHON" -B -m eating_behavior.alerts resume
```

`resume` 会恢复所有因鉴权或契约暂停的事件，并继续发送后续积压。它不复活 `dead` 或 `cancelled`。
后端契约使用接收时间入库，积压事件可能被显示为较晚发生；恢复前应核对时效需求，不静默删除。
生产库固定接口地址，不能用同一数据库切换测试与生产 URL；需要独立数据库。

禁止在游标消费前清空或轮转 JSONL。检测到文件替换时应优先恢复原文件；如需从新文件重新接入，
停止生产者和发送服务、核对未导入数据，再显式 `disable` 后 `bind`。
此操作取消该源未完成队列并从新文件末尾开始，不是历史回放工具。
备份数据库使用 SQLite 在线备份接口或停服后完整备份；不能只复制正在写入的主数据库而遗漏 WAL。

## 已实现边界与后续工作

文件记录完成后，游标和 outbox 事务、固定 ID、租约恢复支持发送重启恢复与请求重试。
这不是推理状态机与 outbox 的统一事务；进程崩溃仍可能丢失尚未持久写入的状态转换，
状态机重启也可能产生不同 ID 的重复语义事件。后端仍须实现事件 ID 唯一约束与重复成功响应。

当前不自动清理已发送记录或轮转事件文件，以保留去重依据。后续需补充有明确保留期限的归档、
磁盘容量监控、状态机检查点、端到端对账和多路持续运行验收。
首次部署已通过 387 项 Python 测试，覆盖映射、幂等、重启恢复、网络异常和本地 HTTP 契约。
2026-09-07 驼峰 JSON 字段与业务响应处理变更后，全量 408 项测试通过；覆盖旧队列转换、固定 ID 重试、业务错误不误判成功。
数字用户 ID 与用户名拆分后，全量 426 项测试通过，包含 v1 备份迁移、失败回滚、结果不确定时拒绝改写、游标保留和 JSON 数字类型验证。
测试通过不替代真实后端的鉴权与接收验收。
