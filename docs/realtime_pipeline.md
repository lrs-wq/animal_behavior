# 实时主链路设计

当前第一版主入口：

```powershell
$env:PYTHONPATH="D:\gogelila\Eating_behavior\Eating_behavior_formal_project\src"
D:\Anaconda3\envs\yolo\python.exe -m eating_behavior.main --source D:\path\to\video_or_stream
```

`--source` 兼容：

- 单个视频文件
- 视频文件夹
- 摄像头编号，例如 `0`
- RTSP/HTTP 视频流

## 第一版处理链路

```text
视频/视频流
  -> 猫检测 cat / feeding_head
  -> 项目侧显式同类 NMS
  -> 原始 cat 框 -> 毛色分类、身份判断、feeding_head 关联和事件状态机
  -> 原始 cat 框 -> source 内 track -> bbox 平滑 -> RTMPose
  -> 姿态质量/几何门控 -> 按 (source_id, track_id) 做关键点时序滤波
  -> 吃饭/趴卧/站立/蹲姿分类与姿态变化确认
  -> 猫出现提醒/猫吃饭/姿态变化/余粮监控事件写入 announcement_events
```

## 检测、跟踪与姿态边界

`detection.nms_iou` 控制项目自身的 class-aware NMS。它在模型输出转换为 `Detection` 后始终执行，只让同一类别的重叠框互相抑制；不同类别（例如 `cat` 与 `feeding_head`）不会互相删除。这样即使端到端检测权重绕过 Ultralytics 内部 NMS，项目仍会得到确定的去重结果。`detection.iou` 仍会传给检测后端，但不能替代项目侧 NMS。

NMS 后的 cat 框有两条用途不同的分支：

- 身份、毛色、`feeding_head` 关联、进食判断和事件记录始终使用未平滑的检测框；
- 姿态分支在每个 `source_id` 内分配临时 `track_id`，再平滑 bbox，平滑框只作为 RTMPose 的 top-down crop 和姿态坐标参考。

这里的 `track_id` 只是同一路视频会话内的姿态时序键，不是猫身份。匹配结合 IoU、中心距离和短时运动预测；候选近似并列时不继承旧 track，避免把一只猫的关键点历史带给另一只猫。视频会话结束、时间戳回退或 track 过期后，相应状态会清理。

RTMPose 原始输出先经过置信度滞回、bbox 面积/短边、关键点边界、AP-10K 骨架连通性与骨长等质量/几何门控。通过门控的坐标在 bbox 局部坐标系中按 `(source_id, track_id)` 使用 One Euro 滤波；短暂缺点可按关键点类型有限 hold 并衰减置信度，超过桥接时间或 reset 间隔后不再沿用历史。过滤后的关键点只参与新增的姿态变化事件，不会把姿态分数、平滑框或 track 写入身份和原有进食事件证据，因此不改变原有身份与 `cat_eating` 语义。

主要配置键位于 `configs/pipeline.yaml`：

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

`pose.quality.enabled: false` 时质量门控和关键点时序滤波整体旁路；`tracking.enabled: false` 时姿态直接使用当前检测框且不产生可复用的 track 历史。

## 播报

当前只记录播报事件，不做实际播报。

输出：

```text
outputs/realtime/announcement_events.jsonl
outputs/realtime/announcement_events.csv
```

事件类型：

| event_type | 说明 |
| --- | --- |
| `cat_passed` | 兼容事件名：画面出现任意猫及持续在场周期提醒 |
| `cat_eating` | 检测到猫正在吃饭 |
| `cat_posture_changed` | 首次确认姿态或姿态在 eating / lying / standing / crouching 之间变化 |
| `bowl_food_status` | 碗内余粮监控占位事件，后续补充 |

`cat_passed`、`cat_eating` 和 `cat_posture_changed` 都会输出 `cat_name`。如果当前无法确认身份，`cat_name` 输出为空字符串，内部诊断信息仍保留 `identity_method`、`identity_reason` 和候选猫列表。

姿态状态优先按 `(source_id, household_id, cat_name)` 保存；身份尚未确认时只在当前视频会话内按 `track_id` 暂存，不能把一个新 Track 猜成之前的未知猫。可信姿态需要持续满足 `transition_confirm_seconds` 和 `min_confirmed_observations` 才提交。首次可信姿态生成一条事件，之后只有姿态变化才生成；无猫、`held`、`missing`、`rejected` 或姿态推理失败都不会写成 `unknown/absent`，而是保留上一次已确认状态。

`FrameAnalysis.posture_states` 独立返回保留状态，因此无猫帧即使 `FrameAnalysis.cats` 为空，仍能看到上一次姿态且 `visible=false`。已确认身份的状态会跨媒体重连保留；匿名 Track 状态在视频会话结束时清理。`cat_posture_changed` 写入成功后才提交新状态；写入失败只记录限频告警，冻结后的事件信封会复用同一个 `record_id`、时间和证据，在后续帧（包括无猫帧）重试。已确认身份的待提交转换跨断流保留；匿名 Track 清理后只保留有界的内存待发信封。JSONL 或 CSV 其中一个已经写入时，同一个 Recorder 实例内的待提交重试只补尚未完成的文件。底层追加由跨进程文件锁串行化：半条写入失败时只在持锁期间回滚本次残片；完整写入后外层才报错时则按冻结记录校验，确认已存在后不重复追加。所有 JSONL 事件顶层都带 `record_id`，CSV 在 `details_json._record_id` 输出同一值。近期重复调用也会按 `record_id` 幂等，但完成记录索引是有界内存，不是跨进程数据库；进程重启后的待发状态恢复尚未实现，下游仍应按 `record_id` 幂等消费。

`eating` 姿态必须同时有可靠关键点和无歧义的 `feeding_head -> cat` 关联，不能只凭低头判断。`cat_posture_changed(posture=eating)` 是即时姿态变化；原有 `cat_eating` 是经过进入/退出门控后形成的完整进食区间事件，两者可以由下游按用途分别消费。

`cat_passed` 按 `(household_id, source_id)` 维护画面级在场状态，不按逐帧猫名、毛色或姿态 `track_id` 分桶。同一帧有多只猫也只生成一条提醒：

```text
首次检测到任意猫 -> 立即提醒
持续有猫在画面 -> 到 repeat_interval_seconds 时补充提醒
画面没有猫 -> 暂停提醒
重新出现且周期已到 -> 立即提醒
重新出现但周期未到 -> 周期到期且仍有猫时补充提醒
```

短于 `absence_confirm_seconds` 的检测缺口仍视为同一次在场。默认配置以及可选的用户/家庭覆盖值：

```yaml
events:
  cat_presence:
    repeat_interval_seconds: 1800.0
    absence_confirm_seconds: 2.0
    household_repeat_interval_seconds:
      customer_a_home: 900.0
```

事件写入成功后才消费本次提醒周期；文件写入失败时，下一帧仍可重试。当前保持 `cat_passed` / `message_key=cat_passed`，避免破坏既有消费端，事件 `details.semantics` 为 `any_cat_presence_reminder`。

`cat_eating` 的身份判断顺序：

```text
当前 feeding_head 关联到的 cat 身份
  -> 如果当前帧无法确认，回看同一路视频最近 identity_history_seconds 秒内的确认身份
  -> 历史身份需要和当前关联猫的毛色一致；没有毛色信息时才使用最近确认身份
  -> 仍无法确认则 cat_name 留空
```

## 家庭猫配置

在 `configs/pipeline.yaml` 中配置：

```yaml
identity:
  cats:
    - name: ragdoll
      coat_class: mixed_white_ragdoll
    - name: silver_pure
      coat_class: silver_gray_white
    - name: silver_mixed
      coat_class: silver_gray_white
```

单进程处理多个客户时，运行时通过稳定 `source_id` / `camera_id` 区分家庭，视频流地址只用于拉流：

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
    - household_id: customer_a_home
      source_ids:
        - cam_customer_a_01
      cats:
        - name: 小蓝
          coat_class: blue_cat
    - household_id: customer_b_home
      source_ids:
        - cam_customer_b_01
      cats:
        - name: 小蓝
          coat_class: blue_cat
```

万级客户规模下，全量客户、家庭、猫咪和摄像头关系应存放在中心数据库。推理 worker 启动时只加载调度系统分配给它的本地快照，一般是 20 路以上 `source_id` 对应的数据。每帧会按运行时 `source_id` 精确查找家庭注册表。即使两个客户都有同名猫或相同 `coat_class`，身份判断、身份时间线和事件限频也只在各自 `source_id` 内生效。事件顶层的 `source` 字段输出稳定 `source_id`，另有 `household_id` 便于下游直接按客户家庭聚合。未匹配到家庭且没有默认 `identity.cats` 时，内部身份方法标记为 `unregistered_source`，对外 `cat_name` 留空。

兼容调试命令：

```powershell
python -m eating_behavior.main --source rtsp://debug.example/live --source-id cam_debug_01
```

生产快照命令：

```powershell
python -m eating_behavior.main --pipeline-config configs/pipeline.yaml
```

如果 `silver_gray_white` 有多只猫，系统不会只凭毛色确认，内部会标记 `needs_identity_confirmation`，事件输出的 `cat_name` 留空，等待 ReID 分支接入。
