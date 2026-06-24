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
  -> cat 裁剪图毛色分类
  -> 查询家庭猫配置
  -> 如果该毛色只有一只注册猫，直接确认身份
  -> 如果该毛色有多只猫，进入身份确认分支
  -> feeding_head 触发吃饭事件
  -> 猫经过/猫吃饭/余粮监控事件写入 announcement_events
```

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
| `cat_passed` | 检测到猫经过 |
| `cat_eating` | 检测到猫正在吃饭 |
| `bowl_food_status` | 碗内余粮监控占位事件，后续补充 |

`cat_passed` 和 `cat_eating` 都会输出 `cat_name`。如果当前无法确认身份，`cat_name` 输出为空字符串，内部诊断信息仍保留 `identity_method`、`identity_reason` 和候选猫列表。

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

如果 `silver_gray_white` 有多只猫，系统不会只凭毛色确认，内部会标记 `needs_identity_confirmation`，事件输出的 `cat_name` 留空，等待 ReID 分支接入。
