# Eating Behavior 当前知识

## 目标

正式运行时接入视频文件、目录、摄像头编号或 RTSP/HTTP 视频流，完成猫检测、家庭范围内身份确认、进食/姿态判断和结构化事件记录。

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

## 模型治理

- 正式 `src/eating_behavior/` 运行时不训练模型。
- `training/`、姿态数据工具和 W&B 属于离线研发。
- 训练产物不能自动进入生产；必须评估、复制到受控目录、计算 SHA-256，并显式更新 `configs/models.yaml`。
- 当前生产注册表校验检测模型，并以配置与 checkpoint 双 SHA-256 固定 RTMPose；模型权重仍通过受控目录带外提供，不进入 Git。

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
- Viewer 只读取 `/run/eating-behavior-camera02-ai/status.json`，对外返回字段白名单；源地址、事件内容和其他运行明细不进入 HTTP 响应。
- `eating-behavior-camera02-annotated.service` 是独立的连续分段录像单元，默认逐帧推理，不与无录像 AI 直播单元混用。

## 运行时与训练边界

- 生产 worker 不依赖训练数据、W&B 或训练入口。
- 原始视频、客户流地址、模型权重和生产事件内容不能进入知识图谱。
- 推理间隔、事件节流和状态保持变化属于语义变化，必须更新图谱或当前知识。

实时分支、HEAD、未提交文件和知识漂移状态由 `kg context` 动态生成。
