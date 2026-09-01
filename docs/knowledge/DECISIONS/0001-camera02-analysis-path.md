# 0001：camera02 原始流与 AI 分析流隔离

- 状态：accepted
- 日期：2026-09-01
- 负责人：Eating Behavior
- 影响节点：`eating.pipeline.camera02_analysis`、`eating.service.viewer`、`eating.model.pose.production`

## 背景

camera02 需要在不改变原始监控流、不把运行事件或源信息暴露给浏览器的前提下，提供可审查的实时姿态与行为分析。逐帧运行检测、姿态和状态机的 GPU 成本过高，而训练产物不能直接进入生产。

## 决策

1. 原始 `live/camera02` 保持不变；独立 systemd 单元复用正式 `RealtimePipeline`，发布 `live/camera02-ai` 标注流。
2. AI 直播单元使用 `--no-recording`，运行状态写入受控 `/run` 目录。Viewer 只返回明确的状态字段白名单。
3. Viewer 的 AI 入口默认关闭，只能由部署 drop-in 显式开启；关闭时流元数据、选择接口和状态接口均不可用。
4. AI 直播每 3 个输入帧运行一次新推理，首帧和重连后的首帧强制推理；中间帧复用最近分析结果。
5. 生产检测与 RTMPose 资产必须位于允许目录，配置与 checkpoint 均校验 SHA-256；权重继续带外管理，不进入 Git。

## 备选方案

- 覆盖原始 camera02 流：会失去独立回退路径，也让推理故障影响原始监控，未采用。
- 逐帧推理：结果更新最及时，但当前资源成本不合适，保留为连续录像单元的默认行为。
- 把完整状态文件直接返回浏览器：可能泄露源地址、路径或事件内容，未采用。

## 后果

- 原始与分析播放路径可以独立授权、启停和回退。
- 分析框、姿态和事件状态最多复用 2 个中间帧；状态机也只在新推理帧推进，因此调整间隔属于业务语义变化。
- Viewer 与分析生产器通过状态文件 schema 松耦合，但新增字段必须经过服务端白名单才能对外返回。
- 部署需要同时管理 AI 发布单元、Viewer 功能开关和模型资产；仓库提交本身不执行部署或服务重启。

## 证据

- `configs/models.yaml`
- `configs/pipeline.yaml`
- `tools/run_live_annotated_recording.py`
- `tests/test_live_inference_interval.py`
- `deploy/systemd/eating-behavior-camera02-ai-live.service`
- `deploy/systemd/joyai-stream-viewer-analysis-enabled.conf`
- `web/stream-viewer/server.mjs`
- `web/stream-viewer/tests/analysis-feature.test.mjs`
