# 0002：camera02 AI 运行输出持久保存到数据盘

- 状态：accepted
- 日期：2026-09-05
- 影响：camera02 AI 部署、Viewer 状态文件路径与输出生命周期
- 补充：0001-camera02-analysis-path.md 的运行输出目录决策

## 背景

原 AI 服务通过 systemd RuntimeDirectory 将状态、有效配置和事件写入 /run。
停止服务或重启主机可能删除这些文件。用户要求将 AI 直播输出切换到 /data。

## 决策

1. AI 服务输出根目录使用 /data/Eating_behavior/runtime/camera02-ai/。
2. 使用 RequiresMountsFor 和 /data 挂载点条件，避免数据盘未挂载时误写系统盘。
3. 启动前创建 root:joyai-viewer、0750 目录，沿用 UMask=0027；不再使用会随服务停止清理的 RuntimeDirectory 管理输出。
4. Viewer 默认与部署配置均读取该目录中的 status.json。路径校验使用规范化的绝对路径，只允许 /data/Eating_behavior/runtime/ 或兼容旧部署的 /run/；状态字段白名单和登录要求保持不变。
5. 迁移时先保留旧运行目录，停止生产者后复制旧文件并校验，再切换服务。事件继续追加，状态与有效配置在启动后更新。

## 边界

原始流与 AI 流仍独立；AI 单元仍无录像、每 3 帧推理一次。
文件持久保存不等于已实现 HTTP 上报、Outbox、事件状态检查点或跨进程语义去重。
输出文件应位于 Git 工作树外，不能将事件内容或有效配置快照加入知识库。

## 验证与回退

运行 Viewer 语法检查、集成测试和 systemd 单元校验。
部署后确认新目录位于 /data、Viewer 用户可读状态、状态持续刷新及直播恢复。
回退前停止 AI，保存数据盘新增文件，再恢复旧单元与 Viewer；不能覆盖或删除迁移后新增事件。

## 证据

- deploy/systemd/eating-behavior-camera02-ai-live.service
- web/stream-viewer/deploy/joyai-stream-viewer.service
- web/stream-viewer/server.mjs
- web/stream-viewer/tests/analysis-feature.test.mjs
- web/stream-viewer/README.md
