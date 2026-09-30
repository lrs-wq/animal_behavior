# 0003：服务器运行代码与数据分盘

- 状态：accepted
- 日期：2026-09-05
- 范围：Eating_behavior 服务器部署
- 来源：用户明确要求 /root 保存运行代码，/data 保存数据

## 决策

1. /root/Eating_behavior 保存源码、脚本、部署模板、知识及可执行 Python 姿态配置。
2. 模型权重迁至 /data/Eating_behavior/models，以原始 SHA-256 验证；配置使用真实普通文件，保留禁止权重文件软链接等已有模型安全校验。
3. 数据集、媒体、事件、持久状态、测试输出、缓存与备份统一落在 /data/Eating_behavior。服务将 XDG/Ultralytics 缓存指向数据盘，并禁用运行时生成 Python 字节码。
4. Viewer 以只读 systemd BindReadOnlyPaths 加载 /root 源码，挂载入口保留 /opt/joyai-stream-viewer，维持专用用户及 ProtectHome；取消独立维护的部署代码副本。
5. /opt 的语言环境、/usr 的系统工具及 /etc 的已安装单元/凭据属于基础设施。/run 仅用于可丢弃运行设施，不能用于持久业务输出。
6. /root 下指向 /data 的目录链接可以用于兼容路径；审计依据实际存储位置，不能仅看路径字符串。

## 验证

- 使用正式模型校验器核对三个权重与 Python 配置的原 SHA-256、允许目录及普通文件属性。
- 模型/配置相关 Python 测试、Viewer 测试及 systemd 单元检查。
- 重启后确认 AI 持续推理、新状态写入 /data，Viewer 私有挂载中的代码 inode 与 /root 源码一致且为只读。
- 复制校验成功、实际服务验证通过后清理 /root 的原权重副本；迁移记录及恢复资料保留在数据盘。

## 证据

- AGENTS.md
- configs/models.yaml
- deploy/systemd/eating-behavior-camera02-ai-live.service
- deploy/systemd/eating-behavior-camera02-annotated.service
- web/stream-viewer/deploy/joyai-stream-viewer.service
- web/stream-viewer/README.md
