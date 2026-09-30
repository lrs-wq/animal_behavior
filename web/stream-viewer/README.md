# JoyAI 实时视频监控页

部署路径：`https://ai.hzzootech.com/joyai/`

功能：

- 单一运维账号登录，不提供注册接口；
- HttpOnly、Secure、SameSite 会话 Cookie；
- 登录限速与 scrypt 密码摘要；
- 登录后同时查看 `camera01`、`camera02`、`camera03`、`camera04`；
- Nginx `auth_request` 校验播放请求，浏览器无法获得媒体服务凭据；
- Nginx 直接转发 MediaMTX 低延迟 HLS，Node.js 不承载视频带宽。
- camera02 的 AI 分析视图由功能开关控制；关闭时页面和接口均不显示入口。

## 环境变量

服务从 `/etc/joyai-stream-viewer.env` 读取：

```text
JOYAI_VIEWER_HOST=127.0.0.1
JOYAI_VIEWER_PORT=8099
JOYAI_VIEWER_USERNAME=viewer
JOYAI_VIEWER_PASSWORD_HASH=scrypt$16384$8$1$...
JOYAI_VIEWER_SESSION_SECRET=至少32字符的随机值
JOYAI_VIEWER_SESSION_TTL_SECONDS=43200
JOYAI_ANALYSIS_ENABLED=false
JOYAI_ANALYSIS_STATUS_PATH=/data/Eating_behavior/runtime/camera02-ai/status.json
JOYAI_ANALYSIS_STATUS_MAX_AGE_SECONDS=15
```

密码只保存 scrypt 摘要。页面和 HLS 均必须经 HTTPS 访问。

`JOYAI_ANALYSIS_ENABLED=true` 时，camera02 会显示“原始画面 / AI 分析”切换和
经过字段白名单过滤的实时状态面板。标注流由
`eating-behavior-camera02-ai-live.service` 发布到 `live/camera02-ai`，且不保存
视频分段。仓库中的 Viewer 功能开关默认关闭，部署通过显式 drop-in 选择是否启用。

## AI 运行输出目录

AI 直播服务将状态、有效配置快照和事件文件写入
/data/Eating_behavior/runtime/camera02-ai/。该目录由服务启动前以
root:joyai-viewer、0750 创建，文件继承 UMask=0027；
Viewer 仅读取其中的 status.json 并返回字段白名单。
服务要求 /data 已挂载，停止服务不会删除该目录；仍不保存视频分段。

Viewer 默认使用上述状态文件，配置校验允许规范化后的
/data/Eating_behavior/runtime/ 和兼容旧部署的 /run/ 路径，
不允许借助 .. 跳出这些范围。

部署更新需同时安装 AI 与 Viewer 单元和新版 server.mjs。
旧 /run/eating-behavior-camera02-ai/ 的文件应在停止写入后保留并复制到
新目录，再重启服务；迁移前需暂时保留旧 systemd RuntimeDirectory，防止
停止服务时被自动清理。已有事件文件保留并继续追加，状态快照启动后更新。
磁盘文件持久保存不代表已实现 HTTP 发送、待发队列或状态机跨进程恢复。

## 代码加载位置

运行代码统一来自 /root/Eating_behavior/web/stream-viewer。
服务用 BindReadOnlyPaths 将该目录映射到私有的 /opt/joyai-stream-viewer，
继续以 joyai-viewer 用户运行并保留 ProtectHome=true。/opt 路径只作为只读
挂载入口，不再复制维护另一份代码。状态等数据继续读取 /data。

部署时安装新版 systemd 单元；源码及 public/ 文件需允许服务用户读取。
执行单元校验和 Viewer 测试后重启服务，检查私有挂载源与 /root 文件 inode 一致。
旧部署副本只保留在 /data/Eating_behavior/ops_backups 中用于恢复。
