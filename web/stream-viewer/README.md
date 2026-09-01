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
JOYAI_ANALYSIS_STATUS_PATH=/run/eating-behavior-camera02-ai/status.json
JOYAI_ANALYSIS_STATUS_MAX_AGE_SECONDS=15
```

密码只保存 scrypt 摘要。页面和 HLS 均必须经 HTTPS 访问。

`JOYAI_ANALYSIS_ENABLED=true` 时，camera02 会显示“原始画面 / AI 分析”切换和
经过字段白名单过滤的实时状态面板。标注流由
`eating-behavior-camera02-ai-live.service` 发布到 `live/camera02-ai`，且不保存
视频分段。仓库中的 Viewer 功能开关默认关闭，部署通过显式 drop-in 选择是否启用。
