# JoyAI 实时视频监控页

部署路径：`https://ai.hzzootech.com/joyai/`

功能：

- 单一运维账号登录，不提供注册接口；
- HttpOnly、Secure、SameSite 会话 Cookie；
- 登录限速与 scrypt 密码摘要；
- 登录后同时查看 `camera01`、`camera02`、`camera03`、`camera04`；
- Nginx `auth_request` 校验播放请求，浏览器无法获得媒体服务凭据；
- Nginx 直接转发 MediaMTX 低延迟 HLS，Node.js 不承载视频带宽。

## 环境变量

服务从 `/etc/joyai-stream-viewer.env` 读取：

```text
JOYAI_VIEWER_HOST=127.0.0.1
JOYAI_VIEWER_PORT=8099
JOYAI_VIEWER_USERNAME=viewer
JOYAI_VIEWER_PASSWORD_HASH=scrypt$16384$8$1$...
JOYAI_VIEWER_SESSION_SECRET=至少32字符的随机值
JOYAI_VIEWER_SESSION_TTL_SECONDS=43200
```

密码只保存 scrypt 摘要。页面和 HLS 均必须经 HTTPS 访问。
