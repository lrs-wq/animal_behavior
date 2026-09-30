import { createHmac, randomBytes, scryptSync, timingSafeEqual } from "node:crypto";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { fileURLToPath } from "node:url";
import { isAbsolute, resolve } from "node:path";

const HOST = process.env.JOYAI_VIEWER_HOST || "127.0.0.1";
const PORT = Number.parseInt(process.env.JOYAI_VIEWER_PORT || "8099", 10);
const USERNAME = process.env.JOYAI_VIEWER_USERNAME || "";
const PASSWORD_HASH = process.env.JOYAI_VIEWER_PASSWORD_HASH || "";
const SESSION_SECRET = process.env.JOYAI_VIEWER_SESSION_SECRET || "";
const SESSION_TTL_SECONDS = Number.parseInt(
  process.env.JOYAI_VIEWER_SESSION_TTL_SECONDS || "43200",
  10,
);
const ANALYSIS_ENABLED = /^(1|true|yes|on)$/i.test(
  process.env.JOYAI_ANALYSIS_ENABLED || "false",
);
const ANALYSIS_STATUS_PATH =
  process.env.JOYAI_ANALYSIS_STATUS_PATH ||
  "/data/Eating_behavior/runtime/camera02-ai/status.json";
const ANALYSIS_STATUS_MAX_AGE_SECONDS = Number.parseInt(
  process.env.JOYAI_ANALYSIS_STATUS_MAX_AGE_SECONDS || "15",
  10,
);

const COOKIE_NAME = "joyai_stream_session";
const COOKIE_PATH = "/joyai";
const PUBLIC_DIR = fileURLToPath(new URL("./public/", import.meta.url));
const MAX_BODY_BYTES = 8 * 1024;
const LOGIN_WINDOW_MS = 15 * 60 * 1000;
const MAX_LOGIN_FAILURES = 5;
const SESSION_ID_BYTES = 18;
const HLS_PATH_PATTERN = /^\/live\/([A-Za-z0-9][A-Za-z0-9._-]{0,127})(?:\/|$)/;
const PUBLIC_STREAM_PREFIX = "/joyai/stream";

const staticFiles = new Map([
  ["/", ["index.html", "text/html; charset=utf-8"]],
  ["/index.html", ["index.html", "text/html; charset=utf-8"]],
  ["/app.js", ["app.js", "text/javascript; charset=utf-8"]],
  ["/styles.css", ["styles.css", "text/css; charset=utf-8"]],
]);

const streams = Object.freeze([
  {
    id: "camera01",
    name: "摄像头 01",
    description: "camera01 · 实时画面",
    playbackUrl:
      "/joyai/stream/live/camera01/?autoplay=true&muted=true&controls=true&playsinline=true",
  },
  {
    id: "camera02",
    name: "摄像头 02",
    description: "camera02 · 实时画面",
    playbackUrl:
      "/joyai/stream/live/camera02/?autoplay=true&muted=true&controls=true&playsinline=true",
    ...(ANALYSIS_ENABLED
      ? {
          analysisView: {
            id: "analysis",
            name: "AI 分析",
            description: "camera02 · AI 实时分析",
            streamPath: "live/camera02-ai",
            playbackUrl:
              "/joyai/stream/live/camera02-ai/?autoplay=true&muted=true&controls=true&playsinline=true",
          },
        }
      : {}),
  },
  {
    id: "camera03",
    name: "摄像头 03",
    description: "camera03 · 实时画面",
    playbackUrl:
      "/joyai/stream/live/camera03/?autoplay=true&muted=true&controls=true&playsinline=true",
  },
  {
    id: "camera04",
    name: "摄像头 04",
    description: "camera04 · 实时画面",
    playbackUrl:
      "/joyai/stream/live/camera04/?autoplay=true&muted=true&controls=true&playsinline=true",
  },
]);

const streamsById = new Map(streams.map((stream) => [stream.id, stream]));
const loginAttempts = new Map();
const activeSessions = new Map();
let lastSessionSweep = 0;

function assertConfiguration() {
  const errors = [];
  if (!Number.isInteger(PORT) || PORT < 1 || PORT > 65535) {
    errors.push("JOYAI_VIEWER_PORT 必须是有效端口");
  }
  if (!USERNAME) errors.push("缺少 JOYAI_VIEWER_USERNAME");
  if (!PASSWORD_HASH.startsWith("scrypt$")) {
    errors.push("缺少有效的 JOYAI_VIEWER_PASSWORD_HASH");
  }
  if (SESSION_SECRET.length < 32) {
    errors.push("JOYAI_VIEWER_SESSION_SECRET 至少需要 32 个字符");
  }
  if (!Number.isInteger(SESSION_TTL_SECONDS) || SESSION_TTL_SECONDS < 300) {
    errors.push("JOYAI_VIEWER_SESSION_TTL_SECONDS 至少为 300 秒");
  }
  const resolvedStatusPath = resolve(ANALYSIS_STATUS_PATH);
  if (
    !isAbsolute(ANALYSIS_STATUS_PATH) ||
    !["/run/", "/data/Eating_behavior/runtime/"].some((root) =>
      resolvedStatusPath.startsWith(root),
    )
  ) {
    errors.push(
      "JOYAI_ANALYSIS_STATUS_PATH 必须位于 /run 或 /data/Eating_behavior/runtime 下",
    );
  }
  if (
    !Number.isInteger(ANALYSIS_STATUS_MAX_AGE_SECONDS) ||
    ANALYSIS_STATUS_MAX_AGE_SECONDS < 5 ||
    ANALYSIS_STATUS_MAX_AGE_SECONDS > 300
  ) {
    errors.push("JOYAI_ANALYSIS_STATUS_MAX_AGE_SECONDS 必须在 5 到 300 之间");
  }
  if (errors.length) throw new Error(errors.join("；"));
}

function applySecurityHeaders(res) {
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("X-Frame-Options", "SAMEORIGIN");
  res.setHeader("Referrer-Policy", "no-referrer");
  res.setHeader(
    "Permissions-Policy",
    "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
  );
  res.setHeader(
    "Content-Security-Policy",
    [
      "default-src 'self'",
      "script-src 'self'",
      "style-src 'self'",
      "img-src 'self' data:",
      "connect-src 'self'",
      "media-src 'self' blob:",
      "frame-src 'self'",
      "object-src 'none'",
      "base-uri 'none'",
      "form-action 'self'",
      "frame-ancestors 'self'",
    ].join("; "),
  );
}

function sendJson(res, statusCode, value, extraHeaders = {}) {
  const body = JSON.stringify(value);
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
    "Cache-Control": "no-store",
    ...extraHeaders,
  });
  res.end(body);
}

function sendEmpty(res, statusCode, extraHeaders = {}) {
  res.writeHead(statusCode, {
    "Cache-Control": "no-store",
    ...extraHeaders,
  });
  res.end();
}

function parseCookies(header = "") {
  const cookies = {};
  for (const part of header.split(";")) {
    const separator = part.indexOf("=");
    if (separator < 1) continue;
    const key = part.slice(0, separator).trim();
    const value = part.slice(separator + 1).trim();
    if (key) cookies[key] = value;
  }
  return cookies;
}

function signSession(payload) {
  const encoded = Buffer.from(JSON.stringify(payload)).toString("base64url");
  const signature = createHmac("sha256", SESSION_SECRET)
    .update(encoded)
    .digest("base64url");
  return `${encoded}.${signature}`;
}

function removeSession(session) {
  if (typeof session?.sid === "string") activeSessions.delete(session.sid);
}

function sweepExpiredSessions(now = Math.floor(Date.now() / 1000)) {
  if (now - lastSessionSweep < 60) return;
  lastSessionSweep = now;
  for (const [sessionId, state] of activeSessions) {
    if (!Number.isInteger(state?.expiresAt) || state.expiresAt <= now) {
      activeSessions.delete(sessionId);
    }
  }
}

function verifySession(token) {
  if (typeof token !== "string" || token.length > 2048) return null;
  const parts = token.split(".");
  if (parts.length !== 2) return null;

  const expected = createHmac("sha256", SESSION_SECRET)
    .update(parts[0])
    .digest();
  let supplied;
  try {
    supplied = Buffer.from(parts[1], "base64url");
  } catch {
    return null;
  }
  if (expected.length !== supplied.length || !timingSafeEqual(expected, supplied)) {
    return null;
  }

  try {
    const payload = JSON.parse(Buffer.from(parts[0], "base64url").toString("utf8"));
    const now = Math.floor(Date.now() / 1000);
    if (
      payload?.sub !== USERNAME ||
      typeof payload?.sid !== "string" ||
      payload.sid.length < 16 ||
      payload.sid.length > 128 ||
      !Number.isInteger(payload?.iat) ||
      !Number.isInteger(payload?.exp) ||
      payload.exp <= now ||
      payload.iat > now + 60
    ) {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
}

function sessionFromRequest(req) {
  const token = parseCookies(req.headers.cookie)[COOKIE_NAME];
  return verifySession(token);
}

function verifyPassword(password) {
  if (typeof password !== "string" || password.length < 1 || password.length > 256) {
    return false;
  }
  const parts = PASSWORD_HASH.split("$");
  if (parts.length !== 6 || parts[0] !== "scrypt") return false;

  const cost = Number.parseInt(parts[1], 10);
  const blockSize = Number.parseInt(parts[2], 10);
  const parallelization = Number.parseInt(parts[3], 10);
  const salt = Buffer.from(parts[4], "base64url");
  const expected = Buffer.from(parts[5], "base64url");
  if (
    cost !== 16384 ||
    blockSize !== 8 ||
    parallelization !== 1 ||
    salt.length < 16 ||
    expected.length !== 32
  ) {
    return false;
  }

  try {
    const actual = scryptSync(password, salt, expected.length, {
      N: cost,
      r: blockSize,
      p: parallelization,
      maxmem: 64 * 1024 * 1024,
    });
    return timingSafeEqual(actual, expected);
  } catch {
    return false;
  }
}

async function readJson(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) {
      const error = new Error("请求内容过大");
      error.statusCode = 413;
      throw error;
    }
    chunks.push(chunk);
  }
  if (!chunks.length) return {};
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    const error = new Error("请求格式错误");
    error.statusCode = 400;
    throw error;
  }
}

function requestIp(req) {
  const forwarded = req.headers["x-real-ip"];
  return typeof forwarded === "string" && forwarded.length < 128
    ? forwarded
    : req.socket.remoteAddress || "unknown";
}

function rateState(ip) {
  const now = Date.now();
  const current = loginAttempts.get(ip);
  if (!current || now - current.windowStarted >= LOGIN_WINDOW_MS) {
    const fresh = { failures: 0, windowStarted: now, blockedUntil: 0 };
    loginAttempts.set(ip, fresh);
    return fresh;
  }
  return current;
}

function sameOrigin(req) {
  const origin = req.headers.origin;
  if (!origin) return true;
  const proto = req.headers["x-forwarded-proto"] || "http";
  const host = req.headers["x-forwarded-host"] || req.headers.host;
  return origin === `${proto}://${host}`;
}

async function handleLogin(req, res) {
  if (!sameOrigin(req)) {
    sendJson(res, 403, { error: "请求来源不受信任" });
    return;
  }

  const ip = requestIp(req);
  const state = rateState(ip);
  const now = Date.now();
  if (state.blockedUntil > now) {
    const retryAfter = Math.ceil((state.blockedUntil - now) / 1000);
    sendJson(
      res,
      429,
      { error: "登录尝试过多，请稍后再试" },
      { "Retry-After": String(retryAfter) },
    );
    return;
  }

  let body;
  try {
    body = await readJson(req);
  } catch (error) {
    sendJson(res, error.statusCode || 400, { error: error.message });
    return;
  }

  const usernameMatches =
    typeof body.username === "string" && body.username === USERNAME;
  const passwordMatches = verifyPassword(body.password);
  if (!usernameMatches || !passwordMatches) {
    state.failures += 1;
    if (state.failures >= MAX_LOGIN_FAILURES) {
      state.blockedUntil = now + LOGIN_WINDOW_MS;
    }
    sendJson(res, 401, { error: "用户名或密码错误" });
    return;
  }

  loginAttempts.delete(ip);
  const issuedAt = Math.floor(now / 1000);
  const expiresAt = issuedAt + SESSION_TTL_SECONDS;
  const sessionId = randomBytes(SESSION_ID_BYTES).toString("base64url");
  const token = signSession({
    sub: USERNAME,
    iat: issuedAt,
    exp: expiresAt,
    sid: sessionId,
    nonce: randomBytes(12).toString("base64url"),
  });
  sweepExpiredSessions(issuedAt);
  activeSessions.set(sessionId, {
    expiresAt,
    selectedStreamId: null,
    selectedViewId: null,
    selectedStreamPath: null,
  });
  const cookie = [
    `${COOKIE_NAME}=${token}`,
    `Path=${COOKIE_PATH}`,
    `Max-Age=${SESSION_TTL_SECONDS}`,
    "HttpOnly",
    "Secure",
    "SameSite=Strict",
  ].join("; ");
  sendJson(
    res,
    200,
    { authenticated: true, username: USERNAME, expiresAt },
    { "Set-Cookie": cookie },
  );
}

function requireSession(req, res) {
  const session = sessionFromRequest(req);
  if (!session) {
    sendJson(res, 401, { error: "登录已失效，请重新登录" });
    return null;
  }
  const now = Math.floor(Date.now() / 1000);
  sweepExpiredSessions(now);
  let state = activeSessions.get(session.sid);
  if (!state) {
    sendJson(res, 401, { error: "登录会话已失效，请重新登录" });
    return null;
  }
  state.expiresAt = session.exp;
  session.state = state;
  return session;
}

function streamPathFromRequestPath(originalUri) {
  if (typeof originalUri !== "string") return null;

  let pathname;
  try {
    pathname = new URL(originalUri, "http://joyai.local").pathname;
  } catch {
    return null;
  }

  if (pathname.startsWith(`${PUBLIC_STREAM_PREFIX}/`)) {
    pathname = pathname.slice(PUBLIC_STREAM_PREFIX.length);
  }

  const match = HLS_PATH_PATTERN.exec(pathname);
  return match ? `live/${match[1]}` : null;
}

function selectedView(stream, requestedViewId) {
  const viewId = requestedViewId || "raw";
  if (viewId === "raw") {
    return {
      id: "raw",
      name: "原始画面",
      description: stream.description,
      streamPath: `live/${stream.id}`,
      playbackUrl: stream.playbackUrl,
    };
  }
  if (viewId === "analysis" && stream.analysisView) return stream.analysisView;
  return null;
}

async function handleStreamSelection(req, res) {
  if (!sameOrigin(req)) {
    sendJson(res, 403, { error: "请求来源不受信任" });
    return;
  }
  const session = requireSession(req, res);
  if (!session) return;

  let body;
  try {
    body = await readJson(req);
  } catch (error) {
    sendJson(res, error.statusCode || 400, { error: error.message });
    return;
  }

  const stream = streamsById.get(body.streamId);
  if (!stream) {
    sendJson(res, 404, { error: "视频流不存在" });
    return;
  }

  const view = selectedView(stream, body.viewId);
  if (!view) {
    sendJson(res, 404, { error: "视频模式不存在或尚未启用" });
    return;
  }

  session.state.selectedStreamId = stream.id;
  session.state.selectedViewId = view.id;
  session.state.selectedStreamPath = view.streamPath;
  sendJson(res, 200, {
    selectedStreamId: stream.id,
    selectedViewId: view.id,
    viewName: view.name,
    viewDescription: view.description,
    playbackUrl: view.playbackUrl,
  });
}

function analysisOfflineStatus() {
  return {
    enabled: true,
    available: false,
    state: "offline",
    updatedAt: null,
    lastFrameAt: null,
    detectionConfidence: 0.6,
    poseEnabled: true,
    postureEnabled: true,
    currentCatInstances: 0,
    currentPoseInstances: 0,
    currentEatingState: "",
    currentPostures: [],
    poseErrorCode: "",
    reconnects: 0,
  };
}

function finiteNumber(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

async function handleAnalysisStatus(req, res) {
  if (!requireSession(req, res)) return;
  if (!ANALYSIS_ENABLED) {
    sendJson(res, 404, { error: "AI 分析尚未启用" });
    return;
  }

  let status;
  try {
    const raw = await readFile(ANALYSIS_STATUS_PATH, "utf8");
    if (raw.length > 64 * 1024) throw new Error("status file is too large");
    status = JSON.parse(raw);
  } catch {
    sendJson(res, 200, analysisOfflineStatus());
    return;
  }

  const lastFrameAt =
    typeof status.last_frame_at === "string" ? status.last_frame_at : null;
  const lastFrameMillis = lastFrameAt ? Date.parse(lastFrameAt) : Number.NaN;
  const fresh =
    Number.isFinite(lastFrameMillis) &&
    Date.now() - lastFrameMillis <= ANALYSIS_STATUS_MAX_AGE_SECONDS * 1000;
  const runningStates = new Set([
    "streaming",
    "recording",
    "recording_and_streaming",
  ]);
  const state = typeof status.state === "string" ? status.state : "unknown";
  const currentPostures = Array.isArray(status.current_postures)
    ? status.current_postures
        .filter((value) => typeof value === "string" && value.length <= 64)
        .slice(0, 8)
    : [];

  sendJson(res, 200, {
    enabled: true,
    available: fresh && runningStates.has(state) && !status.last_error,
    state: fresh ? state : "stale",
    updatedAt: typeof status.updated_at === "string" ? status.updated_at : null,
    lastFrameAt,
    detectionConfidence: finiteNumber(status.detection_confidence, 0.6),
    poseEnabled: status.pose_enabled === true,
    postureEnabled: status.posture_enabled === true,
    currentCatInstances: finiteNumber(status.current_cat_instances),
    currentPoseInstances: finiteNumber(status.current_pose_instances),
    currentEatingState:
      typeof status.current_eating_state === "string"
        ? status.current_eating_state.slice(0, 64)
        : "",
    currentPostures,
    poseErrorCode:
      typeof status.last_pose_error_code === "string"
        ? status.last_pose_error_code.slice(0, 80)
        : "",
    reconnects: finiteNumber(status.reconnects),
  });
}

async function serveStatic(pathname, res) {
  const entry = staticFiles.get(pathname);
  if (!entry) return false;
  try {
    const body = await readFile(`${PUBLIC_DIR}${entry[0]}`);
    res.writeHead(200, {
      "Content-Type": entry[1],
      "Content-Length": body.length,
      "Cache-Control": "no-store",
    });
    res.end(body);
  } catch {
    sendJson(res, 500, { error: "页面资源读取失败" });
  }
  return true;
}

async function handleRequest(req, res) {
  applySecurityHeaders(res);
  const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
  const { pathname } = url;

  if ((req.method === "GET" || req.method === "HEAD") && staticFiles.has(pathname)) {
    if (req.method === "HEAD") {
      sendEmpty(res, 200);
      return;
    }
    await serveStatic(pathname, res);
    return;
  }

  if (pathname === "/health" && req.method === "GET") {
    sendJson(res, 200, { ok: true, service: "joyai-stream-viewer" });
    return;
  }

  if (pathname === "/api/login" && req.method === "POST") {
    await handleLogin(req, res);
    return;
  }

  if (pathname === "/api/auth" && req.method === "GET") {
    const session = requireSession(req, res);
    if (!session) return;
    const originalPath = req.headers["x-original-uri"];
    const requestedStreamPath = streamPathFromRequestPath(originalPath);
    if (
      !session.state.selectedStreamPath ||
      requestedStreamPath !== session.state.selectedStreamPath
    ) {
      sendEmpty(res, 403);
      return;
    }
    sendEmpty(res, 204, {
      "X-JoyAI-Authenticated": "1",
      "X-JoyAI-Selected-Stream": session.state.selectedStreamId,
      "X-JoyAI-Selected-View": session.state.selectedViewId,
    });
    return;
  }

  if (pathname === "/api/session" && req.method === "GET") {
    const session = requireSession(req, res);
    if (!session) return;
    sendJson(res, 200, {
      authenticated: true,
      username: session.sub,
      expiresAt: session.exp,
      selectedStreamId: session.state.selectedStreamId,
      selectedViewId: session.state.selectedViewId,
    });
    return;
  }

  if (pathname === "/api/streams" && req.method === "GET") {
    const session = requireSession(req, res);
    if (!session) return;
    sendJson(res, 200, {
      streams,
      selectedStreamId: session.state.selectedStreamId,
      selectedViewId: session.state.selectedViewId,
    });
    return;
  }

  if (pathname === "/api/select-stream" && req.method === "POST") {
    await handleStreamSelection(req, res);
    return;
  }

  if (pathname === "/api/analysis/status" && req.method === "GET") {
    await handleAnalysisStatus(req, res);
    return;
  }

  if (pathname === "/api/logout" && req.method === "POST") {
    if (!sameOrigin(req)) {
      sendJson(res, 403, { error: "请求来源不受信任" });
      return;
    }
    removeSession(sessionFromRequest(req));
    const cookie = [
      `${COOKIE_NAME}=`,
      `Path=${COOKIE_PATH}`,
      "Max-Age=0",
      "HttpOnly",
      "Secure",
      "SameSite=Strict",
    ].join("; ");
    sendJson(res, 200, { ok: true }, { "Set-Cookie": cookie });
    return;
  }

  if (pathname.startsWith("/api/")) {
    sendJson(res, 404, { error: "接口不存在" });
    return;
  }
  sendJson(res, 404, { error: "页面不存在" });
}

assertConfiguration();

const server = createServer((req, res) => {
  handleRequest(req, res).catch((error) => {
    console.error("Unhandled request error", error);
    if (!res.headersSent) sendJson(res, 500, { error: "服务器内部错误" });
    else res.destroy();
  });
});

server.keepAliveTimeout = 65_000;
server.headersTimeout = 70_000;
server.requestTimeout = 15_000;

server.listen(PORT, HOST, () => {
  console.log(`JoyAI stream viewer listening on http://${HOST}:${PORT}`);
});

function shutdown(signal) {
  console.log(`${signal} received, shutting down`);
  server.close((error) => {
    if (error) {
      console.error(error);
      process.exitCode = 1;
    }
  });
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
