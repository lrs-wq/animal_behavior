import { createHmac, randomBytes, scryptSync, timingSafeEqual } from "node:crypto";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { fileURLToPath } from "node:url";

const HOST = process.env.JOYAI_VIEWER_HOST || "127.0.0.1";
const PORT = Number.parseInt(process.env.JOYAI_VIEWER_PORT || "8099", 10);
const USERNAME = process.env.JOYAI_VIEWER_USERNAME || "";
const PASSWORD_HASH = process.env.JOYAI_VIEWER_PASSWORD_HASH || "";
const SESSION_SECRET = process.env.JOYAI_VIEWER_SESSION_SECRET || "";
const SESSION_TTL_SECONDS = Number.parseInt(
  process.env.JOYAI_VIEWER_SESSION_TTL_SECONDS || "43200",
  10,
);

const COOKIE_NAME = "joyai_stream_session";
const COOKIE_PATH = "/joyai";
const PUBLIC_DIR = fileURLToPath(new URL("./public/", import.meta.url));
const MAX_BODY_BYTES = 8 * 1024;
const LOGIN_WINDOW_MS = 15 * 60 * 1000;
const MAX_LOGIN_FAILURES = 5;

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
  },
]);

const loginAttempts = new Map();

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
  const token = signSession({
    sub: USERNAME,
    iat: issuedAt,
    exp: expiresAt,
    nonce: randomBytes(12).toString("base64url"),
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
  return session;
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
    if (!sessionFromRequest(req)) {
      sendEmpty(res, 401);
      return;
    }
    sendEmpty(res, 204, { "X-JoyAI-Authenticated": "1" });
    return;
  }

  if (pathname === "/api/session" && req.method === "GET") {
    const session = requireSession(req, res);
    if (!session) return;
    sendJson(res, 200, {
      authenticated: true,
      username: session.sub,
      expiresAt: session.exp,
    });
    return;
  }

  if (pathname === "/api/streams" && req.method === "GET") {
    if (!requireSession(req, res)) return;
    sendJson(res, 200, { streams });
    return;
  }

  if (pathname === "/api/logout" && req.method === "POST") {
    if (!sameOrigin(req)) {
      sendJson(res, 403, { error: "请求来源不受信任" });
      return;
    }
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
