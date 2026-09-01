import assert from "node:assert/strict";
import { randomBytes, scryptSync } from "node:crypto";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { unlink, writeFile } from "node:fs/promises";
import test from "node:test";

const SERVER_URL = new URL("../server.mjs", import.meta.url);
const USERNAME = "viewer-test";
const PASSWORD = "correct-horse-test";
const STATUS_PATH = `/run/joyai-analysis-test-${process.pid}.json`;

function passwordHash() {
  const salt = randomBytes(16);
  const digest = scryptSync(PASSWORD, salt, 32, {
    N: 16384,
    r: 8,
    p: 1,
    maxmem: 64 * 1024 * 1024,
  });
  return [
    "scrypt",
    "16384",
    "8",
    "1",
    salt.toString("base64url"),
    digest.toString("base64url"),
  ].join("$");
}

async function waitForHealth(origin, child) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (child.exitCode !== null) throw new Error("test server exited early");
    try {
      const response = await fetch(`${origin}/health`);
      if (response.ok) return;
    } catch {
      // The listener may not be ready yet.
    }
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  throw new Error("test server did not become healthy");
}

async function startServer({ analysisEnabled, port }) {
  const child = spawn(process.execPath, [SERVER_URL.pathname], {
    env: {
      ...process.env,
      JOYAI_VIEWER_HOST: "127.0.0.1",
      JOYAI_VIEWER_PORT: String(port),
      JOYAI_VIEWER_USERNAME: USERNAME,
      JOYAI_VIEWER_PASSWORD_HASH: passwordHash(),
      JOYAI_VIEWER_SESSION_SECRET: "test-session-secret-that-is-long-enough-1234",
      JOYAI_VIEWER_SESSION_TTL_SECONDS: "600",
      JOYAI_ANALYSIS_ENABLED: String(analysisEnabled),
      JOYAI_ANALYSIS_STATUS_PATH: STATUS_PATH,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  const origin = `http://127.0.0.1:${port}`;
  await waitForHealth(origin, child);
  return { child, origin };
}

async function stopServer(child) {
  if (child.exitCode !== null) return;
  child.kill("SIGTERM");
  await Promise.race([
    once(child, "exit"),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("test server did not stop")), 3000),
    ),
  ]);
}

async function login(origin) {
  const response = await fetch(`${origin}/api/login`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ username: USERNAME, password: PASSWORD }),
  });
  assert.equal(response.status, 200);
  const setCookie = response.headers.get("set-cookie");
  assert.ok(setCookie);
  return setCookie.split(";", 1)[0];
}

async function api(origin, cookie, path, options = {}) {
  return fetch(`${origin}/api/${path}`, {
    ...options,
    headers: { cookie, ...(options.headers || {}) },
  });
}

test("AI view remains hidden until its server-side feature flag is enabled", async () => {
  const port = 20000 + (process.pid % 10000);

  const hidden = await startServer({ analysisEnabled: false, port });
  try {
    const cookie = await login(hidden.origin);
    const streamsResponse = await api(hidden.origin, cookie, "streams");
    assert.equal(streamsResponse.status, 200);
    const streams = (await streamsResponse.json()).streams;
    assert.equal(
      Object.hasOwn(streams.find((stream) => stream.id === "camera02"), "analysisView"),
      false,
    );
    const hiddenSelection = await api(hidden.origin, cookie, "select-stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ streamId: "camera02", viewId: "analysis" }),
    });
    assert.equal(hiddenSelection.status, 404);
    const analysisStatus = await api(hidden.origin, cookie, "analysis/status");
    assert.equal(analysisStatus.status, 404);
  } finally {
    await stopServer(hidden.child);
  }

  const enabled = await startServer({ analysisEnabled: true, port });
  try {
    const cookie = await login(enabled.origin);
    const streamsResponse = await api(enabled.origin, cookie, "streams");
    const streams = (await streamsResponse.json()).streams;
    assert.equal(
      streams.find((stream) => stream.id === "camera02").analysisView.id,
      "analysis",
    );

    const selection = await api(enabled.origin, cookie, "select-stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ streamId: "camera02", viewId: "analysis" }),
    });
    assert.equal(selection.status, 200);
    assert.equal((await selection.json()).selectedViewId, "analysis");

    const rawAuth = await api(enabled.origin, cookie, "auth", {
      headers: { "x-original-uri": "/joyai/stream/live/camera02/" },
    });
    assert.equal(rawAuth.status, 403);
    const analysisAuth = await api(enabled.origin, cookie, "auth", {
      headers: { "x-original-uri": "/joyai/stream/live/camera02-ai/" },
    });
    assert.equal(analysisAuth.status, 204);

    const analysisStatus = await api(enabled.origin, cookie, "analysis/status");
    assert.equal(analysisStatus.status, 200);
    assert.deepEqual(await analysisStatus.json(), {
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
    });

    const now = new Date().toISOString();
    await writeFile(
      STATUS_PATH,
      JSON.stringify({
        state: "streaming",
        updated_at: now,
        last_frame_at: now,
        detection_confidence: 0.61,
        pose_enabled: true,
        posture_enabled: true,
        current_cat_instances: 1,
        current_pose_instances: 1,
        current_eating_state: "eating",
        current_postures: ["standing"],
        last_pose_error_code: "",
        reconnects: 2,
        private_context: "must-not-be-exposed",
      }),
      { mode: 0o600 },
    );
    const onlineStatusResponse = await api(
      enabled.origin,
      cookie,
      "analysis/status",
    );
    assert.equal(onlineStatusResponse.status, 200);
    const onlineStatus = await onlineStatusResponse.json();
    assert.equal(onlineStatus.available, true);
    assert.equal(onlineStatus.state, "streaming");
    assert.equal(onlineStatus.detectionConfidence, 0.61);
    assert.equal(onlineStatus.currentCatInstances, 1);
    assert.equal(onlineStatus.currentPoseInstances, 1);
    assert.equal(onlineStatus.currentEatingState, "eating");
    assert.deepEqual(onlineStatus.currentPostures, ["standing"]);
    assert.equal(onlineStatus.reconnects, 2);
    assert.equal(Object.hasOwn(onlineStatus, "private_context"), false);
  } finally {
    await unlink(STATUS_PATH).catch((error) => {
      if (error.code !== "ENOENT") throw error;
    });
    await stopServer(enabled.child);
  }
});
