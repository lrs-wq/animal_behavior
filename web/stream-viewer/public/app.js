const BASE_PATH = "/joyai";

const bootView = document.getElementById("bootView");
const loginView = document.getElementById("loginView");
const dashboardView = document.getElementById("dashboardView");
const loginForm = document.getElementById("loginForm");
const loginButton = document.getElementById("loginButton");
const loginMessage = document.getElementById("loginMessage");
const usernameInput = document.getElementById("username");
const passwordInput = document.getElementById("password");
const togglePassword = document.getElementById("togglePassword");
const logoutButton = document.getElementById("logoutButton");
const streamList = document.getElementById("streamList");
const sessionUser = document.getElementById("sessionUser");
const currentTime = document.getElementById("currentTime");
const playerPanel = document.getElementById("playerPanel");
const videoFrame = document.getElementById("videoFrame");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const streamName = document.getElementById("streamName");
const streamDescription = document.getElementById("streamDescription");
const streamHealth = document.getElementById("streamHealth");
const healthText = streamHealth.querySelector("span:last-child");
const refreshButton = document.getElementById("refreshButton");
const fullscreenButton = document.getElementById("fullscreenButton");
const streamModeSwitch = document.getElementById("streamModeSwitch");
const modeButtons = [...streamModeSwitch.querySelectorAll("button[data-view-id]")];
const analysisPanel = document.getElementById("analysisPanel");
const analysisAvailability = document.getElementById("analysisAvailability");
const analysisAvailabilityText = analysisAvailability.querySelector("span:last-child");
const analysisThreshold = document.getElementById("analysisThreshold");
const analysisPose = document.getElementById("analysisPose");
const analysisCats = document.getElementById("analysisCats");
const analysisEating = document.getElementById("analysisEating");
const analysisPosture = document.getElementById("analysisPosture");
const analysisLastUpdate = document.getElementById("analysisLastUpdate");

let sessionTimer = null;
let clockTimer = null;
let analysisTimer = null;
let streams = [];
let selectedStream = null;
let selectedViewId = "raw";
let selectionVersion = 0;
let selectionBusy = false;

async function api(path, options = {}) {
  const response = await fetch(`${BASE_PATH}/api/${path}`, {
    credentials: "same-origin",
    cache: "no-store",
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });

  let data = null;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) data = await response.json();
  return { response, data };
}

function setVisible(view) {
  for (const element of [bootView, loginView, dashboardView]) {
    element.classList.toggle("hidden", element !== view);
  }
}

function setLoginBusy(busy) {
  loginButton.disabled = busy;
  usernameInput.disabled = busy;
  passwordInput.disabled = busy;
  loginButton.querySelector("span:first-child").textContent = busy ? "正在登录…" : "登录";
}

function setSelectionBusy(busy) {
  selectionBusy = busy;
  for (const button of streamList.querySelectorAll("button")) button.disabled = busy;
  for (const button of modeButtons) button.disabled = busy;
}

function stopPlayer() {
  stopAnalysisPolling();
  selectionVersion += 1;
  const frame = videoFrame.querySelector("iframe");
  if (frame) {
    frame.src = "about:blank";
    frame.remove();
  }
}

function resetPlayer() {
  stopPlayer();
  selectedStream = null;
  selectedViewId = "raw";
  streamModeSwitch.classList.add("hidden");
  analysisPanel.classList.add("hidden");
  streamName.textContent = "请选择摄像头";
  streamDescription.textContent = "选择后开始加载实时画面";
  streamHealth.classList.remove("online", "error");
  healthText.textContent = "等待选择";
  refreshButton.disabled = true;
  fullscreenButton.disabled = true;
  videoPlaceholder.className = "video-placeholder idle-placeholder";
  videoPlaceholder.replaceChildren();
  const icon = document.createElement("span");
  icon.className = "placeholder-icon";
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = "▶";
  const title = document.createElement("strong");
  title.textContent = "请选择一路视频";
  const hint = document.createElement("span");
  hint.textContent = "左侧选择摄像头后开始播放";
  videoPlaceholder.append(icon, title, hint);
  videoPlaceholder.classList.remove("hidden");
  updateSelectionButtons();
}

function clearDashboard() {
  stopPlayer();
  streams = [];
  selectedStream = null;
  streamList.replaceChildren();
  resetPlayer();
}

function showLogin(message = "") {
  stopDashboardTimers();
  clearDashboard();
  setVisible(loginView);
  loginMessage.textContent = message;
  passwordInput.value = "";
  window.setTimeout(() => usernameInput.focus(), 50);
}

function updateClock() {
  const now = new Date();
  currentTime.textContent = new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(now);
  currentTime.dateTime = now.toISOString();
}

function startDashboardTimers() {
  stopDashboardTimers();
  updateClock();
  clockTimer = window.setInterval(updateClock, 1000);
  sessionTimer = window.setInterval(checkSessionSilently, 5 * 60 * 1000);
}

function stopDashboardTimers() {
  if (clockTimer) window.clearInterval(clockTimer);
  if (sessionTimer) window.clearInterval(sessionTimer);
  clockTimer = null;
  sessionTimer = null;
  stopAnalysisPolling();
}

function stopAnalysisPolling() {
  if (analysisTimer) window.clearInterval(analysisTimer);
  analysisTimer = null;
}

function updateModeControls() {
  const analysisAvailable = Boolean(selectedStream?.analysisView);
  streamModeSwitch.classList.toggle("hidden", !analysisAvailable);
  for (const button of modeButtons) {
    button.classList.toggle("active", button.dataset.viewId === selectedViewId);
    button.setAttribute("aria-pressed", String(button.dataset.viewId === selectedViewId));
  }
  analysisPanel.classList.toggle(
    "hidden",
    !analysisAvailable || selectedViewId !== "analysis",
  );
}

function eatingStateLabel(value) {
  const labels = {
    idle: "未进食",
    candidate: "疑似进食",
    eating: "进食中",
    exiting: "结束确认中",
  };
  return labels[value] || value || "未知";
}

function renderAnalysisStatus(status) {
  const online = status?.available === true;
  analysisAvailability.classList.toggle("online", online);
  analysisAvailability.classList.toggle("error", !online);
  analysisAvailabilityText.textContent = online ? "AI 在线" : "AI 未运行";
  analysisThreshold.textContent = Number.isFinite(status?.detectionConfidence)
    ? status.detectionConfidence.toFixed(2)
    : "0.60";
  analysisPose.textContent = !online
    ? "不可用"
    : status?.poseEnabled
      ? status.currentPoseInstances > 0
        ? `已识别 ${status.currentPoseInstances}`
        : "开启·无可靠姿态"
      : "关闭";
  analysisCats.textContent = String(status?.currentCatInstances || 0);
  analysisEating.textContent = eatingStateLabel(status?.currentEatingState);
  analysisPosture.textContent = status?.currentPostures?.length
    ? status.currentPostures.join(" / ")
    : "无可靠姿态";
  if (status?.lastFrameAt) {
    const date = new Date(status.lastFrameAt);
    analysisLastUpdate.textContent = Number.isNaN(date.getTime())
      ? "--"
      : new Intl.DateTimeFormat("zh-CN", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        }).format(date);
  } else {
    analysisLastUpdate.textContent = "--";
  }
}

async function refreshAnalysisStatus() {
  if (selectedViewId !== "analysis" || !selectedStream?.analysisView) return;
  try {
    const { response, data } = await api("analysis/status");
    if (response.status === 401) {
      showLogin("登录已失效，请重新登录。");
      return;
    }
    if (!response.ok) throw new Error("analysis status unavailable");
    renderAnalysisStatus(data);
  } catch {
    renderAnalysisStatus({ available: false, detectionConfidence: 0.6 });
  }
}

function startAnalysisPolling() {
  stopAnalysisPolling();
  if (selectedViewId !== "analysis" || !selectedStream?.analysisView) return;
  refreshAnalysisStatus();
  analysisTimer = window.setInterval(refreshAnalysisStatus, 3000);
}

function updateSelectionButtons() {
  for (const button of streamList.querySelectorAll("button[data-stream-id]")) {
    const active = button.dataset.streamId === selectedStream?.id;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    const action = button.querySelector(".selector-action");
    if (action) action.textContent = active ? "观看中" : "观看";
  }
}

function renderStreamList() {
  streamList.replaceChildren();
  const fragment = document.createDocumentFragment();
  for (const stream of streams) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "stream-option";
    button.dataset.streamId = stream.id;
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", "false");

    const indicator = document.createElement("span");
    indicator.className = "camera-indicator";
    indicator.setAttribute("aria-hidden", "true");
    indicator.textContent = "●";
    const details = document.createElement("span");
    details.className = "selector-details";
    const name = document.createElement("strong");
    name.textContent = stream.name;
    const description = document.createElement("small");
    description.textContent = stream.description;
    details.append(name, description);
    const action = document.createElement("span");
    action.className = "selector-action";
    action.textContent = "观看";
    button.append(indicator, details, action);
    button.addEventListener("click", () => selectStream(stream));
    fragment.append(button);
  }
  streamList.append(fragment);
  updateSelectionButtons();
}

function showConnecting(stream, description = stream.description) {
  streamName.textContent = stream.name;
  streamDescription.textContent = description;
  streamHealth.classList.remove("online", "error");
  healthText.textContent = "正在连接";
  refreshButton.disabled = true;
  fullscreenButton.disabled = true;
  videoPlaceholder.className = "video-placeholder";
  videoPlaceholder.replaceChildren();
  const spinner = document.createElement("div");
  spinner.className = "spinner small-spinner";
  spinner.setAttribute("aria-hidden", "true");
  const text = document.createElement("span");
  text.textContent = "正在连接视频流…";
  videoPlaceholder.append(spinner, text);
  videoPlaceholder.classList.remove("hidden");
}

function connectPlayer(stream, playbackUrl, description) {
  stopPlayer();
  const version = selectionVersion;
  showConnecting(stream, description);

  const frame = document.createElement("iframe");
  frame.className = "stream-player";
  frame.title = `${stream.name}实时视频`;
  frame.allow = "autoplay; fullscreen; picture-in-picture";
  frame.allowFullscreen = true;
  frame.referrerPolicy = "same-origin";
  frame.src = `${playbackUrl}&_=${Date.now()}`;
  frame.addEventListener(
    "load",
    () => {
      if (version !== selectionVersion || selectedStream?.id !== stream.id) return;
      videoPlaceholder.classList.add("hidden");
      refreshButton.disabled = false;
      fullscreenButton.disabled = false;
      healthText.textContent = "播放器已连接";
      try {
        const video = frame.contentDocument?.querySelector("video");
        if (!video) return;
        video.addEventListener("playing", () => {
          if (version !== selectionVersion) return;
          streamHealth.classList.remove("error");
          streamHealth.classList.add("online");
          healthText.textContent = "直播中";
        });
        video.addEventListener("waiting", () => {
          if (version !== selectionVersion) return;
          streamHealth.classList.remove("online", "error");
          healthText.textContent = "缓冲中";
        });
        video.addEventListener("error", () => {
          if (version !== selectionVersion) return;
          streamHealth.classList.remove("online");
          streamHealth.classList.add("error");
          healthText.textContent = "视频连接失败";
        });
        if (!video.paused && video.readyState >= 2) {
          streamHealth.classList.add("online");
          healthText.textContent = "直播中";
        }
      } catch {
        // The player remains usable if iframe inspection is restricted.
      }
    },
    { once: true },
  );
  videoFrame.append(frame);
}

async function selectStream(stream, { restore = false, viewId = "raw" } = {}) {
  if (selectionBusy) return;
  if (
    !restore &&
    selectedStream?.id === stream.id &&
    selectedViewId === viewId
  ) {
    return;
  }

  stopPlayer();
  selectedStream = stream;
  selectedViewId = viewId;
  updateSelectionButtons();
  updateModeControls();
  const requestedDescription =
    viewId === "analysis" ? stream.analysisView?.description : stream.description;
  showConnecting(stream, requestedDescription || stream.description);
  setSelectionBusy(true);
  try {
    const { response, data } = await api("select-stream", {
      method: "POST",
      body: JSON.stringify({ streamId: stream.id, viewId }),
    });
    if (response.status === 401) {
      showLogin("登录已失效，请重新登录。");
      return;
    }
    if (!response.ok || !data?.playbackUrl) {
      throw new Error(data?.error || "视频选择失败");
    }
    selectedViewId = data.selectedViewId || viewId;
    updateModeControls();
    connectPlayer(
      stream,
      data.playbackUrl,
      data.viewDescription || requestedDescription || stream.description,
    );
    startAnalysisPolling();
  } catch (error) {
    streamHealth.classList.add("error");
    healthText.textContent = error.message || "视频选择失败";
    videoPlaceholder.classList.remove("hidden");
  } finally {
    setSelectionBusy(false);
    updateSelectionButtons();
  }
}

async function showDashboard(session) {
  setVisible(dashboardView);
  sessionUser.textContent = `当前用户：${session.username}`;
  startDashboardTimers();

  const { response, data } = await api("streams");
  if (response.status === 401) {
    showLogin("登录已失效，请重新登录。");
    return;
  }
  if (!response.ok || !Array.isArray(data?.streams)) {
    clearDashboard();
    streamList.textContent = "视频列表加载失败，请刷新页面重试。";
    return;
  }

  streams = data.streams;
  resetPlayer();
  renderStreamList();

  const selectedStreamId = data.selectedStreamId || session.selectedStreamId;
  const restoredViewId = data.selectedViewId || session.selectedViewId || "raw";
  const previous = streams.find((stream) => stream.id === selectedStreamId);
  if (previous) {
    await selectStream(previous, { restore: true, viewId: restoredViewId });
  }
}

async function checkSessionSilently() {
  try {
    const { response, data } = await api("session");
    if (response.ok && data?.authenticated) return true;
  } catch {
    return true;
  }
  showLogin("登录已失效，请重新登录。");
  return false;
}

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  loginMessage.textContent = "";
  const username = usernameInput.value.trim();
  const password = passwordInput.value;
  if (!username || !password) {
    loginMessage.textContent = "请输入用户名和密码。";
    return;
  }

  setLoginBusy(true);
  try {
    const { response, data } = await api("login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    if (!response.ok) {
      loginMessage.textContent = data?.error || "登录失败，请稍后重试。";
      return;
    }
    await showDashboard(data);
  } catch {
    loginMessage.textContent = "无法连接服务器，请检查网络后重试。";
  } finally {
    setLoginBusy(false);
  }
});

togglePassword.addEventListener("click", () => {
  const showing = passwordInput.type === "text";
  passwordInput.type = showing ? "password" : "text";
  togglePassword.textContent = showing ? "显示" : "隐藏";
  togglePassword.setAttribute("aria-label", showing ? "显示密码" : "隐藏密码");
  passwordInput.focus();
});

refreshButton.addEventListener("click", () => {
  if (selectedStream) {
    selectStream(selectedStream, { restore: true, viewId: selectedViewId });
  }
});

for (const button of modeButtons) {
  button.addEventListener("click", () => {
    if (!selectedStream || button.dataset.viewId === selectedViewId) return;
    selectStream(selectedStream, { viewId: button.dataset.viewId });
  });
}

fullscreenButton.addEventListener("click", async () => {
  try {
    await videoFrame.requestFullscreen();
  } catch {
    streamHealth.classList.add("error");
    healthText.textContent = "浏览器不允许全屏";
  }
});

logoutButton.addEventListener("click", async () => {
  logoutButton.disabled = true;
  stopPlayer();
  try {
    await api("logout", { method: "POST" });
  } finally {
    logoutButton.disabled = false;
    showLogin("已安全退出登录。 ");
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && !dashboardView.classList.contains("hidden")) {
    checkSessionSilently();
    refreshAnalysisStatus();
  }
});

async function bootstrap() {
  try {
    const { response, data } = await api("session");
    if (response.ok && data?.authenticated) {
      await showDashboard(data);
      return;
    }
  } catch {
    showLogin("暂时无法连接服务器，请稍后重试。 ");
    return;
  }
  showLogin();
}

bootstrap();
