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
const streamGrid = document.getElementById("streamGrid");
const streamCount = document.getElementById("streamCount");
const sessionUser = document.getElementById("sessionUser");
const currentTime = document.getElementById("currentTime");
const cardTemplate = document.getElementById("streamCardTemplate");

let sessionTimer = null;
let clockTimer = null;

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
  if (contentType.includes("application/json")) {
    data = await response.json();
  }
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

function showLogin(message = "") {
  stopDashboardTimers();
  clearStreams();
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
}

function clearStreams() {
  for (const frame of streamGrid.querySelectorAll("iframe")) {
    frame.src = "about:blank";
  }
  streamGrid.replaceChildren();
  streamCount.textContent = "0";
}

function createStreamCard(stream) {
  const card = cardTemplate.content.firstElementChild.cloneNode(true);
  const videoFrame = card.querySelector(".video-frame");
  const placeholder = card.querySelector(".video-placeholder");
  const health = card.querySelector(".stream-health");
  const healthText = health.querySelector("span:last-child");

  card.dataset.streamId = stream.id;
  card.querySelector(".stream-name").textContent = stream.name;
  card.querySelector(".stream-description").textContent = stream.description;

  function connect() {
    const oldFrame = videoFrame.querySelector("iframe");
    if (oldFrame) {
      oldFrame.src = "about:blank";
      oldFrame.remove();
    }

    placeholder.classList.remove("hidden");
    health.classList.remove("online", "error");
    healthText.textContent = "正在连接";

    const frame = document.createElement("iframe");
    frame.className = "stream-player";
    frame.title = `${stream.name}实时视频`;
    frame.allow = "autoplay; fullscreen; picture-in-picture";
    frame.allowFullscreen = true;
    frame.referrerPolicy = "same-origin";
    frame.src = `${stream.playbackUrl}&_=${Date.now()}`;
    frame.addEventListener(
      "load",
      () => {
        placeholder.classList.add("hidden");
        healthText.textContent = "播放器已连接";

        // The MediaMTX player is served from the same origin, so its native
        // video events provide a more reliable state than iframe load alone.
        try {
          const video = frame.contentDocument?.querySelector("video");
          if (!video) return;
          video.addEventListener("playing", () => {
            health.classList.remove("error");
            health.classList.add("online");
            healthText.textContent = "直播中";
          });
          video.addEventListener("waiting", () => {
            health.classList.remove("online", "error");
            healthText.textContent = "缓冲中";
          });
          video.addEventListener("error", () => {
            health.classList.remove("online");
            health.classList.add("error");
            healthText.textContent = "视频连接失败";
          });
          if (!video.paused && video.readyState >= 2) {
            health.classList.add("online");
            healthText.textContent = "直播中";
          }
        } catch {
          // The player still works if a browser restricts iframe inspection.
        }
      },
      { once: true },
    );
    videoFrame.append(frame);
  }

  card.querySelector(".refresh-button").addEventListener("click", connect);
  card.querySelector(".fullscreen-button").addEventListener("click", async () => {
    try {
      await videoFrame.requestFullscreen();
    } catch {
      health.classList.add("error");
      healthText.textContent = "浏览器不允许全屏";
    }
  });

  connect();
  return card;
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
    clearStreams();
    streamGrid.textContent = "视频列表加载失败，请刷新页面重试。";
    return;
  }

  clearStreams();
  streamCount.textContent = String(data.streams.length);
  const fragment = document.createDocumentFragment();
  for (const stream of data.streams) fragment.append(createStreamCard(stream));
  streamGrid.append(fragment);
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

logoutButton.addEventListener("click", async () => {
  logoutButton.disabled = true;
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
