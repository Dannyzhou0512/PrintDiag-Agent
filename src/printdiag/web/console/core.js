// Shared API, status, formatting and dialog behavior.
export const state = {
  device: window.__PDC_BOOT_DEVICE__ || null,
  live: null,
  liveHealth: null,
  liveError: false,
  interval: 5000,
};

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
export const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, ch => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
export const icon = (name, cls = "ico") =>
  `<svg class="${cls}" aria-hidden="true"><use href="/console/static/icons.svg#${name}"></use></svg>`;

function number(value, digits) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : null;
}

export function formatTemp(value) {
  const formatted = number(value, 1);
  return formatted === null ? "未知" : `${formatted} °C`;
}
export const formatCoord = value => number(value, 1) ?? "未知";
export const formatPreciseCoord = value => number(value, 3) ?? "未知";
export const formatFlag = (value, yes, no) => value === true ? yes : value === false ? no : "未知";
const dateTime = new Intl.DateTimeFormat("zh-CN", { dateStyle: "short", timeStyle: "medium", hourCycle: "h23" });
const clock = new Intl.DateTimeFormat("zh-CN", { timeStyle: "medium", hourCycle: "h23" });

function dateLabel(value, formatter) {
  const date = new Date(value);
  return !value || Number.isNaN(date.getTime()) ? "未知" : formatter.format(date);
}

export const formatTime = value => dateLabel(value, dateTime);
export const formatClock = value => dateLabel(value, clock);

export async function getJSON(url, options = {}) {
  const response = await fetch(url, options);
  if (response.ok) return response.json();
  const data = await response.json().catch(() => ({}));
  const detail = data.detail || {};
  const error = new Error(detail.message || detail.code || `HTTP ${response.status}`);
  error.code = detail.code || `HTTP_${response.status}`;
  throw error;
}

export function toast(message, timeout = 3000) {
  const node = $("#drawer").open ? $("#drawerToast") : $("#toast");
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(node._timer);
  node._timer = setTimeout(() => node.classList.remove("show"), timeout);
}

export function openDrawer(title, html) {
  const drawer = $("#drawer");
  $("#drawerTitle").textContent = title;
  $("#drawerBody").innerHTML = html;
  if (!drawer.open) drawer.showModal();
  drawer.scrollTop = 0;
  $("#drawerClose").focus();
}

export function closeDrawer() {
  $("#drawer")?.close();
}

export function sessionRead(key, fallback = null) {
  try { return JSON.parse(sessionStorage.getItem(key)) ?? fallback; }
  catch { return fallback; }
}

export function sessionWrite(key, value) {
  try { sessionStorage.setItem(key, JSON.stringify(value)); } catch { /* storage disabled */ }
}

export function events() { return sessionRead("pdc.events", []); }

export function recordEvent(type, title, detail, source, status = "ok") {
  const row = { id: `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
    time: new Date().toISOString(), type, title, detail, source, status };
  sessionWrite("pdc.events", [row, ...events()].slice(0, 200));
  window.dispatchEvent(new CustomEvent("pdc:event", { detail: row }));
}

export function statusChip(text, kind = "slate") {
  return `<span class="tag ${kind}">${escapeHtml(text)}</span>`;
}

export function emptyState(title, detail = "") {
  return `<div class="empty"><strong>${escapeHtml(title)}</strong>${detail ? `<p>${escapeHtml(detail)}</p>` : ""}</div>`;
}

export function loadingState(label) {
  return `<div class="loading-state" role="status"><span class="skeleton" aria-hidden="true"></span>${escapeHtml(label)}</div>`;
}

export function downloadText(text, filename) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function livePresentation() {
  const { live, liveHealth, liveError } = state;
  if (!live) return liveError ? ["模型状态不可用", "warn"] : ["读取模型状态", "neutral"];
  if (live.mock === true) return ["离线 Mock 模型", "neutral"];
  if (!live.configured) return ["模型未配置", "warn"];
  if (!live.available) return [live.remaining === 0 ? "GPU 预算用尽" : "Live 不可用", "warn"];
  if (liveHealth?.model_ready === true) return ["Qwen 已就绪", "ok"];
  return [liveError ? "模型状态不可用" : liveHealth ? "模型未就绪" : "检查模型服务", "warn"];
}

function setIndicator(id, label, tone) {
  const node = $(id);
  if (node.textContent !== label) node.textContent = label;
  node.dataset.tone = tone;
}

export function devicePresentation(device) {
  if (!device) return ["读取设备", "neutral"];
  if (device.source === "SYNTHETIC_FIXTURE") return ["Mock 示例 · 设备未连接", "neutral"];
  if (device.online !== true) return ["设备离线", "bad"];
  if (device.source === "LIVE_SIMULATOR" && device.device_connected === true) return ["模拟器在线", "ok"];
  return ["设备来源未核实", "warn"];
}

export function updateTopbar() {
  setIndicator("#deviceBadge", ...devicePresentation(state.device));
  setIndicator("#qwenBadge", ...livePresentation());
}

let liveRequest;
export function loadLive() {
  if (liveRequest) return liveRequest;
  liveRequest = (async () => {
    try {
      state.live = await getJSON("/api/live/status");
      state.liveError = false;
      if (state.live.configured && state.live.available && state.liveHealth == null) {
        state.liveHealth = await getJSON("/api/live/health");
      }
    } catch {
      state.liveError = true;
      state.liveHealth = null;
    } finally {
      updateTopbar();
      window.dispatchEvent(new Event("pdc:live"));
      liveRequest = null;
    }
  })();
  return liveRequest;
}

let deviceRequest;
let timer;
let backoff = 0;
let previousDeviceLabel;

function publishDevice(device) {
  state.device = device;
  const [label, tone] = devicePresentation(device);
  if (previousDeviceLabel !== label) {
    recordEvent("device", label, device.error || device.source || "状态接口不可用",
      "设备状态", tone === "ok" || tone === "neutral" ? "ok" : "warn");
  }
  previousDeviceLabel = label;
  updateTopbar();
  window.dispatchEvent(new CustomEvent("pdc:device", { detail: device }));
}

export function loadDevice() {
  if (deviceRequest) return deviceRequest;
  clearTimeout(timer);
  deviceRequest = (async () => {
    try {
      const device = await getJSON("/api/device/status", { signal: AbortSignal.timeout(9000) });
      backoff = 0;
      publishDevice(device);
    } catch (error) {
      backoff = Math.min(60000, Math.max(2000, backoff * 2));
      publishDevice({ online: false, error: error.message });
    } finally {
      deviceRequest = null;
      schedulePoll();
    }
    return state.device;
  })();
  return deviceRequest;
}

function schedulePoll() {
  clearTimeout(timer);
  if (state.interval && !document.hidden) timer = setTimeout(loadDevice, Math.max(state.interval, backoff));
}

export function setRefreshInterval(ms) {
  state.interval = ms;
  sessionWrite("pdc.refresh", ms);
  schedulePoll();
}

export function initDevicePolling() {
  const saved = Number(sessionRead("pdc.refresh", 5000));
  state.interval = [0, 5000, 10000, 30000].includes(saved) ? saved : 5000;
  const previous = events().find(event => event.type === "device");
  previousDeviceLabel = previous?.title;
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearTimeout(timer);
    else if (state.interval) loadDevice();
  });
  if (state.device) {
    publishDevice(state.device);
    schedulePoll();
  } else {
    loadDevice();
  }
}
