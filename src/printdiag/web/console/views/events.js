import {
  $, $$, emptyState, escapeHtml, events, formatClock, formatTime, getJSON, icon, loadingState, openDrawer, statusChip,
} from "../core.js";

const labels = { device: "设备", agent: "诊断", control: "控制审计" };
const tones = { device: "green", agent: "blue", control: "slate" };
let filter = "all";
let audit = [];
let error;
let loading = true;
let pending = 0;
let visibleRows = [];

function controlEvent(record) {
  const outcome = record.outcome || record.status || "未知";
  const operation = record.axis ? `${record.axis.toUpperCase()}${record.direction} ${record.step_mm} mm` : "人工控制";
  return { type: "control", title: `${operation} · ${outcome}`, time: record.utc || record.started_utc,
    detail: record.error || record.note || "", status: ["COMPLETED", "SUCCESS", "armed"].includes(outcome) ? "ok" : "warn",
    source: "服务端控制审计", record };
}

function renderRows() {
  const list = $("#eventTimeline");
  visibleRows = [...events().filter(item => ["device", "agent"].includes(item.type) && item.title !== "事件视图已刷新"),
    ...audit.map(controlEvent)].filter(item => filter === "all" || item.type === filter)
    .sort((first, second) => (Date.parse(second.time) || 0) - (Date.parse(first.time) || 0));
  list.innerHTML = visibleRows.length ? visibleRows.map((event, index) => `<button type="button" class="event-row" data-event="${index}">
    <time class="mono">${escapeHtml(formatClock(event.time))}</time>
    <span class="event-content"><strong>${escapeHtml(event.title)}</strong>
      <span>${escapeHtml(event.type === "control" ? event.source : `${event.source} · 浏览器会话`)}</span></span>
    <span class="event-labels">${statusChip(labels[event.type], tones[event.type])}${event.status === "warn" ? statusChip("需注意", "amber") : ""}</span>
    ${icon("arrow")}</button>`).join("") : loading && ["all", "control"].includes(filter)
    ? loadingState("读取控制审计") : emptyState("当前筛选无事件");
  const info = $("#eventCount");
  info.textContent = `${visibleRows.length} 条事件`;
  $("#auditNotice").hidden = !error || !["all", "control"].includes(filter);
  $("#auditNotice").textContent = error ? `控制审计读取失败 · ${error.message}` : "";
  $("#eventLatest").hidden = true;
  pending = 0;
}

async function loadAudit() {
  const button = $("#refreshEvents");
  button.disabled = true;
  loading = true;
  try {
    const data = await getJSON("/api/simulator/control/history?limit=30");
    audit = data.records || [];
    error = undefined;
  } catch (failure) {
    audit = [];
    error = failure;
  } finally {
    loading = false;
    button.disabled = false;
    renderRows();
  }
}

function showDetail(event) {
  const record = event.record;
  openDrawer("事件详情", `<div class="msg-meta">${statusChip(labels[event.type], tones[event.type])}</div>
    <h3>${escapeHtml(event.title)}</h3><p class="muted">${escapeHtml(formatTime(event.time))} · ${escapeHtml(event.source)}</p>
    ${event.detail ? `<p>${escapeHtml(event.detail)}</p>` : ""}
    ${record ? `<dl class="kv"><div class="kv-row"><dt>请求 ID</dt><dd>${escapeHtml(record.request_id || "未知")}</dd></div>
      <div class="kv-row"><dt>操作来源</dt><dd>${escapeHtml(record.source || "未知")}</dd></div>
      <div class="kv-row"><dt>结果</dt><dd>${escapeHtml(record.outcome || record.status || "未知")}</dd></div></dl>`
      : '<p class="small muted">本事件保存在当前浏览器标签的会话中。</p>'}`);
}

export function render() {
  return `<header class="page-head"><div><h1>实时事件流</h1>
    <p class="page-sub">查看本标签页的设备与诊断事件，以及服务端最近 30 条控制审计。</p></div>
    <button class="btn" id="refreshEvents">${icon("refresh")}刷新控制审计</button></header>
  <section class="card">
    <div class="workspace-toolbar">
      <div class="filter-row" aria-label="事件类型">${[["all", "全部"], ...Object.entries(labels)].map(([value, label]) =>
        `<button class="btn small filter-button" data-filter="${value}" aria-pressed="${value === filter}">${label}</button>`).join("")}</div>
      <span class="small muted" id="eventCount" role="status"></span>
    </div>
    <p class="notice error inset" id="auditNotice" hidden></p>
    <div class="event-scroll" tabindex="0" aria-label="事件列表" id="eventScroll"><div id="eventTimeline"></div></div>
    <div class="card-body" id="eventFooter"><button class="btn small" id="eventLatest" hidden>查看新事件</button></div>
  </section>`;
}

export function mount() {
  const scroll = $("#eventScroll");
  $$("[data-filter]").forEach(button => button.addEventListener("click", () => {
    filter = button.dataset.filter;
    $$("[data-filter]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
    renderRows();
    scroll.scrollTop = 0;
  }));
  $("#refreshEvents").addEventListener("click", loadAudit);
  $("#eventTimeline").addEventListener("click", event => {
    const row = event.target.closest("[data-event]");
    if (row) showDetail(visibleRows[Number(row.dataset.event)]);
  });
  $("#eventLatest").addEventListener("click", () => {
    renderRows();
    scroll.scrollTop = 0;
    scroll.focus();
  });
  window.addEventListener("pdc:event", event => {
    if (filter !== "all" && filter !== event.detail.type) return;
    if (scroll.scrollTop <= 16 && !scroll.contains(document.activeElement) && !$("#drawer").open) renderRows();
    else {
      const button = $("#eventLatest");
      button.textContent = `查看 ${++pending} 条新事件`;
      button.hidden = false;
    }
  });
  renderRows();
  loadAudit();
}
