import {
  $, devicePresentation, escapeHtml, events, formatClock, formatCoord, formatFlag, formatTemp, formatTime, getJSON, icon,
  loadDevice, setRefreshInterval, state, statusChip,
} from "../core.js";

function setText(id, value) {
  const node = $(id);
  if (node.textContent !== String(value)) node.textContent = value;
}

function updateDevice(d) {
  const online = d?.online === true;
  const temp = online ? d.temperature || {} : {};
  const pos = online ? d.position_mm || {} : {};
  const issueCodes = online ? [...new Set(d.issues || [])] : [];
  const connection = $("#overviewConnection");
  const [label, tone] = devicePresentation(d);
  connection.dataset.tone = tone;
  setText("#overviewConnection", label);
  setText("#deviceName", d?.source === "SYNTHETIC_FIXTURE" ? "设备状态示例" : "Marlin Simulator");
  setText("#deviceSource", d?.source || "未知");
  setText("#deviceObserved", formatTime(online && d.observed_at_utc));
  setText("#hotendValue", formatTemp(temp.hotend));
  setText("#hotendTarget", formatTemp(temp.hotend_target));
  setText("#bedValue", formatTemp(temp.bed));
  setText("#bedTarget", formatTemp(temp.bed_target));
  for (const axis of ["x", "y", "z"]) setText(`#axis-${axis}`, formatCoord(pos[axis]));
  const endstops = online ? d.endstops?.components || [] : [];
  for (const axis of ["x", "y", "z"]) {
    const component = endstops.find(item => item.axis === `${axis}_min` || item.axis === axis);
    setText(`#limit-${axis}`, formatFlag(component?.triggered, "触发", "未触发"));
  }
  setText("#deviceIdle", formatFlag(online ? d.idle : null, "空闲 (Idle)", "忙碌 (Busy)"));
  setText("#deviceQueue", online ? d.moves_queued ?? "未知" : "未知");
  setText("#thermalFaults", online ? d.thermal_fault_count ?? "未知" : "未知");
  const notice = $("#deviceNotice");
  notice.hidden = !d || (online && !issueCodes.length);
  setText("#deviceNotice", !online ? (d?.error || "设备状态暂不可用") : `读取异常 · ${issueCodes.join(" · ")}`);
  if (!d?.device_connected) updateControl(null);
}

function axesState(flags, yes, no) {
  const values = ["x", "y", "z"].map(axis => flags?.[axis]);
  return values.includes(false) ? no : values.every(value => value === true) ? yes : "未知";
}

function updateControl(control) {
  setText("#deviceHomed", axesState(control?.firmware?.axes_homed, "三轴已归零", "有轴未归零"));
  setText("#deviceTrusted", axesState(control?.firmware?.axes_trusted, "三轴可信", "有轴不可信"));
  const hold = control?.armed === true && control.hold?.active === true && control.hold?.remaining_ms > 0;
  setText("#operatorAccess", !control ? "未知" : !control.enabled ? "未启用" :
    control.motion_ready !== true ? "当前阻止" : hold ? "快照中已授权" : "等待人工授权");
  setText("#controlObserved", control ? formatClock(Date.now()) : "未知");
}

async function readControl() {
  try {
    const control = await getJSON("/api/simulator/control/status", { signal: AbortSignal.timeout(9000) });
    if (state.device?.device_connected) updateControl(control);
  } catch {
    updateControl(null);
  }
}

function updateEvents() {
  const rows = events().filter(event => ["device", "agent"].includes(event.type) && event.title !== "事件视图已刷新").slice(0, 4);
  $("#recentEvents").innerHTML = rows.length ? rows.map(event => `<li>
    <span class="indicator" data-tone="${event.status === "warn" ? "warn" : "neutral"}">${escapeHtml(event.title)}</span>
    <time datetime="${escapeHtml(event.time)}">${escapeHtml(formatClock(event.time))}</time></li>`).join("")
    : '<li class="muted">当前浏览器标签暂无事件。</li>';
}

export function render() {
  return `<header class="page-head"><div><h1>设备总览</h1>
    <p class="page-sub">查看本地模拟器读数、运动状态与控制权限。</p></div>
    <a class="btn primary" href="/console/diagnosis">${icon("chat")}开始 AI 诊断</a></header>
  <section class="card device-summary">
    <div class="device-id"><span class="device-avatar">${icon("cpu")}</span><div><h2 id="deviceName">Marlin Simulator</h2>
      <div class="row device-source"><span class="indicator" id="overviewConnection">正在采集</span>
        <span class="muted" id="deviceSource">未知</span></div></div></div>
    <div class="device-refresh">
      <div class="small muted">最近采集 <time id="deviceObserved">未知</time></div>
      <div class="row"><select class="select" id="refreshInterval" aria-label="自动刷新间隔">
        <option value="5000">每 5 秒</option><option value="10000">每 10 秒</option>
        <option value="30000">每 30 秒</option><option value="0">手动刷新</option></select>
        <button class="btn" id="refreshDevice">${icon("refresh")}刷新状态</button></div>
    </div>
  </section>
  <p class="notice error" id="deviceNotice" hidden></p>
  <div class="grid grid-sidebar overview-layout">
    <section class="card telemetry">
      <div class="card-head"><h2 class="card-title">设备读数</h2>${statusChip("只读", "slate")}</div>
      <div class="readout-grid">
        <div class="readout"><h3>喷嘴温度</h3><strong id="hotendValue">未知</strong><span>目标 <span id="hotendTarget">未知</span></span></div>
        <div class="readout"><h3>热床温度</h3><strong id="bedValue">未知</strong><span>目标 <span id="bedTarget">未知</span></span></div>
      </div>
      <div class="card-head inspector-divider"><h2 class="card-title">坐标与限位</h2><span class="small muted">mm</span></div>
      <div class="readout-grid axis-readouts">${["x", "y", "z"].map(axis => `<div class="readout">
        <h3>${axis.toUpperCase()} 轴</h3><strong id="axis-${axis}">未知</strong>
        <span>最小限位 <span id="limit-${axis}">未知</span></span></div>`).join("")}</div>
      <dl class="telemetry-footer">
        <div><dt>运动状态</dt><dd id="deviceIdle">未知</dd></div>
        <div><dt>待执行运动</dt><dd id="deviceQueue">未知</dd></div>
        <div><dt>热故障计数</dt><dd id="thermalFaults">未知</dd></div>
      </dl>
    </section>
    <aside class="grid">
      <section class="card"><div class="card-head"><h2 class="card-title">控制权限</h2><a class="text-link" href="/console/simulator">查看仿真控制 ${icon("arrow")}</a></div>
        <div class="card-body"><dl class="kv">
          <div class="kv-row"><dt>Agent</dt><dd>只读诊断</dd></div>
          <div class="kv-row"><dt>人工控制</dt><dd id="operatorAccess">未知</dd></div>
          <div class="kv-row"><dt>归零</dt><dd id="deviceHomed">未知</dd></div>
          <div class="kv-row"><dt>轴可信状态</dt><dd id="deviceTrusted">未知</dd></div>
          <div class="kv-row"><dt>控制状态最近读取</dt><dd id="controlObserved">未知</dd></div>
        </dl></div>
      </section>
      <section class="card"><div class="card-head"><h2 class="card-title">近期事件</h2><a class="text-link" href="/console/events">查看事件 ${icon("arrow")}</a></div>
        <ul class="recent-events" id="recentEvents"></ul></section>
    </aside>
  </div>`;
}

export function mount() {
  const interval = $("#refreshInterval");
  interval.addEventListener("change", () => setRefreshInterval(Number(interval.value)));
  $("#refreshDevice").addEventListener("click", async event => {
    const button = event.currentTarget;
    button.disabled = true;
    try { await loadDevice(); await readControl(); }
    finally { button.disabled = false; }
  });
  window.addEventListener("pdc:device", event => {
    updateDevice(event.detail);
    interval.value = String(state.interval);
  });
  window.addEventListener("pdc:event", updateEvents);
  updateDevice(state.device);
  updateEvents();
  readControl();
}
