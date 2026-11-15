import {
  $, $$, devicePresentation, emptyState, escapeHtml, formatClock, formatCoord, formatFlag, formatPreciseCoord, formatTemp, formatTime,
  getJSON, icon, loadDevice, loadingState, state, statusChip,
} from "../core.js";

let control = null;
let step = 0.1;
let inFlight = false;
let refreshRequest;
let holdDeadline = 0;
let holdTimer;
let records = [];
const axes = ["x", "y", "z"];
const outcomes = { COMPLETED: "green", SUCCESS: "green", REJECTED: "amber", FAILED: "red", UNKNOWN: "amber" };

function holdSeconds() {
  return Math.max(0, Math.ceil((holdDeadline - performance.now()) / 1000));
}

function ready() {
  return control?.enabled === true && control.available === true && control.motion_ready === true
    && control.position_trusted === true && control.idle?.idle === true
    && axes.every(axis => control.firmware?.axes_homed?.[axis] === true && control.firmware?.axes_trusted?.[axis] === true);
}

function hasHold() {
  return control?.armed === true && control.hold?.active === true && holdSeconds() > 0;
}

function jogAllowed(axis, direction) {
  const current = control?.position?.[axis];
  const bounds = control?.bounds?.[axis];
  if (!ready() || !hasHold() || inFlight || refreshRequest || !control.steps_mm?.includes(step)
      || !Number.isFinite(current) || !Array.isArray(bounds)) return false;
  const target = Math.round((current + (direction === "+" ? step : -step)) * 1000) / 1000;
  return target >= bounds[0] && target <= bounds[1];
}

function coordinate(position) {
  return position && axes.every(axis => Number.isFinite(position[axis]))
    ? axes.map(axis => formatPreciseCoord(position[axis])).join(" / ")
    : "未知";
}

function flags(values, yes, no) {
  return axes.map(axis => `${axis.toUpperCase()} ${formatFlag(values?.[axis], yes, no)}`).join(" · ");
}

function fact(label, value) {
  return `<div class="kv-row"><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`;
}

function outcomeChip(record) {
  const outcome = record.outcome || record.status || "未知";
  return statusChip(outcome, outcomes[outcome] || "slate");
}

function updateButtons() {
  const locked = inFlight || Boolean(refreshRequest);
  $("#refreshControl").disabled = locked;
  $("#armControl").hidden = !ready() || hasHold();
  $("#armControl").disabled = locked;
  $("#renewHold").hidden = !hasHold();
  $("#renewHold").disabled = locked;
  $("#disarmControl").hidden = !hasHold();
  $("#disarmControl").disabled = locked;
  for (const button of $$("[data-jog]")) {
    const [axis, direction] = button.dataset.jog;
    button.disabled = !jogAllowed(axis, direction);
  }
}

function updateReadiness() {
  const allowed = ready() && hasHold();
  const label = control ? allowed ? "READY" : "BLOCKED" : "读取中";
  const reason = !control ? "等待固件状态" : allowed ? "人工授权与 Hold 有效" :
    control.blocked_reason || control.reason || (ready() ? "等待人工授权或 Hold 已到期" : "安全条件未满足");
  $("#controlReadiness").innerHTML = statusChip(label, allowed ? "green" : control ? "red" : "slate");
  $("#controlReason").textContent = reason;
  $("#currentHold").textContent = control?.hold?.active && holdSeconds() > 0 ? `${holdSeconds()} 秒` : "未生效";
  $("#currentArm").textContent = hasHold() ? "有效" : "未授权";
  updateButtons();
}

function syncControl() {
  const fw = control?.firmware || {};
  const session = control?.session;
  $("#controlSession").textContent = session?.state || (session?.active ? "已获取" : control?.available ? "无活动会话" : "未知");
  $("#controlHomed").textContent = flags(fw.axes_homed, "已归零", "未归零");
  $("#controlTrusted").textContent = flags(fw.axes_trusted, "可信", "不可信");
  $("#controlBounds").textContent = control?.bounds
    ? axes.map(axis => `${axis.toUpperCase()} ${control.bounds[axis]?.join("–") || "未知"}`).join(" / ") + " mm"
    : "未知";
  $("#controlSteps").textContent = control?.steps_mm?.join(" / ") || "未知";
  $$("#stepOptions button").forEach(button => {
    button.disabled = !control?.steps_mm?.includes(Number(button.dataset.step));
  });
  const direction = control?.suggested_jog;
  $("#jogSuggestion").textContent = ready() && direction
    ? `建议人工首步 ${direction.axis.toUpperCase()}${direction.direction} ${direction.step_mm} mm` : "";
  $("#controlProtocol").innerHTML = [
    fact("协议", control?.protocol || "未知"), fact("绝对坐标模式", formatFlag(fw.absolute_mode, "启用", "关闭")),
    fact("坐标一致性", formatFlag(control?.coordinate_semantics_verified, "已验证", "未验证")),
    fact("逻辑位置 (mm)", coordinate(fw.logical_position)), fact("原生位置 (mm)", coordinate(fw.native_position)),
    fact("工作偏移 (mm)", coordinate(fw.workspace_offset)), fact("运动学位置 (mm)", coordinate(control?.ground_truth)),
    fact("快照 ID", control?.snapshot_id || "未知"), fact("授权快照 ID", control?.armed_snapshot_id || "无"),
    fact("Hold ID", control?.hold_id ?? "无"),
    fact("Command Session ID", session?.session_id || "无"), fact("Command ID", session?.command_id || "无"),
    fact("命令已接受", formatFlag(session?.accepted, "是", "否")),
    fact("固件会话", fw.firmware_session_id ?? "未知"), fact("固件 generation", fw.state_generation ?? "未知"),
    fact("门禁错误", control?.errors?.join(" · ") || (control?.available ? "无" : "未知")),
  ].join("");
  updateDevice(state.device);
  updateReadiness();
}

function updateDevice(device) {
  const pos = device?.online ? device.position_mm : null;
  axes.forEach(axis => { $(`#sim-${axis}`).textContent = formatCoord(pos?.[axis]); });
  $("#simOnline").textContent = devicePresentation(device)[0];
  $("#simIdle").textContent = formatFlag(device?.online ? device.idle : null, "空闲 (Idle)", "忙碌 (Busy)");
  $("#simQueue").textContent = device?.online ? device.moves_queued ?? "未知" : "未知";
  $("#simObserved").textContent = formatTime(device?.observed_at_utc);
  $("#simHotend").textContent = formatTemp(device?.online ? device.temperature?.hotend : null);
  $("#simBed").textContent = formatTemp(device?.online ? device.temperature?.bed : null);
}

function showAudit(error) {
  const latest = records.find(row => row.axis) || records[0];
  $("#lastControl").innerHTML = latest ? `${outcomeChip(latest)}
    <span class="muted">${escapeHtml(formatTime(latest.utc || latest.started_utc))}</span>` : escapeHtml(error ? "读取失败" : "暂无记录");
  $("#controlHistory").innerHTML = error ? emptyState("无法读取控制审计", error.message)
    : records.length ? records.map(record => `<div class="audit-row">
      <time class="mono">${escapeHtml(formatClock(record.utc || record.started_utc))}</time>
      <div><strong>${escapeHtml(record.axis ? `${record.axis.toUpperCase()}${record.direction} ${record.step_mm} mm` : record.command || record.outcome)}</strong>
      <small>${escapeHtml([record.source, record.request_id?.slice(0, 8), record.error].filter(Boolean).join(" · "))}</small>
      ${record.before && record.after ? `<small>${escapeHtml(coordinate(record.before))} → ${escapeHtml(coordinate(record.after))} mm</small>` : ""}</div>
      ${outcomeChip(record)}
    </div>`).join("") : emptyState("暂无控制记录");
}

function refreshControl() {
  if (refreshRequest) return refreshRequest;
  const startedAt = performance.now();
  const options = () => ({ signal: AbortSignal.timeout(9000) });
  refreshRequest = Promise.allSettled([
    getJSON("/api/simulator/control/status", options()),
    getJSON("/api/simulator/control/history?limit=30", options()),
  ]).then(([status, history]) => {
    clearInterval(holdTimer);
    control = status.status === "fulfilled" ? status.value :
      { blocked_reason: "控制状态读取失败", errors: [status.reason.message] };
    holdDeadline = control.hold?.active && Number.isFinite(control.hold.remaining_ms)
      ? startedAt + Math.max(0, control.hold.remaining_ms) : 0;
    if (holdSeconds() > 0) holdTimer = setInterval(() => {
      updateReadiness();
      if (!holdSeconds()) clearInterval(holdTimer);
    }, 1000);
    syncControl();
    records = history.status === "fulfilled" ? history.value.records || [] : [];
    showAudit(history.status === "rejected" ? history.reason : null);
  }).finally(() => { refreshRequest = null; updateButtons(); });
  updateButtons();
  return refreshRequest;
}

async function submitControl(path, label, body) {
  if (inFlight || refreshRequest) return;
  inFlight = true;
  updateButtons();
  const message = $("#controlMessage");
  message.className = "small notice";
  message.textContent = `${label} · 请求处理中`;
  try {
    const result = await getJSON(path, {
      method: "POST",
      ...(body ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}),
    });
    const outcome = result.status || result.code || (result.ok ? "成功" : "未确认成功");
    message.textContent = [label, outcome, result.message, result.error, result.request_id?.slice(0, 8)].filter(Boolean).join(" · ");
    if (result.ok !== true) message.classList.add("error");
    if (path.endsWith("/jog")) await loadDevice();
  } catch (error) {
    message.classList.add("error");
    message.textContent = `${label} · 未确认成功 · ${error.message}`;
  } finally {
    await refreshControl();
    inFlight = false;
    updateButtons();
  }
}

export function render() {
  return `<header class="page-head"><div><h1>仿真控制</h1>
    <p class="page-sub">本地 Marlin Simulator · 人工授权的单轴点动</p></div>
    <button class="btn" id="refreshControl">${icon("refresh")}重新验证（只读）</button></header>
  <div class="safety-banner">${icon("shield")}<div><strong>Simulator-only</strong>
    控制限于本地模拟器。AI Agent 只能读取设备状态。</div></div>
  <div class="grid grid-sidebar simulator-layout">
    <section class="card control-card">
      <div class="card-head"><h2 class="card-title">运动控制</h2><span class="small muted">单轴 / mm</span></div>
      <div class="control-summary">
        <div><h3>当前控制</h3><div id="controlReadiness">${statusChip("读取中")}</div><p id="controlReason" class="muted small">等待读取</p></div>
        <div><h3>上次操作</h3><div id="lastControl" class="row small">等待读取</div></div>
      </div>
      <div class="card-body">
        <div class="step-row"><span class="small muted">步长</span><div class="segmented" id="stepOptions" aria-label="点动步长">
          ${[0.1, 1, 10].map(value => `<button type="button" class="seg" data-step="${value}" aria-pressed="${value === step}" disabled>${value} mm</button>`).join("")}
        </div></div>
        <div class="jog-grid" aria-label="三轴点动">
          ${axes.map(axis => `<div class="jog-axis"><span>${axis.toUpperCase()} 轴</span>
            <div class="row">${["-", "+"].map(direction => `<button type="button" class="btn jog" data-jog="${axis + direction}" disabled>
              ${axis.toUpperCase()}${direction === "-" ? "−" : "+"}</button>`).join("")}</div></div>`).join("")}
        </div>
        <p class="small muted" id="jogSuggestion"></p>
        <p id="controlMessage" class="small muted" role="status">每次操作重新验证固件状态、边界和会话。</p>
        <p class="small muted">Home 未开放。请在 Simulator 的 SerialHost 面板手动归零，再重新验证。急停、暂停、加热和风扇控制未开放。</p>
      </div>
    </section>
    <div class="grid">
      <section class="card">
        <div class="card-head"><h2 class="card-title">人工授权与会话</h2></div>
        <div class="card-body"><dl class="kv">
          <div class="kv-row"><dt>人工授权</dt><dd id="currentArm">未知</dd></div>
          <div class="kv-row"><dt>Hold 剩余</dt><dd id="currentHold">未知</dd></div>
          <div class="kv-row"><dt>Command Session</dt><dd id="controlSession">未知</dd></div>
          <div class="kv-row"><dt>固件归零</dt><dd id="controlHomed">未知</dd></div>
          <div class="kv-row"><dt>轴可信状态</dt><dd id="controlTrusted">未知</dd></div>
          <div class="kv-row"><dt>边界</dt><dd id="controlBounds">未知</dd></div>
          <div class="kv-row"><dt>步长白名单 (mm)</dt><dd id="controlSteps">未知</dd></div></dl>
          <div class="row control-actions">
            <button type="button" class="btn primary" id="armControl" hidden disabled>确认并建立 Hold</button>
            <button type="button" class="btn" id="renewHold" hidden disabled>续期 Hold</button>
            <button type="button" class="btn" id="disarmControl" hidden disabled>退出控制</button>
          </div>
          <details class="details"><summary>固件与协议详情</summary><dl id="controlProtocol" class="kv"></dl></details>
        </div>
      </section>
      <section class="card"><div class="card-head"><h2 class="card-title">设备快照</h2><span class="small muted">只读</span></div>
        <div class="axis-readouts readout-grid">${axes.map(axis => `<div class="readout"><h3>${axis.toUpperCase()} / mm</h3><strong id="sim-${axis}">未知</strong></div>`).join("")}</div>
        <div class="card-body"><dl class="kv">
          <div class="kv-row"><dt>连接</dt><dd id="simOnline">未知</dd></div>
          <div class="kv-row"><dt>运动状态</dt><dd id="simIdle">未知</dd></div>
          <div class="kv-row"><dt>待执行运动</dt><dd id="simQueue">未知</dd></div>
          <div class="kv-row"><dt>喷嘴 / 热床</dt><dd><span id="simHotend">未知</span> / <span id="simBed">未知</span></dd></div>
          <div class="kv-row"><dt>最近设备采集</dt><dd id="simObserved">未知</dd></div>
        </dl></div>
      </section>
    </div>
  </div>
  <section class="card audit-card">
    <div class="card-head"><h2 class="card-title">控制审计</h2><span class="small muted">服务端 · 最近 30 条</span></div>
    <div class="card-body" id="controlHistory">${loadingState("读取控制审计")}</div>
  </section>`;
}

export function mount() {
  $$("#stepOptions button").forEach(button => button.addEventListener("click", () => {
    step = Number(button.dataset.step);
    $$("#stepOptions button").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
    updateButtons();
  }));
  $$("[data-jog]").forEach(button => button.addEventListener("click", () => {
    const [axis, direction] = button.dataset.jog;
    if (jogAllowed(axis, direction)) {
      submitControl("/api/simulator/control/jog", `${axis.toUpperCase()}${direction} ${step} mm`, { axis, direction, step });
    }
  }));
  $("#armControl").addEventListener("click", () => {
    if (!ready() || hasHold() || !window.confirm("确认仅对本地 Marlin Simulator 建立限时人工控制授权？")) return;
    submitControl("/api/simulator/control/arm", "建立 Hold", { confirm_local_simulator: true });
  });
  $("#renewHold").addEventListener("click", () => {
    if (hasHold()) submitControl("/api/simulator/control/hold/renew", "续期 Hold");
  });
  $("#disarmControl").addEventListener("click", () => {
    if (hasHold()) submitControl("/api/simulator/control/disarm", "退出控制");
  });
  $("#refreshControl").addEventListener("click", () => { loadDevice(); refreshControl(); });
  window.addEventListener("pdc:device", event => updateDevice(event.detail));
  updateDevice(state.device);
  refreshControl();
}
