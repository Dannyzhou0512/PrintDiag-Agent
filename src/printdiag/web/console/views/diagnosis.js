import {
  $, $$, closeDrawer, downloadText, emptyState, escapeHtml, getJSON, icon,
  livePresentation, loadLive, openDrawer, recordEvent, sessionRead, sessionWrite, state, statusChip, toast,
} from "../core.js";
import { reportMarkup, reportMarkdown, reportText } from "./report.js";

const suggestions = ["当前打印机是什么状态？", "喷嘴无法升温怎么办？", "模型出现欠挤出，怎么排查？"];
const sections = [
  ["当前证据", "evidence"], ["可能原因（待验证）", "possible_causes"],
  ["建议检查", "next_steps"], ["尚不能确认", "uncertain"],
];
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
let busy = false;
let finishTyping;

function source(payload) {
  const { meta = {}, answer = {} } = payload;
  if (meta.live_inference === true) return ["Live Qwen", "green"];
  if (meta.scenario_id) return ["保存回放", "amber"];
  if (meta.mock === true) return ["Mock 测试", "slate"];
  if (answer.mode === "DEVICE_STATUS_ONLY") return ["仅设备数据", "blue"];
  if (payload.report) return ["来源未知", "slate"];
  return ["本地问答", "slate"];
}

function asText(answer, bullet = "• ") {
  const text = [answer.conclusion || ""];
  for (const [label, key] of sections) {
    if (answer[key]?.length) text.push("", label, ...answer[key].map(value => bullet + value));
  }
  return text.join("\n");
}

function answerMarkup(payload) {
  const { answer = {}, meta = {} } = payload;
  const model = meta.model || answer.model;
  if (answer.diagnosis_view) {
    const saved = payload.history?.saved && payload.history.diagnosis_id;
    return reportMarkup(answer.diagnosis_view) + `<div class="answer-actions">
      <button class="btn small ghost" data-action="copy">复制报告</button>
      <button class="btn small" data-action="export">${icon("download")}导出 Markdown</button>
      ${saved ? `<a class="btn small ghost" href="/console/history?diagnosis=${encodeURIComponent(payload.history.diagnosis_id)}">查看历史记录</a>` : ""}
    </div>${payload.history?.saved === false ? '<p class="small notice">历史保存失败，可导出当前报告。</p>' : ""}`;
  }
  return `<div class="msg-meta">${statusChip(...source(payload))}
    ${model ? statusChip(String(model).toUpperCase()) : ""}
    ${payload.report?.status ? statusChip(payload.report.status) : ""}</div>
    <p class="answer-conclusion">${escapeHtml(answer.conclusion || "暂无可显示的回答")}</p>
    ${sections.filter(([, key]) => answer[key]?.length).map(([label, key]) =>
      `<section class="answer-section"><h3>${label}</h3><ul>${answer[key].map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul></section>`
    ).join("")}
    <div class="answer-actions">
      <button class="btn small ghost" data-action="copy">复制回答</button>
        <button class="btn small ghost" data-action="export">${icon("download")}导出 Markdown</button>
    </div>`;
}

function atBottom(chat) {
  return chat.scrollHeight - chat.scrollTop - chat.clientHeight <= 48;
}

function followChat(change, force = false) {
  const chat = $("#chatStage");
  const follow = force || atBottom(chat);
  change(chat);
  if (follow) chat.scrollTop = chat.scrollHeight;
  $("#chatLatest").hidden = atBottom(chat);
}

function addMessage(role, html) {
  const message = document.createElement("article");
  message.className = `msg ${role}`;
  message.innerHTML = `<span class="avatar" aria-hidden="true">${role === "user" ? "你" : "AI"}</span><div class="bubble">${html}</div>`;
  followChat(chat => chat.append(message), role === "user");
  return $(".bubble", message);
}

function setBusy(value) {
  busy = value;
  $("#sendDiagnosis").disabled = value;
  $("#sendDiagnosis").textContent = value ? "正在诊断…" : "提交诊断";
  $("#demoReplay").disabled = value;
  $("#newChat").disabled = value;
}

function showStage(label, elapsed) {
  const stages = $("#stageList");
  $(".stage-item.current", stages)?.classList.replace("current", "done");
  stages.insertAdjacentHTML("beforeend", `<li class="stage-item current"><span class="stage-dot" aria-hidden="true"></span>
    <span>${escapeHtml(label)}</span><span class="stage-time">${Number.isFinite(elapsed) ? Math.round(elapsed) + " ms" : ""}</span></li>`);
  $("#stageStatus").textContent = label;
}

function finishStages(ok) {
  $(".stage-item.current")?.classList.replace("current", ok ? "done" : "failed");
  $("#stageStatus").textContent = ok ? "处理完成" : "请求失败";
}

function startRequest(question, replay = false) {
  finishTyping?.();
  $("#chatIntro").hidden = true;
  $("#stageList").replaceChildren();
  $("#stageStatus").textContent = replay ? "正在读取保存的回放" : "等待服务端阶段";
  setBusy(true);
  addMessage("user", `<p>${escapeHtml(question)}</p>${replay ? '<small>保存回放</small>' : ""}`);
  return addMessage("assistant", '<div class="stage"><span class="spinner" aria-hidden="true"></span>正在处理…</div>');
}

function showAnswer(bubble, payload, question, animate = true) {
  const answer = payload.answer || {};
  const reveal = () => {
    const focused = bubble.contains(document.activeElement);
    const view = answer.diagnosis_view;
    if (view) bubble.closest(".msg").classList.add("report-message");
    followChat(() => { bubble.innerHTML = answerMarkup(payload); });
    if (view) {
      const chat = $("#chatStage");
      chat.scrollTop += bubble.getBoundingClientRect().top - chat.getBoundingClientRect().top - 12;
    }
    if (view && (focused || document.activeElement === document.body)) {
      $(".report-title", bubble).focus({ preventScroll: true });
    } else if (focused) $("button", bubble).focus({ preventScroll: true });
    bubble.addEventListener("click", async event => {
      const action = event.target.closest("button")?.dataset.action;
      if (action === "copy") {
        try { await navigator.clipboard.writeText(view ? reportText(view) : asText(answer)); toast("已复制回答"); }
        catch { toast("无法复制，可导出 Markdown"); }
      } else if (action === "export") {
        const markdown = view ? reportMarkdown(view) : "# PrintDiag 回答\n\n" +
          [question, "", asText(answer)].map(value => value.split("\n").map(line => "    " + line).join("\n")).join("\n");
        downloadText(markdown, `printdiag-${Date.now()}.md`);
      }
    });
  };
  if (answer.diagnosis_view) { reveal(); return; }
  const text = [...asText(answer)];
  if (!animate || reducedMotion.matches || !text.length) { reveal(); return; }
  followChat(() => {
    bubble.innerHTML = '<div class="typing-body" aria-live="off"></div><button class="btn small" type="button" id="skipTyping">立即显示全文</button>';
  });
  const body = $(".typing-body", bubble);
  const chunkSize = Math.max(2, Math.ceil(text.length / 85));
  let index = 0;
  let timer;
  finishTyping = () => {
    clearTimeout(timer);
    finishTyping = undefined;
    reveal();
  };
  const tick = () => {
    index += chunkSize;
    followChat(() => { body.textContent = text.slice(0, index).join(""); });
    if (index < text.length) timer = setTimeout(tick, 23);
    else finishTyping();
  };
  $("#skipTyping", bubble).addEventListener("click", finishTyping);
  tick();
}

function showFailure(bubble, question, error) {
  finishStages(false);
  const exhausted = error.code === "LIVE_BUDGET_EXHAUSTED";
  const message = exhausted ? "本轮模型调用预算已用完。你仍可查看历史，或使用保存回放。"
    : error.code === "LIVE_UNAVAILABLE" ? "当前模型服务不可用。可以使用保存回放查看报告示例。"
      : error.code === "LIVE_BUSY" ? "模型正在处理其他请求，请稍后再试。"
        : "本次请求未完成，请检查服务状态后再试。";
  followChat(() => {
    bubble.innerHTML = `<p class="error-text">${escapeHtml(message)}</p>
      ${exhausted ? '<a class="btn small" href="/console/history">查看历史</a>' : '<button class="btn small" type="button">编辑问题</button>'}`;
  });
  $("button", bubble)?.addEventListener("click", () => {
    const input = $("#symptomInput");
    if (!input.value.trim()) input.value = question;
    input.focus();
  });
  recordEvent("agent", "诊断请求失败", error.code || error.message, "诊断工作流", "warn");
}

async function readStream(response, onStage) {
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = data.detail || {};
    const error = new Error(detail.message || `请求失败（HTTP ${response.status}）`);
    error.code = detail.code;
    throw error;
  }
  if (!response.body) throw new Error("诊断响应为空");
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let pending = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      pending += done ? "\n" : value;
      const lines = pending.split("\n");
      pending = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "stage") onStage(event);
        if (event.type === "final" && event.answer) return event;
        if (event.type === "error") {
          const error = new Error(event.message || "诊断失败");
          error.code = event.code;
          throw error;
        }
      }
      if (done) throw new Error("连接已结束，未收到完整诊断结果");
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

async function sendDiagnosis(event) {
  event.preventDefault();
  if (busy) return;
  const input = $("#symptomInput");
  const question = input.value.trim();
  if (!question) { input.focus(); toast("请输入故障症状或问题"); return; }
  const model = $("#diagModel").value;
  const v2 = $("#diagV2").checked;
  const bubble = startRequest(question);
  input.value = "";
  try {
    const response = await fetch("/api/ask/stream", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symptom: question, model, v2, mode: "auto" }),
    });
    const result = await readStream(response, stage => {
      showStage(stage.label, stage.elapsed_ms);
      recordEvent("agent", stage.label, "服务端 NDJSON 阶段", "诊断工作流");
    });
    finishStages(true);
    showAnswer(bubble, result, question);
    recordEvent("agent", "回答已生成", source(result)[0], "诊断工作流");
  } catch (error) {
    showFailure(bubble, question, error);
  } finally {
    setBusy(false);
    loadLive();
  }
}

async function runReplay(scenario) {
  if (busy) return;
  closeDrawer();
  const bubble = startRequest(scenario.symptom, true);
  try {
    const payload = await getJSON("/api/demo/diagnose", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario_id: scenario.scenario_id, symptom: scenario.symptom, v2: $("#diagV2").checked }),
    });
    finishStages(true);
    showAnswer(bubble, payload, scenario.symptom, false);
    recordEvent("agent", "回放已完成", scenario.scenario_id, "保存回放");
  } catch (error) {
    showFailure(bubble, scenario.symptom, error);
  } finally {
    setBusy(false);
  }
}

async function showScenarios() {
  const button = $("#demoReplay");
  button.disabled = true;
  try {
    const { scenarios = [] } = await getJSON("/api/scenarios");
    if (busy) return;
    openDrawer("Demo 回放", `<p class="muted">选择已保存的场景。回放使用固定问题及保存的模型输出。</p>
      ${scenarios.map((item, index) => `<button type="button" class="doc-result" data-scenario="${index}">
        <strong>${escapeHtml(item.title)}</strong><span>${escapeHtml(item.description)}</span>
        <small>${escapeHtml([item.scenario_id, item.device_source, item.model_source].join(" · "))}</small></button>`).join("")
        || emptyState("暂无回放场景")}`);
    $$("#drawerBody [data-scenario]").forEach(row => row.addEventListener("click", () => runReplay(scenarios[row.dataset.scenario])));
  } catch (error) {
    toast(`场景读取失败：${error.message}`);
  } finally {
    button.disabled = busy;
  }
}

function updateLivePanel() {
  const [label, tone] = livePresentation();
  const live = state.live;
  const context = $("#livePanel");
  context.innerHTML = `<span class="indicator" data-tone="${tone}">${escapeHtml(label)}</span>
    <span class="small muted">${live ? `${live.mock === true ? "本地样例" : "GPU 预算"} ${escapeHtml(live.used)} / ${escapeHtml(live.budget)}` : ""}</span>`;
}

export function render() {
  return `<header class="page-head"><div><h1>AI 智能诊断</h1>
    <p class="page-sub">查询设备状态，或描述故障以获取带证据的排查建议。</p></div>
    <div class="head-actions"><button class="btn" id="demoReplay">${icon("history")}Demo 回放</button>
      <button class="btn" id="newChat">${icon("plus")}新对话</button></div></header>
  <div class="grid grid-sidebar diagnosis-layout">
    <section class="card chat-workspace" aria-label="诊断对话">
      <div class="diag-toolbar">
        <label class="field-inline">模型<select class="select" id="diagModel"><option value="base">Qwen Base</option><option value="lora">Qwen LoRA V0.3</option></select></label>
        <label class="field-inline"><input type="checkbox" id="diagV2" checked>V2 证据策略</label>
      </div>
      <div class="chat-stage" id="chatStage" tabindex="0" aria-label="消息记录">
        <div class="chat-intro" id="chatIntro">
          <h2>描述打印异常</h2>
          <p>写下出现问题的阶段、实际看到的现象，以及已有报错。</p>
          <div class="row suggestions">${suggestions.map(text => `<button class="btn small" data-suggestion="${escapeHtml(text)}">${escapeHtml(text)}</button>`).join("")}</div>
        </div>
      </div>
      <div class="composer-wrap">
        <button class="btn small chat-latest" type="button" id="chatLatest" hidden>跳到最新消息</button>
        <form class="composer" id="diagnosisForm">
          <textarea id="symptomInput" aria-label="诊断问题" maxlength="500" required placeholder="例如：打印外壁明显偏细，相邻走线之间有规则空隙。"></textarea>
          <div class="composer-foot"><span class="small muted">Enter 发送 · Shift+Enter 换行</span><button class="btn primary" id="sendDiagnosis">提交诊断</button></div>
        </form>
      </div>
    </section>
    <aside class="card diagnosis-inspector">
      <div class="card-head"><h2 class="card-title">执行阶段</h2></div>
      <div class="card-body">
        <div id="stageStatus" role="status">等待提问</div>
        <ol class="stage-list" id="stageList"></ol>
      </div>
      <div class="card-head inspector-divider"><h2 class="card-title">模型服务</h2><button class="btn ghost small" id="refreshLive" type="button">刷新</button></div>
      <div class="card-body">
        <div class="live-panel" id="livePanel"></div>
        <p class="small muted">问候、身份和设备状态查询不使用 GPU。</p>
      </div>
    </aside>
  </div>`;
}

export function mount() {
  const model = $("#diagModel");
  const v2 = $("#diagV2");
  const input = $("#symptomInput");
  model.value = sessionRead("pdc.diag.model") === "lora" ? "lora" : "base";
  v2.checked = sessionRead("pdc.diag.v2", true);
  model.addEventListener("change", () => sessionWrite("pdc.diag.model", model.value));
  v2.addEventListener("change", () => sessionWrite("pdc.diag.v2", v2.checked));
  $$("[data-suggestion]").forEach(button => button.addEventListener("click", () => {
    input.value = button.dataset.suggestion;
    input.focus();
  }));
  $("#diagnosisForm").addEventListener("submit", sendDiagnosis);
  input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (!busy) $("#diagnosisForm").requestSubmit();
    }
  });
  $("#chatStage").addEventListener("scroll", event => { $("#chatLatest").hidden = atBottom(event.currentTarget); });
  $("#chatLatest").addEventListener("click", () => followChat(() => {}, true));
  $("#demoReplay").addEventListener("click", showScenarios);
  $("#newChat").addEventListener("click", () => {
    finishTyping?.();
    $$("#chatStage .msg").forEach(message => message.remove());
    $("#chatIntro").hidden = false;
    $("#chatLatest").hidden = true;
    $("#stageList").replaceChildren();
    $("#stageStatus").textContent = "等待提问";
    input.value = "";
    input.focus();
  });
  $("#refreshLive").addEventListener("click", async event => {
    const button = event.currentTarget;
    button.disabled = true;
    state.liveHealth = null;
    try { await loadLive(); } finally { button.disabled = false; }
  });
  window.addEventListener("pdc:live", updateLivePanel);
  updateLivePanel();
}
