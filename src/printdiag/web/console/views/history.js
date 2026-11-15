import { $, downloadText, emptyState, escapeHtml, formatTime, getJSON, icon, loadingState, openDrawer, statusChip, toast } from "../core.js";
import { reportMarkup } from "./report.js";

const pageSize = 20;
const statuses = { complete: "完成", partial: "部分证据", insufficient_evidence: "证据不足", failed: "失败" };
const sources = { LIVE_MODEL: "真实 Live 模型", SAVED_REPLAY: "保存回放", MOCK_TEST: "Mock 测试", DEVICE_ONLY: "仅设备数据" };
let offset = 0;
let total = 0;
let listRequest;
let detailRequest;

function sourceLabel(mode, status) {
  return mode === "LIVE_MODEL" && status === "failed" ? "Live 请求失败" : sources[mode] || mode || "未知";
}

function statusTone(status) {
  return status === "complete" ? "green" : status === "failed" ? "red" : status ? "amber" : "slate";
}

function sourceChip(record) {
  const { source_mode: mode, diagnosis_status: status } = record;
  const tone = mode === "LIVE_MODEL" ? (status === "failed" ? "red" : "blue") : mode === "SAVED_REPLAY" ? "amber" : "slate";
  return statusChip(sourceLabel(mode, status), tone);
}

function reliability(record) {
  if (record.diagnosis_status === "insufficient_evidence") return "证据不足";
  if (record.diagnosis_status === "partial") return "部分证据";
  if (record.diagnosis_status === "failed") return "无可用报告";
  return record.v03_validation?.valid === true ? "已校验" : "未知";
}

function row(record) {
  const id = escapeHtml(record.diagnosis_id);
  const status = record.diagnosis_status;
  const display = record.display || {};
  const safety = display.safety || {};
  const tone = safety.status === "blocked" ? "red" : safety.status === "review_required" ? "amber" : "slate";
  return `<tr role="row">
    <td role="cell" class="history-question"><button type="button" class="row-title" data-open="${id}">${escapeHtml(display.title || "打印问题排查")}</button>
      <div class="row-sub">${escapeHtml(display.excerpt || "无已保存的现象摘要")}</div></td>
    <td role="cell" class="history-source">${sourceChip(record)}</td>
    <td role="cell" class="history-model">${escapeHtml(({ base: "Base", lora: "LoRA" })[record.model_variant] || "—")}</td>
    <td role="cell" class="history-status">${statusChip(statuses[status] || status || "未知", statusTone(status))}</td>
    <td role="cell" class="history-evidence">${escapeHtml(reliability(record))}</td>
    <td role="cell" class="history-time mono">${escapeHtml(formatTime(record.created_at))}<span class="history-safety" data-tone="${tone}">${escapeHtml(safety.label || "安全状态未评估")}</span></td>
  </tr>`;
}

function tableMessage(html) {
  $("#historyBody").innerHTML = `<tr class="table-message" role="row"><td role="cell" colspan="6">${html}</td></tr>`;
}

function updatePager(ready) {
  const page = Math.floor(offset / pageSize) + 1;
  const pages = Math.max(1, Math.ceil(total / pageSize));
  $("#historyPagerLabel").textContent = ready ? `第 ${page} / ${pages} 页 · 共 ${total} 条` : "—";
  $("#historyPrev").disabled = !ready || offset === 0;
  $("#historyNext").disabled = !ready || offset + pageSize >= total;
}

const filterFields = { status: "#historyStatus", source_mode: "#historySource", model: "#historyModel", q: "#historySearch" };

async function loadHistory() {
  listRequest?.abort();
  const request = new AbortController();
  listRequest = request;
  updatePager(false);
  tableMessage(loadingState("读取 SQLite 诊断记录"));
  $("#historyCount").textContent = "读取中";
  const filters = new URLSearchParams();
  for (const [key, selector] of Object.entries(filterFields)) {
    const value = $(selector).value.trim();
    if (value) filters.set(key, value);
  }
  const url = new URL(location.href);
  for (const key of Object.keys(filterFields)) url.searchParams.delete(key);
  filters.forEach((value, key) => url.searchParams.set(key, value));
  if (offset) url.searchParams.set("offset", offset);
  else url.searchParams.delete("offset");
  window.history.replaceState(null, "", url);
  const query = new URLSearchParams(filters);
  query.set("limit", pageSize);
  query.set("offset", offset);
  try {
    const data = await getJSON(`/api/history?${query}`, { signal: request.signal });
    if (listRequest !== request) return;
    total = data.total;
    if (offset && offset >= total) {
      offset = Math.max(0, Math.floor((total - 1) / pageSize) * pageSize);
      return loadHistory();
    }
    if (data.items.length) $("#historyBody").innerHTML = data.items.map(row).join("");
    else tableMessage(emptyState(filters.size ? "没有符合条件的诊断" : "诊断库为空",
      filters.size ? "调整筛选条件后重试。" : "完成并保存诊断后可在此查询。"));
    $("#historyCount").textContent = `SQLite · ${total} 条`;
    updatePager(true);
  } catch (error) {
    if (listRequest !== request || error.name === "AbortError") return;
    tableMessage(emptyState("历史读取失败", error.message));
    $("#historyCount").textContent = "SQLite 暂不可用";
  }
}

async function openDetail(id) {
  detailRequest?.abort();
  const request = new AbortController();
  detailRequest = request;
  openDrawer("诊断详情", loadingState("读取诊断记录"));
  try {
    const { record } = await getJSON(`/api/history/${encodeURIComponent(id)}`, { signal: request.signal });
    if (detailRequest !== request || !$("#drawer").open) return;
    $("#drawerBody").innerHTML = `<p class="small muted">${escapeHtml(formatTime(record.created_at))}</p>
      ${reportMarkup(record.diagnosis_view, { saved: true })}
      <div><button class="btn primary" id="drawerExport">${icon("download")}导出 Markdown</button></div>`;
    $("#drawerExport").addEventListener("click", event => exportReport(record.diagnosis_id, event.currentTarget));
  } catch (error) {
    if (detailRequest !== request || !$("#drawer").open) return;
    $("#drawerBody").innerHTML = `${emptyState("诊断详情读取失败", error.message)}<button class="btn" id="retryHistoryDetail">重试</button>`;
    $("#retryHistoryDetail").addEventListener("click", () => openDetail(id));
  }
}

async function exportReport(id, button) {
  button.disabled = true;
  try {
    const response = await fetch(`/api/history/${encodeURIComponent(id)}/report`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    downloadText(await response.text(), `printdiag-history-${id}.md`);
    toast("Markdown 已导出");
  } catch (error) {
    toast(`导出失败 · ${error.message}`);
  } finally {
    button.disabled = false;
  }
}

export function render() {
  return `<header class="page-head"><div><h1>诊断历史</h1>
    <p class="page-sub">查看、筛选和导出已保存的诊断报告。</p></div>
    <button class="btn" id="historyRefresh">${icon("refresh")}刷新历史</button></header>
  <section class="card">
    <div class="workspace-toolbar"><h2 class="card-title">诊断记录</h2><span class="small muted" role="status" id="historyCount">SQLite</span></div>
    <form class="history-filters" id="historyFilters">
      <label class="search">${icon("search")}<input class="input" type="search" id="historySearch" maxlength="200" aria-label="搜索问题或类别" placeholder="搜索问题或类别"></label>
      <select class="select" id="historyStatus" aria-label="诊断状态"><option value="">全部状态</option>${Object.entries(statuses).map(([id, label]) => `<option value="${id}">${label}</option>`).join("")}</select>
      <select class="select" id="historySource" aria-label="数据来源"><option value="">全部来源</option>${Object.entries(sources).map(([id, label]) => `<option value="${id}">${label}</option>`).join("")}</select>
      <select class="select" id="historyModel" aria-label="模型版本"><option value="">全部模型</option><option value="base">Base</option><option value="lora">LoRA</option></select>
    </form>
    <div class="table-wrap"><table class="data history-table" role="table" aria-label="诊断历史">
      <thead role="rowgroup"><tr role="row"><th scope="col">初步判断</th><th scope="col">来源</th><th scope="col">模型</th>
        <th scope="col">状态</th><th scope="col">证据</th><th scope="col" aria-sort="descending">时间 / 安全</th></tr></thead>
      <tbody id="historyBody" role="rowgroup"></tbody></table></div>
    <footer class="workspace-toolbar pager"><span class="small muted" id="historyPagerLabel">—</span>
      <div class="row"><button class="btn small" id="historyPrev" disabled>上一页</button><button class="btn small" id="historyNext" disabled>下一页</button></div>
    </footer>
  </section>`;
}

export function mount() {
  const params = new URLSearchParams(location.search);
  for (const [key, selector] of Object.entries(filterFields)) {
    const field = $(selector);
    const value = params.get(key);
    if (key === "q") field.value = (value || "").slice(0, 200);
    else if ([...field.options].some(option => option.value === value)) field.value = value;
  }
  const initialOffset = Number(params.get("offset"));
  if (Number.isSafeInteger(initialOffset) && initialOffset > 0 && initialOffset < 10000000) offset = Math.floor(initialOffset / pageSize) * pageSize;
  let timer;
  const reset = () => { clearTimeout(timer); offset = 0; loadHistory(); };
  $("#historyFilters").addEventListener("submit", event => { event.preventDefault(); reset(); });
  $("#historyFilters").addEventListener("change", event => { if (event.target.matches("select")) reset(); });
  $("#historySearch").addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(reset, 220); });
  $("#historyRefresh").addEventListener("click", () => { clearTimeout(timer); loadHistory(); });
  $("#historyPrev").addEventListener("click", () => { offset -= pageSize; loadHistory(); });
  $("#historyNext").addEventListener("click", () => { offset += pageSize; loadHistory(); });
  $("#historyBody").addEventListener("click", event => {
    const button = event.target.closest("[data-open]");
    if (button) openDetail(button.dataset.open);
  });
  loadHistory();
  if (params.get("diagnosis")) openDetail(params.get("diagnosis"));
}
