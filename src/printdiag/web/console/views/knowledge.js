import {
  $, emptyState, escapeHtml, getJSON, icon, loadingState, openDrawer, statusChip,
} from "../core.js";

let inventory;
let category = "";
let rows = [];
let request;
let busy = false;
let timer;
const maxHits = 10;

function sameChunk(first, second) {
  return first.document_id === second.document_id && first.chunk_id === second.chunk_id;
}

function presentDoc(doc, index) {
  return `<button type="button" class="doc-result" data-doc="${index}">
    <span class="doc-heading"><strong>${escapeHtml(doc.title || doc.document_id)}</strong>${icon("arrow")}</span>
    <span>${escapeHtml(doc.excerpt || "")}</span>
    <span class="doc-meta">${statusChip(doc.fault_category || "类别未标注")}
      <small>${escapeHtml([doc.document_id, doc.chunk_id].filter(Boolean).join(" / "))}</small>
      ${Number.isFinite(doc.retrieval_score) ? statusChip(`BM25 ${doc.retrieval_score.toFixed(2)}`, "blue") : ""}</span>
    <small>${escapeHtml(doc.source_reference || "来源未记录")}</small>
  </button>`;
}

function openDoc(doc) {
  const checklist = (title, items) => items?.length ? `<section><h3>${title}</h3><ul>
    ${items.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul></section>` : "";
  openDrawer("知识片段", `<div class="msg-meta">${statusChip(doc.fault_category || "类别未标注")}</div>
    <div><h3>${escapeHtml(doc.title || doc.document_id)}</h3><p>${escapeHtml(doc.excerpt || "")}</p></div>
    <dl class="kv"><div class="kv-row"><dt>文档 / 片段</dt><dd>${escapeHtml([doc.document_id, doc.chunk_id].filter(Boolean).join(" / "))}</dd></div>
      <div class="kv-row"><dt>来源</dt><dd>${escapeHtml(doc.source_reference || "未记录")}</dd></div>
      ${doc.applicable_device ? `<div class="kv-row"><dt>适用设备</dt><dd>${escapeHtml(doc.applicable_device)}</dd></div>` : ""}
      ${Number.isFinite(doc.retrieval_score) ? `<div class="kv-row"><dt>BM25 分数</dt><dd>${escapeHtml(doc.retrieval_score)}</dd></div>` : ""}
    </dl>
    ${checklist("建议检查", doc.recommended_checks)}
    ${checklist("适用限制", doc.limitations)}
    <p class="notice small">此片段为知识检索背景。设备证据见相应诊断报告。</p>`);
}

function renderResults(found, query) {
  rows = category ? found.filter(doc => doc.fault_category === category) : found;
  $("#kbCount").textContent = query
    ? `${rows.length} 条命中${category ? " · 筛选本次 BM25 命中" : ""}`
    : `${rows.length} 个知识片段`;
  $("#kbList").innerHTML = rows.length ? rows.map(presentDoc).join("") :
    emptyState(query ? "没有匹配的知识片段" : "此分类暂无知识片段",
      query ? "可尝试部件名、症状关键词或切换分类。" : "");
}

async function showDocs() {
  if (!inventory?.available || busy) return;
  request?.abort();
  const query = $("#kbSearch").value.trim();
  if (!query) { renderResults(inventory.documents, ""); return; }
  const controller = new AbortController();
  request = controller;
  $("#kbCount").textContent = "检索中";
  $("#kbList").innerHTML = loadingState("正在检索 BM25 索引");
  try {
    const params = new URLSearchParams({ q: query, limit: maxHits });
    const { hits = [] } = await getJSON(`/api/knowledge/search?${params}`, { signal: controller.signal });
    if (request !== controller || controller.signal.aborted) return;
    renderResults(hits.map(hit => ({ ...inventory.documents.find(doc => sameChunk(doc, hit)), ...hit })), query);
  } catch (error) {
    if (request !== controller || controller.signal.aborted) return;
    $("#kbCount").textContent = "检索失败";
    $("#kbList").innerHTML = `${emptyState("知识检索失败", error.message)}<button class="btn" id="retryKnowledgeSearch">重试检索</button>`;
    $("#retryKnowledgeSearch").addEventListener("click", showDocs);
  }
}

async function loadInventory(reference) {
  if (busy) return;
  request?.abort();
  busy = true;
  const refresh = $("#kbRefresh");
  const list = $("#kbList");
  refresh.disabled = true;
  $("#kbSearch").disabled = true;
  $("#kbSubmit").disabled = true;
  $("#kbCategory").disabled = true;
  $("#kbCount").textContent = "读取中";
  list.innerHTML = loadingState("读取维修知识库");
  try {
    const data = await getJSON("/api/knowledge/overview");
    inventory = data;
    const categories = data.categories || [];
    if (!categories.includes(category)) category = "";
    const select = $("#kbCategory");
    select.replaceChildren(new Option("全部分类", ""), ...categories.map(name => new Option(name, name)));
    select.value = category;
    if (!data.available) {
      $("#kbCount").textContent = "索引不可用";
      list.innerHTML = emptyState("知识索引暂不可用", "可以重新加载知识库。");
      return;
    }
    $("#kbSearch").disabled = false;
    $("#kbSubmit").disabled = false;
    select.disabled = false;
    if (reference?.document_id) {
      const doc = data.documents.find(item => reference.chunk_id ? sameChunk(item, reference) : item.document_id === reference.document_id);
      if (doc) openDoc(doc);
      else $("#kbReference").hidden = false;
    }
  } catch (error) {
    inventory = undefined;
    $("#kbCount").textContent = "读取失败";
    list.innerHTML = emptyState("知识库读取失败", error.message);
  } finally {
    busy = false;
    refresh.disabled = false;
  }
  showDocs();
}

export function render() {
  return `<header class="page-head"><div><h1>维修知识库</h1>
    <p class="page-sub">在内置知识中检索故障现象、部件和排查步骤。</p></div>
    <button class="btn" id="kbRefresh">${icon("refresh")}重新加载</button></header>
  <section class="card kb-workspace">
    <div class="workspace-toolbar"><h2 class="card-title">BM25 检索</h2>${statusChip("只读")}</div>
    <div class="card-body">
      <form class="kb-search" id="kbForm">
        <label class="search">${icon("search")}<input class="input" id="kbSearch" type="search" maxlength="120" aria-label="搜索维修知识" placeholder="例如：喷嘴升温、欠挤出、限位开关" disabled></label>
        <select class="select" id="kbCategory" aria-label="知识分类" disabled><option value="">全部分类</option></select>
        <button class="btn primary" id="kbSubmit" disabled>搜索</button>
      </form>
      <p class="notice" id="kbReference" hidden>当前索引中找不到诊断引用的片段。</p>
      <div class="kb-results-head"><h3>知识片段</h3><span class="small muted" id="kbCount" role="status"></span></div>
      <div id="kbList"></div>
    </div>
    <footer class="workspace-toolbar small muted">上传、删除和版本管理规划中</footer>
  </section>`;
}

export function mount() {
  const params = new URLSearchParams(location.search);
  $("#kbRefresh").addEventListener("click", () => loadInventory());
  $("#kbForm").addEventListener("submit", event => { event.preventDefault(); clearTimeout(timer); showDocs(); });
  $("#kbSearch").addEventListener("input", () => {
    clearTimeout(timer);
    request?.abort();
    timer = setTimeout(showDocs, 260);
  });
  $("#kbCategory").addEventListener("change", event => { category = event.target.value; clearTimeout(timer); showDocs(); });
  $("#kbList").addEventListener("click", event => {
    const button = event.target.closest("[data-doc]");
    if (button) openDoc(rows[Number(button.dataset.doc)]);
  });
  loadInventory({ document_id: params.get("document"), chunk_id: params.get("chunk") });
}
