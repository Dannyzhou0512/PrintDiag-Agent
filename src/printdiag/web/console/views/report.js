import { escapeHtml, statusChip } from "../core.js";

function summaryRows(view) {
  const t = view.technical;
  return [
    ["模型与来源", [view.model, view.source, t.model_source].filter(Boolean).join(" · ")],
    ["设备来源", t.device_source],
    ["报告状态", view.status_code],
    ["七字段校验", t.schema_valid === true ? "通过" : t.schema_valid === false ? "未通过" : "未记录"],
    ["V2 证据策略", [t.v2_decision, ...t.v2_codes].filter(Boolean).join(" · ")],
    ["Safety", t.safety],
    ["处理耗时", Number.isFinite(t.elapsed_ms) ? t.elapsed_ms + " ms" : ""],
    ["审计记录", t.audit],
  ];
}

export function reportMarkup(view, { saved = false } = {}) {
  const t = view.technical;
  const e = escapeHtml;
  const sources = t.references.map(hit => {
    const params = new URLSearchParams({ document: hit.document_id, chunk: hit.chunk_id });
    return `<li><a href="/console/knowledge?${e(params)}">${e(hit.source_title || "维护知识条目")}</a>
      <small>${e([hit.document_id, hit.chunk_id, hit.source_reference].filter(Boolean).join(" · "))}</small></li>`;
  }).join("");
  const receipts = t.receipts.map(ref =>
    `<li>${e([ref.tool, ref.evidence_id, ref.observed_at_utc].join(" · "))}</li>`).join("");
  const tone = view.safety.status === "blocked" ? "red" : view.safety.status === "review_required" ? "amber" : "slate";
  return `<div class="report-sheet">
    <header class="report-verdict">
      <div class="report-context">${e([view.model, view.source].filter(Boolean).join(" · "))}</div>
      <div class="report-verdict-line"><h2 class="report-title" tabindex="-1">${e(view.title)}</h2>
        ${statusChip(view.status, view.status_code === "failed" ? "red" : view.status_code === "complete" ? "slate" : "amber")}</div>
      ${view.category ? '<p class="muted">具体原因待验证</p>' : ""}
      ${view.legacy_summary ? `<p>${e(view.legacy_summary)}</p>` : ""}
    </header>
    <section class="report-section"><h3>判断依据</h3>
      ${view.basis.length ? `<dl class="report-basis">${view.basis.map(item =>
        `<div><dt>${e(item.source)}</dt><dd>${e(item.text)}</dd></div>`).join("")}</dl>`
        : '<p class="muted">暂无可展示的现象或观测。</p>'}</section>
    <section class="report-section"><h3>建议检查</h3>
      ${view.checks.length ? `<ol class="report-checks">${view.checks.map(item => `<li>${e(item)}</li>`).join("")}</ol>`
        : '<p class="muted">暂无可发布的检查建议。</p>'}</section>
    <section class="report-safety" data-tone="${tone}"><h3>安全提醒</h3>
      <p>${e(view.safety.notice || (view.safety.status === "clear" ? "本报告未检出需要拦截的维修建议。" : "本轮没有可用的安全审查结果。"))}</p></section>
    <section class="report-section"><h3>尚需确认</h3><p>${e(view.uncertainty)}</p></section>
    <details class="details report-technical"><summary>技术详情</summary>
      <dl class="kv">${summaryRows(view).map(([label, value]) =>
        `<div class="kv-row"><dt>${e(label)}</dt><dd>${e(value || "未记录")}</dd></div>`).join("")}</dl>
      <h3>知识检索引用</h3>${sources ? `<ul class="report-refs">${sources}</ul>` : '<p class="muted">未执行知识检索，或本轮无可用引用。</p>'}
      <h3>只读回执摘要</h3>${receipts ? `<ul>${receipts}</ul>` : '<p class="muted">无可引用的回执。</p>'}
      ${saved ? '<p class="small muted">报告来自已保存的诊断记录。</p>' : ""}
    </details>
  </div>`;
}

const markdownText = value => String(value ?? "").replace(/\s+/g, " ").replace(/[&<>]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[ch]))
  .replace(/[\\`*{}\[\]<>()#.!|~:+]/g, ch => "\\" + ch);

export function reportMarkdown(view) {
  const e = markdownText;
  return [
    "# PrintDiag 诊断报告", "", "## " + e(view.title), e(view.status),
    "", "## 判断依据", ...view.basis.map(item => `- ${e(item.source)}：${e(item.text)}`),
    "", "## 建议检查", ...(view.checks.length ? view.checks.map((item, i) => `${i + 1}. ${e(item)}`) : ["暂无可发布的检查建议。"]),
    "", "## 安全提醒", e(view.safety.notice || view.safety.label),
    "", "## 尚需确认", e(view.uncertainty),
    "", "<details>", "<summary>技术详情</summary>", "", ...summaryRows(view).map(([label, value]) => `- ${label}：${e(value)}`),
    "", "### RAG 来源", ...view.technical.references.map(hit => `- ${e(hit.document_id)} / ${e(hit.chunk_id)} / ${e(hit.source_reference)}`),
    "", "### 设备证据", ...view.technical.receipts.map(ref => `- ${e(ref.tool)} / ${e(ref.evidence_id)} / ${e(ref.observed_at_utc)}`),
    "", "</details>", "",
  ].join("\n");
}

export function reportText(view) {
  return [view.title, view.status, "", "判断依据", ...view.basis.map(item => `${item.source}：${item.text}`),
    "", "建议检查", ...view.checks.map((item, i) => `${i + 1}. ${item}`),
    "", "安全提醒", view.safety.notice || view.safety.label, "", "尚需确认", view.uncertainty].join("\n");
}
