import { $, $$, closeDrawer, initDevicePolling, loadLive } from "./core.js";
import * as overview from "./views/overview.js";
import * as diagnosis from "./views/diagnosis.js";
import * as simulator from "./views/simulator.js";
import * as events from "./views/events.js";
import * as history from "./views/history.js";
import * as knowledge from "./views/knowledge.js";

const routes = { overview, diagnosis, simulator, events, history, knowledge };
const titles = {
  overview: "设备总览", diagnosis: "AI 智能诊断", simulator: "仿真控制",
  events: "实时事件流", history: "诊断历史", knowledge: "维修知识库", settings: "设置",
};

const page = location.pathname.split("/").filter(Boolean).at(-1);
const module = routes[page];
document.body.dataset.page = page;
document.title = `${titles[page] || "控制台"} · PrintDiag Console`;
$(`[data-nav="${page}"]`)?.setAttribute("aria-current", "page");

if (module) {
  $("#view").innerHTML = module.render();
  module.mount();
} else {
  $("#view").innerHTML = '<header class="page-head"><div><h1>设置</h1><p class="page-sub">认证、用户偏好与权限管理规划中。</p></div></header>';
}

const toggle = $("#sidebarToggle");
const sidebar = $("#sidebar");
const sidebarBackdrop = $("#sidebarBackdrop");
const compact = matchMedia("(max-width: 900px)");

function setNavigation(open, restore = false) {
  sidebar.classList.toggle("open", open);
  sidebar.inert = compact.matches && !open;
  sidebarBackdrop.hidden = !open;
  $(".main").inert = open;
  toggle.setAttribute("aria-expanded", String(open));
  if (open) ($("#sidebar [aria-current]") || $("#sidebar nav a")).focus();
  else if (restore && compact.matches) toggle.focus();
}

toggle.addEventListener("click", () => setNavigation(true));
sidebarBackdrop.addEventListener("click", () => setNavigation(false, true));
compact.addEventListener("change", () => setNavigation(false));
document.addEventListener("keydown", event => {
  if (!sidebar.classList.contains("open")) return;
  if (event.key === "Escape") setNavigation(false, true);
  if (event.key !== "Tab") return;
  const links = $$("#sidebar a");
  const first = links[0];
  const last = links.at(-1);
  if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
  if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
});
setNavigation(false);
$("#drawerClose").addEventListener("click", closeDrawer);
$("#drawer").addEventListener("click", event => {
  if (event.target !== event.currentTarget) return;
  const { left, right, top, bottom } = event.currentTarget.getBoundingClientRect();
  if (event.clientX < left || event.clientX > right || event.clientY < top || event.clientY > bottom) closeDrawer();
});

initDevicePolling();
loadLive();
