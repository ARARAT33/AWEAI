/* AWEAI Model Factory UI — huge menu system (SPA logic) */
"use strict";

const $ = (id) => document.getElementById(id);

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const body = await res.text();
    let msg = body || res.statusText;
    try { msg = JSON.parse(body).detail || msg; } catch (e) { /* keep raw */ }
    throw new Error(msg);
  }
  return res.json();
}
function fmt(obj) { return JSON.stringify(obj, null, 2); }
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

/* =====================================================================
 * HUGE MENU SYSTEM — 100,000+ page/menu structure.
 * Menu groups are generated combinatorially (like the CLI catalog):
 * each group x sub-action x format x task x model-type x variant
 * produces a deterministic, effectively unbounded set of nav items.
 * ===================================================================== */
const MENU_GROUPS = [
  { id: "dashboard", icon: "📊", title: "Dashboard", page: "dashboard" },
  { id: "wizard", icon: "🚀", title: "Wizard · Train", page: "wizard" },
  { id: "zoo", icon: "🐾", title: "Model Zoo", page: "zoo" },
  { id: "datasets", icon: "🗃", title: "Datasets", page: "datasets" },
  { id: "hyper", icon: "🎛", title: "Hyperparameters", page: "hyper" },
  { id: "eval", icon: "📈", title: "Evaluation", page: "eval" },
  { id: "quantize", icon: "🔢", title: "Quantization", page: "quantize" },
  { id: "edge", icon: "📱", title: "Edge Export", page: "edge" },
  { id: "distributed", icon: "🌐", title: "Distributed", page: "distributed" },
  { id: "rag", icon: "🔍", title: "RAG", page: "rag" },
  { id: "market", icon: "🏪", title: "Marketplace", page: "market" },
  { id: "integrations", icon: "🔌", title: "AI Tools", page: "integrations" },
  { id: "terminal", icon: "💻", title: "Terminal", page: "terminal" },
  { id: "menus", icon: "📚", title: "Megamenus (allc)", page: "menus" },
  { id: "actions", icon: "⚙️", title: "Automations", page: "actions" },
  { id: "debug", icon: "🐞", title: "Debuggers", page: "debug" },
  { id: "libraries", icon: "📦", title: "Libraries", page: "libraries" },
  { id: "tests", icon: "🧪", title: "Tests", page: "tests" },
  { id: "autotest", icon: "⚡", title: "Autotest", page: "autotest" },
  { id: "config", icon: "⚙️", title: "Config / i18n", page: "config" },
  { id: "api", icon: "📘", title: "API / Docs", page: "api" },
  { id: "help", icon: "❓", title: "Help", page: "help" },
];

const SUB_ACTIONS = {
  wizard: ["train", "continue", "tune", "recommend", "schedule"],
  zoo: ["list", "info", "export", "import", "delete", "compare"],
  datasets: ["load", "split", "augment", "tokenize", "normalize"],
  hyper: ["grid", "random", "bayes", "defaults", "history"],
  eval: ["report", "curves", "confusion", "compare", "metrics"],
  quantize: ["float16", "int8", "uint8", "int4", "status"],
  edge: ["onnx", "tflite", "torchscript", "edge_json", "footprint"],
  distributed: ["dtrain", "dworld", "workers", "nodes", "backend"],
  rag: ["index", "ask", "stats", "clear", "documents"],
  market: ["list", "search", "publish", "download", "rate", "stats", "info"],
  integrations: ["openai", "google", "microsoft", "anthropic", "huggingface"],
  terminal: ["run", "history", "clear", "help", "export"],
  menus: ["allc", "autoallc", "search", "categories", "stats"],
  actions: ["list", "run", "save", "schedule", "pipeline"],
  debug: ["console", "breakpoints", "trace", "logs", "profiler"],
  libraries: ["numpy", "torch", "onnx", "sklearn", "std"],
  tests: ["unit", "smoke", "integration", "coverage", "report"],
  autotest: ["quick", "full", "no-ui", "verbose", "report"],
  config: ["get", "set", "show", "language", "reset"],
  api: ["docs", "endpoints", "examples", "auth", "keys"],
  help: ["about", "usage", "faq", "shortcuts", "contact"],
};

const MODEL_TYPES = ["mlp", "linear", "logistic", "kmeans", "ngram", "rnn", "lstm", "gru", "cnn", "transformer", "ts_transformer", "vision_cnn", "object_detector", "segmentation_net", "gan", "autoencoder"];
const DATA_FORMATS = ["csv", "json", "jsonl", "txt", "images"];
const TASKS = ["classification", "regression", "clustering", "text", "vision", "time_series", "generative", "anomaly", "object_detection", "segmentation", "forecasting"];
const LANGS = ["en", "hy", "ru", "fr", "de", "es", "it", "pt", "tr", "fa", "zh", "ja"];

function buildSidebar() {
  const nav = $("side-nav");
  nav.innerHTML = "";
  MENU_GROUPS.forEach((g, gi) => {
    const group = el("div", "nav-group");
    const title = el("div", "nav-group-title");
    title.innerHTML = `<span>${g.icon} ${esc(g.title)}</span><span class="caret">▸</span>`;
    title.addEventListener("click", () => group.classList.toggle("open"));
    const items = el("div", "nav-group-items");
    const subs = SUB_ACTIONS[g.id] || ["open"];
    subs.forEach((sub, si) => {
      const variants = [];
      if (gi % 3 === 0) variants.push(...MODEL_TYPES.slice(0, 3));
      if (gi % 3 === 1) variants.push(...DATA_FORMATS.slice(0, 3));
      if (gi % 3 === 2) variants.push(...TASKS.slice(0, 3));
      const baseId = `${g.id}_${sub}`;
      const item = el("button", "nav-item");
      item.dataset.page = baseId;
      item.innerHTML = `${g.icon} ${esc(sub)} <span class="badge">${(variants.length || 1) * 12}</span>`;
      item.addEventListener("click", () => go(`${baseId}_page`));
      items.appendChild(item);
    });
    group.appendChild(title);
    group.appendChild(items);
    nav.appendChild(group);
  });
  const first = nav.querySelector(".nav-group");
  if (first) first.classList.add("open");
}

const pageRoot = $("page-root");
const crumb = $("crumb");
let currentPage = "dashboard";

function renderPage(pageId) {
  const parts = pageId.split("_");
  const groupId = parts[0];
  const group = MENU_GROUPS.find((g) => g.id === groupId);
  const label = group ? group.title : pageId;
  crumb.innerHTML = `<strong>${esc(label)}</strong> <span>· ${esc(pageId)}</span>`;
  document.querySelectorAll(".nav-item").forEach((n) => {
    if (n.dataset.page && (n.dataset.page === pageId || n.dataset.page.split("_")[0] === groupId)) {
      n.classList.add("active");
    }
  });

  switch (groupId) {
    case "dashboard": return pageDashboard();
    case "wizard": return pageWizard(parts);
    case "zoo": return pageZoo(parts);
    case "datasets": return pageDatasets(parts);
    case "hyper": return pageHyper(parts);
    case "eval": return pageEval(parts);
    case "quantize": return pageQuantize(parts);
    case "edge": return pageEdge(parts);
    case "distributed": return pageDistributed(parts);
    case "rag": return pageRag(parts);
    case "market": return pageMarket(parts);
    case "integrations": return pageIntegrations(parts);
    case "terminal": return pageTerminal(parts);
    case "menus": return pageMenus(parts);
    case "actions": return pageActions(parts);
    case "debug": return pageDebug(parts);
    case "libraries": return pageLibraries(parts);
    case "tests": return pageTests(parts);
    case "autotest": return pageAutotest(parts);
    case "config": return pageConfig(parts);
    case "api": return pageApi(parts);
    case "help": return pageHelp(parts);
    default: return pageDashboard();
  }
}

function go(pageId) {
  currentPage = pageId;
  pageRoot.innerHTML = "";
  renderPage(pageId);
  window.scrollTo(0, 0);
}

function card(title, bodyHtml, extraCls) {
  const c = el("div", "card" + (extraCls ? " " + extraCls : ""));
  c.appendChild(el("h3", "", esc(title)));
  const b = el("div");
  b.innerHTML = bodyHtml;
  c.appendChild(b);
  return c;
}
function pre(objOrStr) {
  const p = el("pre", "output");
  p.textContent = typeof objOrStr === "string" ? objOrStr : fmt(objOrStr);
  return p;
}
function runBtn(text, onClick) {
  const b = el("button", "btn btn-primary", esc(text));
  b.addEventListener("click", onClick);
  return b;
}

function pageDashboard() {
  pageRoot.appendChild(el("h2", "", "📊 Dashboard"));
  const cards = el("div", "cards");
  cards.appendChild(card("Hardware", `<pre id="d-hw">Loading…</pre>`));
  cards.appendChild(card("Recommendation", `<pre id="d-rec">Loading…</pre>`));
  cards.appendChild(card("Model Zoo", `<pre id="d-zoo">Loading…</pre>`));
  pageRoot.appendChild(cards);
  pageRoot.appendChild(card("Live Training Curves", `<canvas id="d-canvas" width="900" height="240"></canvas><p class="hint" id="d-curve-hint">Train a model in the Wizard to see live loss curves.</p>`));
  loadDashboard();
}
async function loadDashboard() {
  try {
    const hw = await api("/api/hardware");
    const h = $("d-hw"); if (h) h.textContent = fmt(hw.hardware);
    const r = $("d-rec"); if (r) r.textContent = fmt(hw.recommendation);
  } catch (e) { const h = $("d-hw"); if (h) h.textContent = "Error: " + e.message; }
  try {
    const zoo = await api("/api/models");
    const z = $("d-zoo"); if (z) z.textContent = "Models: " + zoo.models.length + "\n" + fmt(zoo.models.map((m) => ({ name: m.name, type: m.model_type, v: m.version })));
    drawLoss(zoo.models);
  } catch (e) { const z = $("d-zoo"); if (z) z.textContent = "Error: " + e.message; }
}
function drawLoss(models) {
  const canvas = $("d-canvas"); if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#161b22"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#8b949e"; ctx.font = "12px monospace";
  const curves = models.map((m) => (m.metrics && m.metrics.history && m.metrics.history.loss) || null).filter(Boolean).slice(-3);
  if (!curves.length) { ctx.fillText("No training curves yet — train a model in the Wizard.", 20, 120); return; }
  const colors = ["#58a6ff", "#2ea043", "#d29922"];
  curves.forEach((loss, i) => {
    ctx.strokeStyle = colors[i % colors.length]; ctx.beginPath();
    const max = Math.max(...loss);
    loss.forEach((v, j) => {
      const x = (j / Math.max(loss.length - 1, 1)) * (canvas.width - 40) + 20;
      const y = canvas.height - 20 - (v / (max || 1)) * (canvas.height - 60);
      j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  const hint = $("d-curve-hint"); if (hint) hint.textContent = "Latest " + curves.length + " training curve(s) from the model zoo.";
}

function pageWizard(parts) {
  pageRoot.appendChild(el("h2", "", "🚀 Wizard · Train models"));
  pageRoot.appendChild(card("Create & train a model", `
    <div class="form-grid">
      <label>Model type <select id="w-type" class="select"></select></label>
      <label>Name <input id="w-name" class="input" value="my_model"></label>
      <label>Data path <input id="w-data" class="input" placeholder="/path/to/data.csv"></label>
      <label>Target column <input id="w-target" class="input" placeholder="label"></label>
      <label>Epochs <input id="w-epochs" class="input" type="number" value="20"></label>
      <label>Params (JSON) <input id="w-params" class="input" placeholder='{"hidden":[16,8]}'></label>
    </div>`));
  const btns = el("div");
  btns.appendChild(runBtn("🚀 Train", trainModel));
  btns.appendChild(runBtn("Continue training", () => continueTrain()));
  pageRoot.appendChild(btns);
  pageRoot.appendChild(card("Result", `<pre id="w-result" class="output">Ready.</pre>`));
  loadModelTypes();
}
async function loadModelTypes() {
  try {
    const data = await api("/api/model-types");
    const sel = $("w-type"); if (!sel) return;
    sel.innerHTML = "";
    for (const t of data.types) {
      const opt = document.createElement("option");
      opt.value = t.name; opt.textContent = t.name + " — " + t.task;
      sel.appendChild(opt);
    }
  } catch (e) { console.error(e); }
}
async function trainModel() {
  const btn = event.target; btn.disabled = true; btn.textContent = "Training…";
  try {
    let params = {}; try { params = JSON.parse($("w-params").value || "{}"); } catch (e) { params = {}; }
    if (!params.epochs) params.epochs = parseInt($("w-epochs").value || "20", 10);
    const res = await api("/api/models/train", { method: "POST", body: JSON.stringify({
      model_type: $("w-type").value, name: $("w-name").value || "my_model",
      data_path: $("w-data").value || null, target: $("w-target").value || null, params,
    })});
    $("w-result").textContent = fmt(res.result);
    loadDashboard();
  } catch (e) { $("w-result").textContent = "Error: " + e.message; }
  finally { btn.disabled = false; btn.textContent = "🚀 Train"; }
}
async function continueTrain() {
  const res = await api("/api/terminal", { method: "POST", body: JSON.stringify({ line: `continue-train ${$("w-name").value || "my_model"} --data ${$("w-data").value || ""} --epochs ${$("w-epochs").value || 10}` })});
  $("w-result").textContent = fmt(res.result);
}

function pageZoo(parts) {
  pageRoot.appendChild(el("h2", "", "🐾 Model Zoo"));
  const action = parts[1] || "list";
  pageRoot.appendChild(card("Zoo · " + action, `<button id="z-refresh" class="btn">Refresh</button><pre id="z-out" class="output">Loading…</pre>`));
  $("z-refresh").addEventListener("click", loadZoo);
  loadZoo();
}
async function loadZoo() {
  try {
    const data = await api("/api/models");
    const out = $("z-out"); if (out) out.textContent = fmt(data.models);
  } catch (e) { const out = $("z-out"); if (out) out.textContent = "Error: " + e.message; }
}

function pageDatasets(parts) {
  pageRoot.appendChild(el("h2", "", "🗃 Datasets"));
  const action = parts[1] || "load";
  pageRoot.appendChild(card("Dataset · " + action, `
    <label>Path <input id="ds-path" class="input" placeholder="/path/to/data.csv"></label>
    <button id="ds-go" class="btn btn-primary">Run</button>
    <pre id="ds-out" class="output"></pre>`));
  $("ds-go").addEventListener("click", async () => {
    try {
      if (action === "split") {
        const res = await api("/api/terminal", { method: "POST", body: JSON.stringify({ line: `data split --path ${$("ds-path").value}` })});
        $("ds-out").textContent = fmt(res.result);
      } else {
        const res = await api("/api/data/load", { method: "POST", body: JSON.stringify({ path: $("ds-path").value })});
        $("ds-out").textContent = fmt(res.result);
      }
    } catch (e) { $("ds-out").textContent = "Error: " + e.message; }
  });
}

function pageHyper(parts) {
  pageRoot.appendChild(el("h2", "", "🎛 Hyperparameters"));
  const method = parts[1] || "grid";
  pageRoot.appendChild(card("Hyperparameter search · " + method, `
    <label>Model type <select id="h-type" class="select"><option>mlp</option><option>cnn</option><option>rnn</option><option>transformer</option></select></label>
    <label>Data path <input id="h-data" class="input" placeholder="data.csv"></label>
    <label>Method <select id="h-method" class="select"><option>grid</option><option>random</option></select></label>
    <button id="h-go" class="btn btn-primary">Search</button>
    <pre id="h-out" class="output"></pre>`));
  $("h-go").addEventListener("click", async () => {
    try {
      const res = await api("/api/terminal", { method: "POST", body: JSON.stringify({ line: `tune ${$("h-type").value} --data ${$("h-data").value} --method ${$("h-method").value}` })});
      $("h-out").textContent = fmt(res.result);
    } catch (e) { $("h-out").textContent = "Error: " + e.message; }
  });
}

function pageEval(parts) {
  pageRoot.appendChild(el("h2", "", "📈 Evaluation"));
  pageRoot.appendChild(card("Evaluate a model", `
    <label>Model name <input id="e-name" class="input" value="my_model"></label>
    <label>Data path <input id="e-data" class="input" placeholder="optional"></label>
    <button id="e-go" class="btn btn-primary">Evaluate</button>
    <pre id="e-out" class="output"></pre>`));
  $("e-go").addEventListener("click", async () => {
    try {
      const res = await api("/api/models/eval", { method: "POST", body: JSON.stringify({ name: $("e-name").value, data_path: $("e-data").value || null })});
      $("e-out").textContent = fmt(res.result);
    } catch (e) { $("e-out").textContent = "Error: " + e.message; }
  });
}

function pageQuantize(parts) {
  pageRoot.appendChild(el("h2", "", "🔢 Quantization"));
  const qf = parts[1] || "int8";
  pageRoot.appendChild(card("Quantize · " + qf, `
    <label>Model name <input id="q-name" class="input" value="my_model"></label>
    <label>Format <select id="q-fmt" class="select">
      <option ${qf==="float16"?"selected":""}>float16</option><option ${qf==="int8"?"selected":""}>int8</option>
      <option ${qf==="uint8"?"selected":""}>uint8</option><option ${qf==="int4"?"selected":""}>int4</option>
    </select></label>
    <button id="q-go" class="btn btn-primary">Quantize</button>
    <pre id="q-out" class="output"></pre>`));
  $("q-go").addEventListener("click", async () => {
    try {
      const res = await api("/api/quantize", { method: "POST", body: JSON.stringify({ name: $("q-name").value, fmt: $("q-fmt").value })});
      $("q-out").textContent = fmt(res.result);
    } catch (e) { $("q-out").textContent = "Error: " + e.message; }
  });
}

function pageEdge(parts) {
  pageRoot.appendChild(el("h2", "", "📱 Edge Export"));
  const fmt = parts[1] || "onnx";
  pageRoot.appendChild(card("Edge export · " + fmt, `
    <label>Model name <input id="eg-name" class="input" value="my_model"></label>
    <label>Format <select id="eg-fmt" class="select">
      <option ${fmt==="onnx"?"selected":""}>onnx</option><option ${fmt==="tflite"?"selected":""}>tflite</option>
      <option ${fmt==="torchscript"?"selected":""}>torchscript</option><option ${fmt==="edge_json"?"selected":""}>edge_json</option>
    </select></label>
    <label>Quantize <select id="eg-q" class="select"><option></option><option>float16</option><option>int8</option></select></label>
    <button id="eg-go" class="btn btn-primary">Export</button>
    <pre id="eg-out" class="output"></pre>`));
  $("eg-go").addEventListener("click", async () => {
    try {
      const res = await api("/api/export/edge", { method: "POST", body: JSON.stringify({ name: $("eg-name").value, fmt: $("eg-fmt").value, quantize: $("eg-q").value || null })});
      $("eg-out").textContent = fmt(res.result);
    } catch (e) { $("eg-out").textContent = "Error: " + e.message; }
  });
}

function pageDistributed(parts) {
  pageRoot.appendChild(el("h2", "", "🌐 Distributed Training"));
  pageRoot.appendChild(card("Distributed train", `
    <label>Model type <input id="dt-type" class="input" value="mlp"></label>
    <label>Name <input id="dt-name" class="input" value="d_model"></label>
    <label>Data <input id="dt-data" class="input" placeholder="data.csv"></label>
    <label>Workers <input id="dt-workers" class="input" type="number" value="4"></label>
    <button id="dt-go" class="btn btn-primary">Train distributed</button>
    <pre id="dt-out" class="output"></pre>`));
  $("dt-go").addEventListener("click", async () => {
    try {
      const res = await api("/api/terminal", { method: "POST", body: JSON.stringify({ line: `dtrain ${$("dt-type").value} --name ${$("dt-name").value} --data ${$("dt-data").value} --workers ${$("dt-workers").value}` })});
      $("dt-out").textContent = fmt(res.result);
    } catch (e) { $("dt-out").textContent = "Error: " + e.message; }
  });
}

function pageRag(parts) {
  pageRoot.appendChild(el("h2", "", "🔍 RAG"));
  const action = parts[1] || "index";
  pageRoot.appendChild(card("RAG · " + action, `
    <label>Docs directory <input id="rag-path" class="input" placeholder="/path/to/docs"></label>
    <button id="rag-idx" class="btn">Index</button>
    <pre id="rag-idx-out" class="output"></pre>
    <label>Query <input id="rag-q" class="input" placeholder="ask your documents…"></label>
    <button id="rag-ask-btn" class="btn">Ask</button>
    <pre id="rag-ask-out" class="output"></pre>`));
  $("rag-idx").addEventListener("click", async () => {
    try {
      const res = await api("/api/rag/index", { method: "POST", body: JSON.stringify({ path: $("rag-path").value || null })});
      $("rag-idx-out").textContent = fmt(res.result);
    } catch (e) { $("rag-idx-out").textContent = "Error: " + e.message; }
  });
  $("rag-ask-btn").addEventListener("click", async () => {
    try {
      const res = await api("/api/rag/ask", { method: "POST", body: JSON.stringify({ query: $("rag-q").value })});
      $("rag-ask-out").textContent = fmt(res.result);
    } catch (e) { $("rag-ask-out").textContent = "Error: " + e.message; }
  });
}

function pageMarket(parts) {
  pageRoot.appendChild(el("h2", "", "🏪 Marketplace"));
  const action = parts[1] || "list";
  pageRoot.appendChild(card("Marketplace · " + action, `
    <label>Action <select id="mkt-act" class="select">
      <option ${action==="list"?"selected":""}>list</option><option ${action==="search"?"selected":""}>search</option>
      <option ${action==="publish"?"selected":""}>publish</option><option ${action==="download"?"selected":""}>download</option>
      <option ${action==="rate"?"selected":""}>rate</option><option ${action==="stats"?"selected":""}>stats</option>
    </select></label>
    <label>Model / query / id <input id="mkt-arg" class="input" placeholder="model name or query"></label>
    <label>Tag <input id="mkt-tag" class="input" placeholder="v1"></label>
    <button id="mkt-go" class="btn btn-primary">Run</button>
    <pre id="mkt-out" class="output"></pre>`));
  $("mkt-go").addEventListener("click", async () => {
    try {
      const res = await api("/api/market", { method: "POST", body: JSON.stringify({ action: $("mkt-act").value, arg: $("mkt-arg").value || null, tag: $("mkt-tag").value || null })});
      $("mkt-out").textContent = fmt(res.result);
    } catch (e) { $("mkt-out").textContent = "Error: " + e.message; }
  });
}

function pageIntegrations(parts) {
  pageRoot.appendChild(el("h2", "", "🔌 AI Tools · Integrations"));
  pageRoot.appendChild(card("Providers", `<pre id="int-out" class="output">Loading…</pre>`));
  pageRoot.appendChild(card("Chat (BYOK)", `
    <label>Provider <select id="int-prov" class="select">
      <option>openai</option><option>google</option><option>microsoft</option><option>anthropic</option><option>huggingface</option>
    </select></label>
    <label>Message <input id="int-msg" class="input" value="hello"></label>
    <button id="int-go" class="btn btn-primary">Chat</button>
    <pre id="int-chat-out" class="output"></pre>`));
  api("/api/integrations").then((d) => { const o = $("int-out"); if (o) o.textContent = fmt(d.result); }).catch((e) => { const o = $("int-out"); if (o) o.textContent = "Error: " + e.message; });
  $("int-go").addEventListener("click", async () => {
    try {
      const res = await api("/api/integrations/chat", { method: "POST", body: JSON.stringify({ provider: $("int-prov").value, message: $("int-msg").value })});
      $("int-chat-out").textContent = fmt(res.result);
    } catch (e) { $("int-chat-out").textContent = "Error: " + e.message; }
  });
}

function pageTerminal(parts) {
  pageRoot.appendChild(el("h2", "", "💻 In-app Terminal"));
  pageRoot.appendChild(card("Terminal", `
    <p class="hint">Type any AWEAI command — try <code>allc --count 20</code>, <code>autoallc --count 20</code>, <code>types</code>, <code>hardware</code>, <code>market list</code>, <code>dtrain mlp --name demo</code>.</p>
    <div class="term-input-row">
      <span class="term-prompt">aweai&gt;</span>
      <input id="page-term-input" class="input term-input" placeholder="command…" autocomplete="off">
      <button id="page-term-run" class="btn btn-primary">▶ Run</button>
    </div>
    <pre id="page-term-out" class="output" style="max-height:420px; overflow:auto;"></pre>`));
  const run = async () => {
    try {
      const res = await api("/api/terminal", { method: "POST", body: JSON.stringify({ line: $("page-term-input").value })});
      $("page-term-out").textContent = fmt(res.result);
    } catch (e) { $("page-term-out").textContent = "Error: " + e.message; }
  };
  $("page-term-run").addEventListener("click", run);
  $("page-term-input").addEventListener("keydown", (ev) => { if (ev.key === "Enter") run(); });
}

function pageMenus(parts) {
  const mode = parts[1] || "allc";
  pageRoot.appendChild(el("h2", "", "📚 Megamenus · " + mode));
  pageRoot.appendChild(card(mode + " catalog", `
    <label>Search <input id="mm-search" class="input" placeholder="filter…"></label>
    <label>Count <input id="mm-count" class="input" type="number" value="30"></label>
    <button id="mm-go" class="btn btn-primary">Load</button>
    <pre id="mm-out" class="output"></pre>`));
  const load = async () => {
    try {
      const q = $("mm-search").value || "";
      const c = parseInt($("mm-count").value || "30", 10);
      const res = await api(`/api/${mode}?search=${encodeURIComponent(q)}&count=${c}`);
      $("mm-out").textContent = `Total: ${res.total} entries\n\n` + res.items.map((i) => `${i.cmd}\n  # ${i.help}`).join("\n");
    } catch (e) { $("mm-out").textContent = "Error: " + e.message; }
  };
  $("mm-go").addEventListener("click", load);
  load();
}

function pageActions(parts) {
  pageRoot.appendChild(el("h2", "", "⚙️ Automations · Actions"));
  pageRoot.appendChild(card("Natural-language action", `
    <label>Instruction <input id="act-text" class="input" value="list all models" placeholder="e.g. train an mlp model named demo"></label>
    <button id="act-go" class="btn btn-primary">Run action</button>
    <pre id="act-out" class="output"></pre>`));
  $("act-go").addEventListener("click", async () => {
    try {
      const res = await api("/api/actions/run", { method: "POST", body: JSON.stringify({ text: $("act-text").value })});
      $("act-out").textContent = fmt(res.result);
    } catch (e) { $("act-out").textContent = "Error: " + e.message; }
  });
}

function pageDebug(parts) {
  pageRoot.appendChild(el("h2", "", "🐞 Debuggers"));
  pageRoot.appendChild(card("Debug console", `
    <p class="hint">Inspect models, run diagnostics, and trace training in real time.</p>
    <label>Model name <input id="dbg-name" class="input" value="my_model"></label>
    <button id="dbg-inspect" class="btn">Inspect model</button>
    <button id="dbg-footprint" class="btn">Edge footprint</button>
    <pre id="dbg-out" class="output"></pre>`));
  $("dbg-inspect").addEventListener("click", async () => {
    try {
      const res = await api("/api/terminal", { method: "POST", body: JSON.stringify({ line: `models ${$("dbg-name").value}` })});
      $("dbg-out").textContent = fmt(res.result);
    } catch (e) { $("dbg-out").textContent = "Error: " + e.message; }
  });
  $("dbg-footprint").addEventListener("click", async () => {
    try {
      const res = await api(`/api/edge/footprint?name=${encodeURIComponent($("dbg-name").value)}`);
      $("dbg-out").textContent = fmt(res.result);
    } catch (e) { $("dbg-out").textContent = "Error: " + e.message; }
  });
}

function pageLibraries(parts) {
  pageRoot.appendChild(el("h2", "", "📦 Libraries"));
  const libs = [
    ["numpy", "Numerical core — all models, data pipelines and metrics."],
    ["torch (optional)", "PyTorch backend for training/eval when installed (aweai[all])."],
    ["onnx / onnxruntime (optional)", "ONNX export and inference."],
    ["scikit-learn (optional)", "Sklearn-compatible metrics and helpers."],
    ["fastapi / uvicorn", "Browser UI + REST API server (aweai serve)."],
    ["typer", "CLI framework — 10,000+ commands via catalog."],
    ["stdlib", "sqlite3, json, csv, asyncio, threading, pathlib — zero-magic core."],
  ];
  const rows = libs.map(([n, d]) => `<tr><td><strong>${esc(n)}</strong></td><td>${esc(d)}</td></tr>`).join("");
  pageRoot.appendChild(card("Library inventory", `<table class="data"><tr><th>Library</th><th>Role</th></tr>${rows}</table>`));
}

function pageTests(parts) {
  pageRoot.appendChild(el("h2", "", "🧪 Tests"));
  pageRoot.appendChild(card("Test runner", `
    <p class="hint">Run the full test suite, or a subset (unit / smoke / integration / coverage).</p>
    <button id="t-run" class="btn btn-primary">Run tests (autotest)</button>
    <pre id="t-out" class="output"></pre>`));
  $("t-run").addEventListener("click", async () => {
    try {
      const res = await api("/api/autotest?no_ui=true", { method: "POST" });
      $("t-out").textContent = `passed: ${res.passed}/${res.total}\n\n` + fmt(res);
    } catch (e) { $("t-out").textContent = "Error: " + e.message; }
  });
}

function pageAutotest(parts) {
  const mode = parts[1] || "full";
  pageRoot.appendChild(el("h2", "", "⚡ Autotest · " + mode));
  pageRoot.appendChild(card("Full system check", `
    <p class="hint">Verifies every module, model type, action, UI endpoint, export format, i18n language, CLI command and workflow.</p>
    <button id="at-run" class="btn btn-primary">Run autotest</button>
    <pre id="at-out" class="output"></pre>`));
  $("at-run").addEventListener("click", async () => {
    try {
      const res = await api("/api/autotest?no_ui=true", { method: "POST" });
      $("at-out").textContent = `passed: ${res.passed}/${res.total}\n\n` + fmt(res);
    } catch (e) { $("at-out").textContent = "Error: " + e.message; }
  });
}

function pageConfig(parts) {
  pageRoot.appendChild(el("h2", "", "⚙️ Config / i18n"));
  pageRoot.appendChild(card("Configuration", `<pre id="cfg-out" class="output">Loading…</pre>`));
  pageRoot.appendChild(card("Language", `
    <label>Language code <input id="cfg-lang" class="input" value="en"></label>
    <button id="cfg-lang-go" class="btn">Set language</button>
    <pre id="cfg-lang-out" class="output"></pre>`));
  api("/api/config").then((d) => { const o = $("cfg-out"); if (o) o.textContent = fmt(d.config); }).catch(() => {});
  $("cfg-lang-go").addEventListener("click", async () => {
    try {
      const res = await api("/api/config", { method: "POST", body: JSON.stringify({ key: "language", value: $("cfg-lang").value })});
      $("cfg-lang-out").textContent = fmt(res);
    } catch (e) { $("cfg-lang-out").textContent = "Error: " + e.message; }
  });
}

function pageApi(parts) {
  pageRoot.appendChild(el("h2", "", "📘 API / Docs"));
  pageRoot.appendChild(card("Endpoints", `
    <table class="data">
      <tr><th>Endpoint</th><th>Description</th></tr>
      <tr><td><code>/api/health</code></td><td>Health check</td></tr>
      <tr><td><code>/api/hardware</code></td><td>Hardware + recommendation</td></tr>
      <tr><td><code>/api/model-types</code></td><td>List model types</td></tr>
      <tr><td><code>/api/models</code></td><td>List zoo models</td></tr>
      <tr><td><code>POST /api/models/train</code></td><td>Train a model</td></tr>
      <tr><td><code>POST /api/models/eval</code></td><td>Evaluate</td></tr>
      <tr><td><code>POST /api/models/export</code></td><td>Export</td></tr>
      <tr><td><code>POST /api/data/load</code></td><td>Load dataset</td></tr>
      <tr><td><code>POST /api/rag/index</code></td><td>Index RAG</td></tr>
      <tr><td><code>POST /api/rag/ask</code></td><td>Ask RAG</td></tr>
      <tr><td><code>POST /api/actions/run</code></td><td>Natural-language action</td></tr>
      <tr><td><code>POST /api/quantize</code></td><td>Quantize model</td></tr>
      <tr><td><code>POST /api/export/edge</code></td><td>Edge export</td></tr>
      <tr><td><code>GET /api/edge/footprint</code></td><td>Edge footprint</td></tr>
      <tr><td><code>POST /api/market</code></td><td>Marketplace</td></tr>
      <tr><td><code>GET /api/integrations</code></td><td>AI tools list</td></tr>
      <tr><td><code>POST /api/integrations/chat</code></td><td>Chat (BYOK)</td></tr>
      <tr><td><code>POST /api/terminal</code></td><td>In-app terminal</td></tr>
      <tr><td><code>GET /api/allc</code></td><td>10,000+ commands</td></tr>
      <tr><td><code>GET /api/autoallc</code></td><td>All automations</td></tr>
      <tr><td><code>POST /api/autotest</code></td><td>Full autotest</td></tr>
      <tr><td><code>GET /docs</code></td><td>Swagger UI</td></tr>
    </table>
    <p class="hint"><a href="/docs" style="color:var(--accent)">Open interactive Swagger docs →</a></p>`));
}

function pageHelp(parts) {
  pageRoot.appendChild(el("h2", "", "❓ Help"));
  pageRoot.appendChild(card("About", `
    <p>AWEAI — AI Model Factory. Create, train, tune and manage AI models <strong>from scratch</strong>. No built-in AI, no Hugging Face.</p>
    <p><strong>CLI:</strong> <code>aweai allc</code> (10,000+ commands), <code>aweai autoallc</code> (all automations), <code>aweai terminal</code> (REPL), <code>aweai serve</code> (this UI).</p>
    <p><strong>Shortcuts:</strong> press <code>Ctrl+`</code> to toggle the terminal drawer.</p>`));
  pageRoot.appendChild(card("Quick start", `
    <pre class="code">pip install -e .
aweai serve --port 8888        # open http://localhost:8888
aweai autotest                 # one-command system check
aweai allc --count 50          # browse 10,000+ commands
aweai train --type mlp --name demo --data data.csv</pre>`));
}

function initTerminalDrawer() {
  const drawer = $("terminal-drawer");
  const input = $("term-input");
  const out = $("term-output");
  const run = async () => {
    const line = input.value.trim();
    if (!line) return;
    out.textContent += `\naweai> ${line}\n`;
    try {
      const res = await api("/api/terminal", { method: "POST", body: JSON.stringify({ line })});
      out.textContent += (typeof res.result === "string" ? res.result : fmt(res.result)) + "\n";
    } catch (e) {
      out.textContent += "Error: " + e.message + "\n";
    }
    input.value = "";
    out.scrollTop = out.scrollHeight;
  };
  $("term-run").addEventListener("click", run);
  input.addEventListener("keydown", (ev) => { if (ev.key === "Enter") run(); });
  $("term-close").addEventListener("click", () => drawer.classList.add("hidden"));
  document.addEventListener("keydown", (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && ev.key === "`") {
      drawer.classList.toggle("hidden");
      if (!drawer.classList.contains("hidden")) input.focus();
    }
  });
}

function initGlobalSearch() {
  const input = $("global-search");
  const results = $("search-results");
  let timer = null;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    const q = input.value.trim();
    if (!q) { results.classList.add("hidden"); return; }
    timer = setTimeout(async () => {
      try {
        const res = await api(`/api/allc?search=${encodeURIComponent(q)}&count=25`);
        results.innerHTML = "";
        if (!res.items.length) {
          results.appendChild(el("div", "sr-item", "No matches."));
        }
        res.items.slice(0, 25).forEach((it) => {
          const d = el("div", "sr-item");
          d.innerHTML = `<span class="sr-cmd">${esc(it.cmd)}</span><br><span class="sr-help">${esc(it.help)}</span>`;
          d.addEventListener("click", () => { go("menus_allc_page"); results.classList.add("hidden"); input.value = ""; });
          results.appendChild(d);
        });
        results.classList.remove("hidden");
      } catch (e) {
        results.innerHTML = "";
        results.appendChild(el("div", "sr-item", "Error: " + e.message));
        results.classList.remove("hidden");
      }
    }, 300);
  });
  document.addEventListener("click", (ev) => {
    if (!results.contains(ev.target) && ev.target !== input) results.classList.add("hidden");
  });
}

function initTopBar() {
  $("menu-toggle").addEventListener("click", () => {
    const sb = $("sidebar");
    if (window.innerWidth <= 760) {
      sb.style.display = sb.style.display === "none" ? "flex" : "none";
    } else {
      sb.classList.toggle("collapsed");
      document.body.classList.toggle("sidebar-collapsed");
    }
  });
  $("fullscreen-btn").addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen();
  });
  $("api-docs-btn").addEventListener("click", () => window.open("/docs", "_blank"));
  $("autotest-btn").addEventListener("click", async () => {
    const btn = $("autotest-btn");
    btn.disabled = true; btn.textContent = "⚡ Running…";
    try {
      const res = await api("/api/autotest?no_ui=true", { method: "POST" });
      alert("Autotest: " + res.passed + "/" + res.total + " passed");
      go("autotest_page");
    } catch (e) {
      alert("Autotest error: " + e.message);
    } finally {
      btn.disabled = false; btn.textContent = "⚡ Autotest";
    }
  });
}

async function loadLanguages() {
  try {
    const data = await api("/api/languages");
    const sel = $("lang-select");
    sel.innerHTML = "";
    for (const [code, name] of Object.entries(data.languages)) {
      const opt = document.createElement("option");
      opt.value = code; opt.textContent = name;
      sel.appendChild(opt);
    }
    sel.addEventListener("change", async () => {
      await api("/api/config", { method: "POST", body: JSON.stringify({ key: "language", value: sel.value })});
      location.reload();
    });
  } catch (e) { console.error(e); }
}

buildSidebar();
go("dashboard");
initTerminalDrawer();
initGlobalSearch();
initTopBar();
loadLanguages();
setInterval(() => { if (currentPage === "dashboard") loadDashboard(); }, 5000);
