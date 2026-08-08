/* AWEAI Model Factory UI — SPA logic */
"use strict";

const $ = (id) => document.getElementById(id);

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || res.statusText);
  }
  return res.json();
}

function fmt(obj) {
  return JSON.stringify(obj, null, 2);
}

/* ---- Tabs ---- */
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((s) => s.classList.remove("active"));
    btn.classList.add("active");
    $("tab-" + btn.dataset.tab).classList.add("active");
  });
});

/* ---- Languages ---- */
async function loadLanguages() {
  try {
    const data = await api("/api/languages");
    const sel = $("lang-select");
    sel.innerHTML = "";
    for (const [code, name] of Object.entries(data.languages)) {
      const opt = document.createElement("option");
      opt.value = code;
      opt.textContent = name;
      sel.appendChild(opt);
    }
    sel.addEventListener("change", async () => {
      await api("/api/config", {
        method: "POST",
        body: JSON.stringify({ key: "language", value: sel.value }),
      });
      location.reload();
    });
  } catch (e) {
    console.error(e);
  }
}

/* ---- Dashboard ---- */
async function loadDashboard() {
  try {
    const hw = await api("/api/hardware");
    $("hardware-info").textContent = fmt(hw.hardware);
    $("recommendation-info").textContent = fmt(hw.recommendation);
  } catch (e) {
    $("hardware-info").textContent = "Error: " + e.message;
  }
  try {
    const zoo = await api("/api/models");
    $("zoo-count").textContent = "Models: " + zoo.models.length + "\n" + fmt(zoo.models.map((m) => ({ name: m.name, type: m.model_type, v: m.version })));
    drawLoss(zoo.models);
  } catch (e) {
    $("zoo-count").textContent = "Error: " + e.message;
  }
}

function drawLoss(models) {
  const canvas = $("loss-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#161b22";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#8b949e";
  ctx.font = "12px monospace";
  const curves = models.map((m) => (m.metrics && m.metrics.history && m.metrics.history.loss) || null).filter(Boolean).slice(-3);
  if (!curves.length) {
    ctx.fillText("No training curves yet — train a model in the Wizard.", 20, 120);
    return;
  }
  const colors = ["#58a6ff", "#2ea043", "#d29922"];
  curves.forEach((loss, i) => {
    ctx.strokeStyle = colors[i % colors.length];
    ctx.beginPath();
    const max = Math.max(...loss);
    loss.forEach((v, j) => {
      const x = (j / Math.max(loss.length - 1, 1)) * (canvas.width - 40) + 20;
      const y = canvas.height - 20 - (v / (max || 1)) * (canvas.height - 60);
      j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  $("curve-hint").textContent = "Latest " + curves.length + " training curve(s) from the model zoo.";
}

/* ---- Wizard ---- */
async function loadModelTypes() {
  try {
    const data = await api("/api/model-types");
    const sel = $("w-model-type");
    sel.innerHTML = "";
    for (const t of data.types) {
      const opt = document.createElement("option");
      opt.value = t.name;
      opt.textContent = t.name + " — " + t.task;
      sel.appendChild(opt);
    }
  } catch (e) {
    console.error(e);
  }
}

$("train-btn").addEventListener("click", async () => {
  const btn = $("train-btn");
  btn.disabled = true;
  btn.textContent = "Training…";
  try {
    let params = {};
    try {
      params = JSON.parse($("w-params").value || "{}");
    } catch (e) {
      params = { epochs: parseInt($("w-epochs").value || "20", 10) };
    }
    if (!params.epochs) params.epochs = parseInt($("w-epochs").value || "20", 10);
    const res = await api("/api/models/train", {
      method: "POST",
      body: JSON.stringify({
        model_type: $("w-model-type").value,
        name: $("w-name").value || "my_model",
        data_path: $("w-data").value || null,
        target: $("w-target").value || null,
        params,
      }),
    });
    $("train-result").textContent = fmt(res.result);
    loadDashboard();
  } catch (e) {
    $("train-result").textContent = "Error: " + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "🚀 Train";
  }
});

/* ---- Model Zoo ---- */
$("refresh-zoo").addEventListener("click", loadZoo);
async function loadZoo() {
  try {
    const data = await api("/api/models");
    $("zoo-list").textContent = fmt(data.models);
  } catch (e) {
    $("zoo-list").textContent = "Error: " + e.message;
  }
}

/* ---- Datasets ---- */
$("ds-load").addEventListener("click", async () => {
  try {
    const res = await api("/api/data/load", {
      method: "POST",
      body: JSON.stringify({ path: $("ds-path").value }),
    });
    $("ds-result").textContent = fmt(res.result);
  } catch (e) {
    $("ds-result").textContent = "Error: " + e.message;
  }
});

$("ds-augment").addEventListener("click", async () => {
  try {
    const texts = $("ds-texts").value.split("\n").map((s) => s.trim()).filter(Boolean);
    const res = await api("/api/data/augment", {
      method: "POST",
      body: JSON.stringify({ texts, n: 1 }),
    });
    $("ds-augment-result").textContent = fmt(res.result);
  } catch (e) {
    $("ds-augment-result").textContent = "Error: " + e.message;
  }
});

/* ---- RAG ---- */
$("rag-index").addEventListener("click", async () => {
  try {
    const res = await api("/api/rag/index", {
      method: "POST",
      body: JSON.stringify({ path: $("rag-path").value || null }),
    });
    $("rag-index-result").textContent = fmt(res.result);
  } catch (e) {
    $("rag-index-result").textContent = "Error: " + e.message;
  }
});

$("rag-ask").addEventListener("click", async () => {
  try {
    const res = await api("/api/rag/ask", {
      method: "POST",
      body: JSON.stringify({ query: $("rag-query").value }),
    });
    $("rag-ask-result").textContent = fmt(res.result);
  } catch (e) {
    $("rag-ask-result").textContent = "Error: " + e.message;
  }
});

/* ---- Actions ---- */
$("act-run").addEventListener("click", async () => {
  try {
    const res = await api("/api/actions/run", {
      method: "POST",
      body: JSON.stringify({ text: $("act-text").value }),
    });
    $("act-result").textContent = fmt(res.result);
  } catch (e) {
    $("act-result").textContent = "Error: " + e.message;
  }
});

/* ---- Autotest button ---- */
$("autotest-btn").addEventListener("click", async () => {
  const btn = $("autotest-btn");
  btn.disabled = true;
  btn.textContent = "🧪 Running…";
  try {
    const res = await api("/api/autotest?no_ui=true", { method: "POST" });
    alert("Autotest: " + res.passed + "/" + res.total + " passed");
    console.log(res);
  } catch (e) {
    alert("Autotest error: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "🧪 Autotest";
  }
});

/* ---- Settings ---- */
async function loadConfig() {
  try {
    const data = await api("/api/config");
    $("config-info").textContent = fmt(data.config);
  } catch (e) {
    $("config-info").textContent = "Error: " + e.message;
  }
}

/* ---- Init ---- */
loadLanguages();
loadDashboard();
loadModelTypes();
loadZoo();
loadConfig();
setInterval(loadDashboard, 5000);
