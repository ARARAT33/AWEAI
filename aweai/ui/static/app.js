/* AWEAI SPA frontend logic: 12-language UI, chat, models, train, RAG, agents, actions. */
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  let LANG = localStorage.getItem("aweai_lang") || "en";
  const I18N = {};

  // ---------- API helpers ----------
  async function api(path, options = {}) {
    const opts = { headers: { "Content-Type": "application/json" }, ...options };
    if (opts.body && typeof opts.body !== "string") opts.body = JSON.stringify(opts.body);
    const res = await fetch(path, opts);
    if (!res.ok) {
      let msg = res.statusText;
      try { const d = await res.json(); msg = d.detail || msg; } catch (e) {}
      throw new Error(msg);
    }
    return res.json();
  }

  // ---------- i18n ----------
  async function loadTranslations() {
    try {
      const data = await api("/api/languages");
      const langs = data.languages || { en: "English" };
      const sel = $("#lang-select"), sel2 = $("#settings-lang");
      for (const [code, name] of Object.entries(langs)) {
        const o = new Option(name, code);
        sel.add(o.cloneNode());
        sel2.add(o);
      }
      sel.value = LANG; sel2.value = LANG;
    } catch (e) { console.warn("i18n load failed", e); }
    try {
      const cfg = await api("/api/config");
      LANG = cfg.config.language || LANG;
    } catch (e) {}
    applyLang();
  }

  function applyLang() {
    localStorage.setItem("aweai_lang", LANG);
    document.documentElement.lang = LANG;
    $$("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      const t = I18N[LANG] && I18N[LANG][key];
      if (t) el.textContent = t;
    });
    $$("[data-i18n-ph]").forEach((el) => {
      const key = el.getAttribute("data-i18n-ph");
      const t = I18N[LANG] && I18N[LANG][key];
      if (t) el.placeholder = t;
    });
    $("#settings-lang").value = LANG;
  }

  async function fetchTranslations() {
    // Translations are served by the backend via /api/languages for codes,
    // but the strings themselves live in the package; we load a small
    // client-side dictionary here (mirrors server i18n).
    I18N.en = { app_title: "Universal AI Toolbox", tagline: "Everything AI in one place", welcome: "Welcome to AWEAI", chat: "Chat", models: "Models", train: "Train", rag: "RAG", agents: "Agents", actions: "Actions", settings: "Settings", language: "Language", send: "Send", enter_message: "Type your message…", loading: "Loading…", error: "Error", started: "Started", stopped: "Stopped", status: "Status", hardware: "Hardware", model: "Model", new_model: "New model", fine_tune: "Fine-tune", continue_training: "Continue training", data_path: "Data path", start: "Start", stop: "Stop", console: "Console", result: "Result", docs: "Documentation", home: "Home" };
    I18N.hy = { app_title: "Համընդհանուր AI գործիք", tagline: "AI ամեն ինչ մեկ տեղում", welcome: "Բարի գալուստ AWEAI", chat: "Զրույց", models: "Մոդելներ", train: "Մարզել", rag: "RAG", agents: "Ագենտներ", actions: "Գործողություններ", settings: "Կարգավորումներ", language: "Լեզու", send: "Ուղարկել", enter_message: "Գրեք ձեր հաղորդագրությունը…", loading: "Բեռնվում է…", error: "Սխալ", started: "Սկսվել է", stopped: "Կանգնեցվել է", status: "Կարգավիճակ", hardware: "Սարքավորում", model: "Մոդել", new_model: "Նոր մոդել", fine_tune: "Fine-tuning", continue_training: "Շարունակել մարզումը", data_path: "Տվյալների ուղի", start: "Սկսել", stop: "Կանգնեցնել", console: "Վահանակ", result: "Արդյունք", docs: "Փաստաթղթեր", home: "Գլխավոր" };
    I18N.ru = { app_title: "Универсальный AI-инструмент", tagline: "Весь ИИ в одном месте", welcome: "Добро пожаловать в AWEAI", chat: "Чат", models: "Модели", train: "Обучение", rag: "RAG", agents: "Агенты", actions: "Действия", settings: "Настройки", language: "Язык", send: "Отправить", enter_message: "Введите сообщение…", loading: "Загрузка…", error: "Ошибка", started: "Запущено", stopped: "Остановлено", status: "Статус", hardware: "Оборудование", model: "Модель", new_model: "Новая модель", fine_tune: "Дообучение", continue_training: "Продолжить обучение", data_path: "Путь к данным", start: "Старт", stop: "Стоп", console: "Консоль", result: "Результат", docs: "Документация", home: "Главная" };
    I18N.fr = { app_title: "Boîte à outils IA universelle", tagline: "Toute l'IA en un seul endroit", welcome: "Bienvenue sur AWEAI", chat: "Discussion", models: "Modèles", train: "Entraînement", rag: "RAG", agents: "Agents", actions: "Actions", settings: "Paramètres", language: "Langue", send: "Envoyer", enter_message: "Écrivez votre message…", loading: "Chargement…", error: "Erreur", started: "Démarré", stopped: "Arrêté", status: "Statut", hardware: "Matériel", model: "Modèle", new_model: "Nouveau modèle", fine_tune: "Ajustement", continue_training: "Continuer l'entraînement", data_path: "Chemin des données", start: "Démarrer", stop: "Arrêter", console: "Console", result: "Résultat", docs: "Documentation", home: "Accueil" };
    I18N.de = { app_title: "Universelles KI-Werkzeug", tagline: "Alles KI an einem Ort", welcome: "Willkommen bei AWEAI", chat: "Chat", models: "Modelle", train: "Training", rag: "RAG", agents: "Agenten", actions: "Aktionen", settings: "Einstellungen", language: "Sprache", send: "Senden", enter_message: "Nachricht eingeben…", loading: "Laden…", error: "Fehler", started: "Gestartet", stopped: "Gestoppt", status: "Status", hardware: "Hardware", model: "Modell", new_model: "Neues Modell", fine_tune: "Feinabstimmung", continue_training: "Training fortsetzen", data_path: "Datenpfad", start: "Starten", stop: "Stoppen", console: "Konsole", result: "Ergebnis", docs: "Dokumentation", home: "Start" };
    I18N.es = { app_title: "Herramienta IA universal", tagline: "Todo lo de IA en un solo lugar", welcome: "Bienvenido a AWEAI", chat: "Chat", models: "Modelos", train: "Entrenar", rag: "RAG", agents: "Agentes", actions: "Acciones", settings: "Ajustes", language: "Idioma", send: "Enviar", enter_message: "Escribe tu mensaje…", loading: "Cargando…", error: "Error", started: "Iniciado", stopped: "Detenido", status: "Estado", hardware: "Hardware", model: "Modelo", new_model: "Nuevo modelo", fine_tune: "Ajuste fino", continue_training: "Continuar entrenamiento", data_path: "Ruta de datos", start: "Iniciar", stop: "Detener", console: "Consola", result: "Resultado", docs: "Documentación", home: "Inicio" };
    I18N.it = { app_title: "Strumento IA universale", tagline: "Tutta l'IA in un posto", welcome: "Benvenuto in AWEAI", chat: "Chat", models: "Modelli", train: "Addestramento", rag: "RAG", agents: "Agenti", actions: "Azioni", settings: "Impostazioni", language: "Lingua", send: "Invia", enter_message: "Scrivi il tuo messaggio…", loading: "Caricamento…", error: "Errore", started: "Avviato", stopped: "Fermato", status: "Stato", hardware: "Hardware", model: "Modello", new_model: "Nuovo modello", fine_tune: "Ottimizzazione", continue_training: "Continua addestramento", data_path: "Percorso dati", start: "Avvia", stop: "Ferma", console: "Console", result: "Risultato", docs: "Documentazione", home: "Home" };
    I18N.pt = { app_title: "Ferramenta de IA universal", tagline: "Toda a IA em um só lugar", welcome: "Bem-vindo ao AWEAI", chat: "Chat", models: "Modelos", train: "Treinar", rag: "RAG", agents: "Agentes", actions: "Ações", settings: "Configurações", language: "Idioma", send: "Enviar", enter_message: "Escreva sua mensagem…", loading: "Carregando…", error: "Erro", started: "Iniciado", stopped: "Parado", status: "Status", hardware: "Hardware", model: "Modelo", new_model: "Novo modelo", fine_tune: "Ajuste fino", continue_training: "Continuar treinamento", data_path: "Caminho dos dados", start: "Iniciar", stop: "Parar", console: "Console", result: "Resultado", docs: "Documentação", home: "Início" };
    I18N.zh = { app_title: "通用 AI 工具箱", tagline: "所有 AI 功能集于一身", welcome: "欢迎使用 AWEAI", chat: "聊天", models: "模型", train: "训练", rag: "RAG", agents: "智能体", actions: "操作", settings: "设置", language: "语言", send: "发送", enter_message: "输入消息…", loading: "加载中…", error: "错误", started: "已启动", stopped: "已停止", status: "状态", hardware: "硬件", model: "模型", new_model: "新模型", fine_tune: "微调", continue_training: "继续训练", data_path: "数据路径", start: "开始", stop: "停止", console: "控制台", result: "结果", docs: "文档", home: "首页" };
    I18N.ja = { app_title: "万能AIツールボックス", tagline: "AIのすべてを一箇所に", welcome: "AWEAIへようこそ", chat: "チャット", models: "モデル", train: "トレーニング", rag: "RAG", agents: "エージェント", actions: "アクション", settings: "設定", language: "言語", send: "送信", enter_message: "メッセージを入力…", loading: "読み込み中…", error: "エラー", started: "開始", stopped: "停止", status: "ステータス", hardware: "ハードウェア", model: "モデル", new_model: "新モデル", fine_tune: "ファインチューニング", continue_training: "トレーニング継続", data_path: "データパス", start: "開始", stop: "停止", console: "コンソール", result: "結果", docs: "ドキュメント", home: "ホーム" };
    I18N.ko = { app_title: "만능 AI 도구 상자", tagline: "모든 AI를 한곳에", welcome: "AWEAI에 오신 것을 환영합니다", chat: "채팅", models: "모델", train: "훈련", rag: "RAG", agents: "에이전트", actions: "작업", settings: "설정", language: "언어", send: "보내기", enter_message: "메시지 입력…", loading: "로딩 중…", error: "오류", started: "시작됨", stopped: "중지됨", status: "상태", hardware: "하드웨어", model: "모델", new_model: "새 모델", fine_tune: "미세 조정", continue_training: "훈련 계속", data_path: "데이터 경로", start: "시작", stop: "중지", console: "콘솔", result: "결과", docs: "문서", home: "홈" };
    I18N.tr = { app_title: "Evrensel Yapay Zeka Aracı", tagline: "Tüm yapay zeka tek yerde", welcome: "AWEAI'ya hoş geldiniz", chat: "Sohbet", models: "Modeller", train: "Eğitim", rag: "RAG", agents: "Ajanlar", actions: "Eylemler", settings: "Ayarlar", language: "Dil", send: "Gönder", enter_message: "Mesajınızı yazın…", loading: "Yükleniyor…", error: "Hata", started: "Başlatıldı", stopped: "Durduruldu", status: "Durum", hardware: "Donanım", model: "Model", new_model: "Yeni model", fine_tune: "İnce ayar", continue_training: "Eğitimi sürdür", data_path: "Veri yolu", start: "Başlat", stop: "Durdur", console: "Konsol", result: "Sonuç", docs: "Dokümantasyon", home: "Ana Sayfa" };
  }

  // ---------- navigation ----------
  function switchView(name) {
    $$(".view").forEach((v) => v.classList.remove("active"));
    $("#view-" + name).classList.add("active");
    $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
    if (name === "models") loadModels();
    if (name === "rag") loadRagStats();
  }

  // ---------- chat ----------
  const history = [];
  async function sendChat(text) {
    const wrap = document.createElement("div");
    wrap.className = "msg user";
    const b = document.createElement("div");
    b.className = "bubble";
    b.textContent = text;
    wrap.appendChild(b);
    $("#chat-messages").appendChild(wrap);

    const botWrap = document.createElement("div");
    botWrap.className = "msg bot";
    const botB = document.createElement("div");
    botB.className = "bubble";
    botB.textContent = "…";
    botWrap.appendChild(botB);
    $("#chat-messages").appendChild(botWrap);
    $("#chat-messages").scrollTop = $("#chat-messages").scrollHeight;

    history.push({ role: "user", content: text });
    try {
      const data = await api("/api/chat", { method: "POST", body: { message: text, history: history.slice(0, -1) } });
      botB.textContent = data.reply;
      history.push({ role: "assistant", content: data.reply });
      $("#chat-status").textContent = data.model || "auto";
    } catch (e) {
      botB.textContent = "⚠ " + e.message;
    }
    $("#chat-messages").scrollTop = $("#chat-messages").scrollHeight;
  }

  // ---------- models ----------
  async function loadModels() {
    try {
      const rec = await api("/api/models/recommended");
      const best = rec.best;
      $("#rec-header").textContent = "Recommended: " + rec.hardware.recommended_tier + " tier";
      $("#rec-model").innerHTML = best
        ? `<strong>${best.id}</strong> — ${best.params_b}B params, ${best.context} ctx, ${best.license} license, ${best.hf}`
        : "—";
      const data = await api("/api/models");
      const rows = (data.catalog || []).map((m) =>
        `<tr><td><strong>${m.id}</strong></td><td>${m.family}</td><td>${m.params_b}B</td><td>${m.context}</td><td>${m.min_ram_gb}GB</td><td><span class="tag">${m.license}</span></td></tr>`
      ).join("");
      $("#model-table").innerHTML = `<table><thead><tr><th>Model</th><th>Family</th><th>Params</th><th>Context</th><th>RAM</th><th>License</th></tr></thead><tbody>${rows}</tbody></table>`;
    } catch (e) {
      $("#model-table").innerHTML = `<span class="err">${e.message}</span>`;
    }
  }

  // ---------- train ----------
  async function runTrain() {
    const out = $("#train-output");
    out.textContent = "…";
    try {
      const body = {
        name: $("#train-name").value || "my_model",
        data: $("#train-data").value || "",
        mode: $("#train-mode").value,
        base_model: $("#train-base").value || null,
        epochs: 1,
      };
      const data = await api("/api/train", { method: "POST", body });
      out.textContent = JSON.stringify(data, null, 2);
    } catch (e) {
      out.textContent = "⚠ " + e.message;
    }
  }

  // ---------- RAG ----------
  async function loadRagStats() {
    try {
      const s = await api("/api/rag/stats");
      $("#rag-stats").textContent = "chunks: " + s.chunks + " | docs: " + s.docs + " | backend: " + s.backend;
    } catch (e) { $("#rag-stats").textContent = "⚠ " + e.message; }
  }
  async function runRagIndex() {
    $("#rag-stats").textContent = "…";
    try {
      const data = await api("/api/rag/index", { method: "POST", body: { path: $("#rag-path").value || "" } });
      $("#rag-stats").textContent = "added: " + data.added + " | chunks: " + data.stats.chunks;
    } catch (e) { $("#rag-stats").textContent = "⚠ " + e.message; }
  }
  async function runRagAsk() {
    $("#rag-answer").textContent = "…";
    try {
      const data = await api("/api/rag/ask", { method: "POST", body: { query: $("#rag-query").value, top_k: 4 } });
      let out = data.answer + "\n\nSources:\n" + (data.sources || []).map((s) => "• " + s.id + ": " + s.text.slice(0, 90)).join("\n");
      $("#rag-answer").textContent = out;
    } catch (e) { $("#rag-answer").textContent = "⚠ " + e.message; }
  }

  // ---------- agents ----------
  async function runAgent() {
    $("#agent-output").textContent = "…";
    try {
      const data = await api("/api/agent/run", { method: "POST", body: { task: $("#agent-task").value || "Say hello", max_steps: 5 } });
      $("#agent-output").textContent = "Final: " + data.final + "\nTool calls: " + data.tool_calls;
    } catch (e) { $("#agent-output").textContent = "⚠ " + e.message; }
  }

  // ---------- actions ----------
  async function runAction() {
    $("#action-output").textContent = "…";
    try {
      const data = await api("/api/actions/run", { method: "POST", body: { text: $("#action-text").value, lang: LANG } });
      $("#action-output").textContent = JSON.stringify(data, null, 2);
    } catch (e) { $("#action-output").textContent = "⚠ " + e.message; }
  }

  // ---------- settings ----------
  async function saveSettings() {
    $("#settings-output").textContent = "…";
    try {
      const values = { language: $("#settings-lang").value, port: parseInt($("#settings-port").value, 10) || 8888, default_model: $("#settings-model").value || null };
      const data = await api("/api/config", { method: "POST", body: { values } });
      LANG = data.config.language;
      applyLang();
      $("#settings-output").textContent = "Saved ✓";
    } catch (e) { $("#settings-output").textContent = "⚠ " + e.message; }
  }

  // ---------- hardware badge ----------
  async function loadHardware() {
    try {
      const hw = await api("/api/hardware");
      const gpu = hw.gpu_names[0] || "CPU";
      $("#hw-badge").textContent = `${gpu} · ${hw.ram_total_gb}GB RAM · ${hw.cpu_count} cores`;
    } catch (e) {}
  }

  // ---------- events ----------
  function bindEvents() {
    $$(".nav-item").forEach((b) => b.addEventListener("click", () => switchView(b.dataset.view)));
    $("#chat-form").addEventListener("submit", (e) => { e.preventDefault(); const v = $("#chat-input").value.trim(); if (v) { $("#chat-input").value = ""; sendChat(v); } });
    $("#refresh-models").addEventListener("click", loadModels);
    $("#train-btn").addEventListener("click", runTrain);
    $("#rag-index-btn").addEventListener("click", runRagIndex);
    $("#rag-ask-btn").addEventListener("click", runRagAsk);
    $("#agent-run-btn").addEventListener("click", runAgent);
    $("#action-run-btn").addEventListener("click", runAction);
    $("#settings-save").addEventListener("click", saveSettings);
    $("#lang-select").addEventListener("change", (e) => { LANG = e.target.value; applyLang(); });
    $("#settings-lang").addEventListener("change", (e) => { LANG = e.target.value; applyLang(); });
  }

  // ---------- init ----------
  async function init() {
    await fetchTranslations();
    await loadTranslations();
    applyLang();
    bindEvents();
    loadHardware();
  }

  init();
})();
