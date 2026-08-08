"""Lightweight zero-dependency i18n engine with 12 languages.

Languages:
    en  English (default)
    hy  Հայերեն (Armenian)
    ru  Русский
    fr  Français
    de  Deutsch
    es  Español
    it  Italiano
    pt  Português
    zh  中文
    ja  日本語
    ko  한국어
    tr  Türkçe

Translation dictionaries live in a single JSON asset file. If the file is
missing (e.g. running from a source checkout without package data), a
built-in fallback table keeps the engine fully functional.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

LANGUAGES = {
    "en": "English",
    "hy": "Հայերեն",
    "ru": "Русский",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
    "it": "Italiano",
    "pt": "Português",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "tr": "Türkçe",
}

# Built-in fallback strings (used when the JSON asset cannot be loaded).
_FALLBACK: Dict[str, Dict[str, str]] = {
    "en": {},
    "hy": {
        "app_title": "AWEAI — Համընդհանուր AI գործիք",
        "tagline": "AI ամեն ինչ մեկ տեղում",
        "welcome": "Բարի գալուստ AWEAI",
        "chat": "Զրույց",
        "models": "Մոդելներ",
        "train": "Մարզել",
        "rag": "RAG",
        "agents": "Ագենտներ",
        "actions": "Գործողություններ",
        "settings": "Կարգավորումներ",
        "language": "Լեզու",
        "send": "Ուղարկել",
        "enter_message": "Գրեք ձեր հաղորդագրությունը…",
        "loading": "Բեռնվում է…",
        "error": "Սխալ",
        "started": "Սկսվել է",
        "stopped": "Կանգնեցվել է",
        "status": "Կարգավիճակ",
        "hardware": "Սարքավորում",
        "model": "Մոդել",
        "new_model": "Նոր մոդել",
        "fine_tune": "Fine-tuning",
        "continue_training": "Շարունակել մարզումը",
        "data_path": "Տվյալների ուղի",
        "start": "Սկսել",
        "stop": "Կանգնեցնել",
        "console": "Վահանակ",
        "result": "Արդյունք",
        "docs": "Փաստաթղթեր",
        "home": "Գլխավոր",
    },
    "ru": {
        "app_title": "AWEAI — Универсальный AI-инструмент",
        "tagline": "Весь ИИ в одном месте",
        "welcome": "Добро пожаловать в AWEAI",
        "chat": "Чат",
        "models": "Модели",
        "train": "Обучение",
        "rag": "RAG",
        "agents": "Агенты",
        "actions": "Действия",
        "settings": "Настройки",
        "language": "Язык",
        "send": "Отправить",
        "enter_message": "Введите сообщение…",
        "loading": "Загрузка…",
        "error": "Ошибка",
        "started": "Запущено",
        "stopped": "Остановлено",
        "status": "Статус",
        "hardware": "Оборудование",
        "model": "Модель",
        "new_model": "Новая модель",
        "fine_tune": "Дообучение",
        "continue_training": "Продолжить обучение",
        "data_path": "Путь к данным",
        "start": "Старт",
        "stop": "Стоп",
        "console": "Консоль",
        "result": "Результат",
        "docs": "Документация",
        "home": "Главная",
    },
    "fr": {
        "app_title": "AWEAI — Boîte à outils IA universelle",
        "tagline": "Toute l'IA en un seul endroit",
        "welcome": "Bienvenue sur AWEAI",
        "chat": "Discussion",
        "models": "Modèles",
        "train": "Entraînement",
        "rag": "RAG",
        "agents": "Agents",
        "actions": "Actions",
        "settings": "Paramètres",
        "language": "Langue",
        "send": "Envoyer",
        "enter_message": "Écrivez votre message…",
        "loading": "Chargement…",
        "error": "Erreur",
        "started": "Démarré",
        "stopped": "Arrêté",
        "status": "Statut",
        "hardware": "Matériel",
        "model": "Modèle",
        "new_model": "Nouveau modèle",
        "fine_tune": "Ajustement",
        "continue_training": "Continuer l'entraînement",
        "data_path": "Chemin des données",
        "start": "Démarrer",
        "stop": "Arrêter",
        "console": "Console",
        "result": "Résultat",
        "docs": "Documentation",
        "home": "Accueil",
    },
    "de": {
        "app_title": "AWEAI — Universelles KI-Werkzeug",
        "tagline": "Alles KI an einem Ort",
        "welcome": "Willkommen bei AWEAI",
        "chat": "Chat",
        "models": "Modelle",
        "train": "Training",
        "rag": "RAG",
        "agents": "Agenten",
        "actions": "Aktionen",
        "settings": "Einstellungen",
        "language": "Sprache",
        "send": "Senden",
        "enter_message": "Nachricht eingeben…",
        "loading": "Laden…",
        "error": "Fehler",
        "started": "Gestartet",
        "stopped": "Gestoppt",
        "status": "Status",
        "hardware": "Hardware",
        "model": "Modell",
        "new_model": "Neues Modell",
        "fine_tune": "Feinabstimmung",
        "continue_training": "Training fortsetzen",
        "data_path": "Datenpfad",
        "start": "Starten",
        "stop": "Stoppen",
        "console": "Konsole",
        "result": "Ergebnis",
        "docs": "Dokumentation",
        "home": "Start",
    },
    "es": {
        "app_title": "AWEAI — Herramienta IA universal",
        "tagline": "Todo lo de IA en un solo lugar",
        "welcome": "Bienvenido a AWEAI",
        "chat": "Chat",
        "models": "Modelos",
        "train": "Entrenar",
        "rag": "RAG",
        "agents": "Agentes",
        "actions": "Acciones",
        "settings": "Ajustes",
        "language": "Idioma",
        "send": "Enviar",
        "enter_message": "Escribe tu mensaje…",
        "loading": "Cargando…",
        "error": "Error",
        "started": "Iniciado",
        "stopped": "Detenido",
        "status": "Estado",
        "hardware": "Hardware",
        "model": "Modelo",
        "new_model": "Nuevo modelo",
        "fine_tune": "Ajuste fino",
        "continue_training": "Continuar entrenamiento",
        "data_path": "Ruta de datos",
        "start": "Iniciar",
        "stop": "Detener",
        "console": "Consola",
        "result": "Resultado",
        "docs": "Documentación",
        "home": "Inicio",
    },
    "it": {
        "app_title": "AWEAI — Strumento IA universale",
        "tagline": "Tutta l'IA in un posto",
        "welcome": "Benvenuto in AWEAI",
        "chat": "Chat",
        "models": "Modelli",
        "train": "Addestramento",
        "rag": "RAG",
        "agents": "Agenti",
        "actions": "Azioni",
        "settings": "Impostazioni",
        "language": "Lingua",
        "send": "Invia",
        "enter_message": "Scrivi il tuo messaggio…",
        "loading": "Caricamento…",
        "error": "Errore",
        "started": "Avviato",
        "stopped": "Fermato",
        "status": "Stato",
        "hardware": "Hardware",
        "model": "Modello",
        "new_model": "Nuovo modello",
        "fine_tune": "Ottimizzazione",
        "continue_training": "Continua addestramento",
        "data_path": "Percorso dati",
        "start": "Avvia",
        "stop": "Ferma",
        "console": "Console",
        "result": "Risultato",
        "docs": "Documentazione",
        "home": "Home",
    },
    "pt": {
        "app_title": "AWEAI — Ferramenta de IA universal",
        "tagline": "Toda a IA em um só lugar",
        "welcome": "Bem-vindo ao AWEAI",
        "chat": "Chat",
        "models": "Modelos",
        "train": "Treinar",
        "rag": "RAG",
        "agents": "Agentes",
        "actions": "Ações",
        "settings": "Configurações",
        "language": "Idioma",
        "send": "Enviar",
        "enter_message": "Escreva sua mensagem…",
        "loading": "Carregando…",
        "error": "Erro",
        "started": "Iniciado",
        "stopped": "Parado",
        "status": "Status",
        "hardware": "Hardware",
        "model": "Modelo",
        "new_model": "Novo modelo",
        "fine_tune": "Ajuste fino",
        "continue_training": "Continuar treinamento",
        "data_path": "Caminho dos dados",
        "start": "Iniciar",
        "stop": "Parar",
        "console": "Console",
        "result": "Resultado",
        "docs": "Documentação",
        "home": "Início",
    },
    "zh": {
        "app_title": "AWEAI — 通用 AI 工具箱",
        "tagline": "所有 AI 功能集于一身",
        "welcome": "欢迎使用 AWEAI",
        "chat": "聊天",
        "models": "模型",
        "train": "训练",
        "rag": "RAG",
        "agents": "智能体",
        "actions": "操作",
        "settings": "设置",
        "language": "语言",
        "send": "发送",
        "enter_message": "输入消息…",
        "loading": "加载中…",
        "error": "错误",
        "started": "已启动",
        "stopped": "已停止",
        "status": "状态",
        "hardware": "硬件",
        "model": "模型",
        "new_model": "新模型",
        "fine_tune": "微调",
        "continue_training": "继续训练",
        "data_path": "数据路径",
        "start": "开始",
        "stop": "停止",
        "console": "控制台",
        "result": "结果",
        "docs": "文档",
        "home": "首页",
    },
    "ja": {
        "app_title": "AWEAI — 万能AIツールボックス",
        "tagline": "AIのすべてを一箇所に",
        "welcome": "AWEAIへようこそ",
        "chat": "チャット",
        "models": "モデル",
        "train": "トレーニング",
        "rag": "RAG",
        "agents": "エージェント",
        "actions": "アクション",
        "settings": "設定",
        "language": "言語",
        "send": "送信",
        "enter_message": "メッセージを入力…",
        "loading": "読み込み中…",
        "error": "エラー",
        "started": "開始",
        "stopped": "停止",
        "status": "ステータス",
        "hardware": "ハードウェア",
        "model": "モデル",
        "new_model": "新モデル",
        "fine_tune": "ファインチューニング",
        "continue_training": "トレーニング継続",
        "data_path": "データパス",
        "start": "開始",
        "stop": "停止",
        "console": "コンソール",
        "result": "結果",
        "docs": "ドキュメント",
        "home": "ホーム",
    },
    "ko": {
        "app_title": "AWEAI — 만능 AI 도구 상자",
        "tagline": "모든 AI를 한곳에",
        "welcome": "AWEAI에 오신 것을 환영합니다",
        "chat": "채팅",
        "models": "모델",
        "train": "훈련",
        "rag": "RAG",
        "agents": "에이전트",
        "actions": "작업",
        "settings": "설정",
        "language": "언어",
        "send": "보내기",
        "enter_message": "메시지 입력…",
        "loading": "로딩 중…",
        "error": "오류",
        "started": "시작됨",
        "stopped": "중지됨",
        "status": "상태",
        "hardware": "하드웨어",
        "model": "모델",
        "new_model": "새 모델",
        "fine_tune": "미세 조정",
        "continue_training": "훈련 계속",
        "data_path": "데이터 경로",
        "start": "시작",
        "stop": "중지",
        "console": "콘솔",
        "result": "결과",
        "docs": "문서",
        "home": "홈",
    },
    "tr": {
        "app_title": "AWEAI — Evrensel Yapay Zeka Aracı",
        "tagline": "Tüm yapay zeka tek yerde",
        "welcome": "AWEAI'ya hoş geldiniz",
        "chat": "Sohbet",
        "models": "Modeller",
        "train": "Eğitim",
        "rag": "RAG",
        "agents": "Ajanlar",
        "actions": "Eylemler",
        "settings": "Ayarlar",
        "language": "Dil",
        "send": "Gönder",
        "enter_message": "Mesajınızı yazın…",
        "loading": "Yükleniyor…",
        "error": "Hata",
        "started": "Başlatıldı",
        "stopped": "Durduruldu",
        "status": "Durum",
        "hardware": "Donanım",
        "model": "Model",
        "new_model": "Yeni model",
        "fine_tune": "İnce ayar",
        "continue_training": "Eğitimi sürdür",
        "data_path": "Veri yolu",
        "start": "Başlat",
        "stop": "Durdur",
        "console": "Konsol",
        "result": "Sonuç",
        "docs": "Dokümantasyon",
        "home": "Ana Sayfa",
    },
}

# English strings (source of truth for every key)
_EN: Dict[str, str] = {
    "app_title": "AWEAI — Universal AI Toolbox",
    "tagline": "Everything AI in one place",
    "welcome": "Welcome to AWEAI",
    "chat": "Chat",
    "models": "Models",
    "train": "Train",
    "rag": "RAG",
    "agents": "Agents",
    "actions": "Actions",
    "settings": "Settings",
    "language": "Language",
    "send": "Send",
    "enter_message": "Type your message…",
    "loading": "Loading…",
    "error": "Error",
    "started": "Started",
    "stopped": "Stopped",
    "status": "Status",
    "hardware": "Hardware",
    "model": "Model",
    "new_model": "New model",
    "fine_tune": "Fine-tune",
    "continue_training": "Continue training",
    "data_path": "Data path",
    "start": "Start",
    "stop": "Stop",
    "console": "Console",
    "result": "Result",
    "docs": "Documentation",
    "home": "Home",
}


def _asset_path() -> Path:
    return Path(__file__).parent / "i18n_assets.json"


def _load_assets() -> Dict[str, Dict[str, str]]:
    try:
        p = _asset_path()
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError):
        pass
    merged: Dict[str, Dict[str, str]] = {"en": _EN}
    for code, strings in _FALLBACK.items():
        if code == "en":
            merged["en"] = _EN
        else:
            merged[code] = {**_EN, **strings}
    return merged


_STRINGS: Dict[str, Dict[str, str]] = _load_assets()


class Translator:
    """Simple gettext-style translator."""

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang if lang in _STRINGS else "en"

    def t(self, key: str, **kwargs) -> str:
        table = _STRINGS.get(self.lang, _STRINGS.get("en", {}))
        text = table.get(key) or _EN.get(key) or key
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, IndexError):
                return text
        return text

    def __call__(self, key: str, **kwargs) -> str:
        return self.t(key, **kwargs)


def get_translator(lang: Optional[str] = None) -> Translator:
    """Return a translator. Pass a language code or None to auto-detect."""
    if lang is None:
        import os

        lang = os.environ.get("AWEAI_LANG", "en")
    return Translator(lang)


def available_languages() -> Dict[str, str]:
    return dict(LANGUAGES)


def supported_langs() -> list:
    return sorted(_STRINGS.keys())
