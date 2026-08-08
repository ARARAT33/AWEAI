"""AWEAI internationalization (i18n) — 10+ languages, English primary,
Armenian included.

Keys are stored in a dict-of-dicts. `t()` looks up a key for the current
language with English fallback. Supported codes:
    en, hy, ru, fr, de, es, it, pt, tr, fa, zh, ja
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from aweai.config import get_config

# Minimal static fallback table for the most common keys; the full table is
# shipped in i18n_assets.json (loaded on demand).
_STATIC: Dict[str, Dict[str, str]] = {
    "app.name": {
        "en": "AWEAI — AI Model Factory",
        "hy": "AWEAI — AI մոդելների ֆաբրիկա",
        "ru": "AWEAI — фабрика ИИ-моделей",
        "fr": "AWEAI — usine de modèles IA",
        "de": "AWEAI — KI-Modellfabrik",
        "es": "AWEAI — fábrica de modelos de IA",
        "it": "AWEAI — fabbrica di modelli IA",
        "pt": "AWEAI — fábrica de modelos de IA",
        "tr": "AWEAI — yapay zeka model fabrikası",
        "fa": "AWEAI — کارخانه مدل‌های هوش مصنوعی",
        "zh": "AWEAI — AI 模型工厂",
        "ja": "AWEAI — AIモデル工場",
    },
    "app.tagline": {
        "en": "Create, train, tune and manage AI models from scratch. No built-in AI, no Hugging Face.",
        "hy": "Ստեղծիր, մարզիր, կարգավորիր և կառավարիր AI մոդելներ զրոյից։ Առանց ներկառուցված AI և Hugging Face:",
        "ru": "Создавайте, обучайте, настраивайте и управляйте ИИ-моделями с нуля. Без встроенного ИИ и Hugging Face.",
        "fr": "Créez, entraînez, ajustez et gérez des modèles IA à partir de zéro. Sans IA intégrée, sans Hugging Face.",
        "de": "Erstelle, trainiere, tune und verwalte KI-Modelle von Grund auf. Kein eingebautes KI, kein Hugging Face.",
        "es": "Crea, entrena, ajusta y gestiona modelos de IA desde cero. Sin IA integrada, sin Hugging Face.",
        "it": "Crea, addestra, ottimizza e gestisci modelli IA da zero. Nessuna IA integrata, nessun Hugging Face.",
        "pt": "Crie, treine, ajuste e gerencie modelos de IA do zero. Sem IA integrada, sem Hugging Face.",
        "tr": "Sıfırdan yapay zeka modelleri oluşturun, eğitin, ayarlayın ve yönetin. Yerleşik yapay zeka yok, Hugging Face yok.",
        "fa": "مدل‌های هوش مصنوعی را از صفر بسازید، آموزش دهید، تنظیم کنید و مدیریت کنید. بدون هوش مصنوعی داخلی و بدون Hugging Face.",
        "zh": "从零创建、训练、调优和管理 AI 模型。无内置 AI，无 Hugging Face。",
        "ja": "ゼロからAIモデルを作成・学習・チューニング・管理。内蔵AIなし、Hugging Faceなし。",
    },
    "common.dashboard": {"en": "Dashboard", "hy": "Վահանակ", "ru": "Панель", "fr": "Tableau de bord", "de": "Dashboard", "es": "Panel", "it": "Dashboard", "pt": "Painel", "tr": "Panel", "fa": "داشبورد", "zh": "仪表盘", "ja": "ダッシュボード"},
    "common.wizard": {"en": "Wizard", "hy": "Հրաշագործ", "ru": "Мастер", "fr": "Assistant", "de": "Assistent", "es": "Asistente", "it": "Procedura guidata", "pt": "Assistente", "tr": "Sihirbaz", "fa": "دستیار", "zh": "向导", "ja": "ウィザード"},
    "common.model_zoo": {"en": "Model Zoo", "hy": "Մոդելների կենդանաբանական այգի", "ru": "Зоопарк моделей", "fr": "Zoo de modèles", "de": "Modell-Zoo", "es": "Zoológico de modelos", "it": "Zoo dei modelli", "pt": "Zoológico de modelos", "tr": "Model hayvanat bahçesi", "fa": "باغ وحش مدل‌ها", "zh": "模型动物园", "ja": "モデル動物園"},
    "common.datasets": {"en": "Datasets", "hy": "Տվյալների բազաներ", "ru": "Наборы данных", "fr": "Jeux de données", "de": "Datensätze", "es": "Conjuntos de datos", "it": "Dataset", "pt": "Conjuntos de dados", "tr": "Veri setleri", "fa": "مجموعه داده‌ها", "zh": "数据集", "ja": "データセット"},
    "common.autotest": {"en": "Autotest", "hy": "Ավտոթեստ", "ru": "Автотест", "fr": "Autotest", "de": "Autotest", "es": "Autotest", "it": "Autotest", "pt": "Autoteste", "tr": "Otomatik test", "fa": "آزمون خودکار", "zh": "自动测试", "ja": "自動テスト"},
    "common.settings": {"en": "Settings", "hy": "Կարգավորումներ", "ru": "Настройки", "fr": "Paramètres", "de": "Einstellungen", "es": "Ajustes", "it": "Impostazioni", "pt": "Configurações", "tr": "Ayarlar", "fa": "تنظیمات", "zh": "设置", "ja": "設定"},
    "common.train": {"en": "Train", "hy": "Մարզել", "ru": "Обучить", "fr": "Entraîner", "de": "Trainieren", "es": "Entrenar", "it": "Addestra", "pt": "Treinar", "tr": "Eğit", "fa": "آموزش", "zh": "训练", "ja": "学習"},
    "common.evaluate": {"en": "Evaluate", "hy": "Գնահատել", "ru": "Оценить", "fr": "Évaluer", "de": "Bewerten", "es": "Evaluar", "it": "Valuta", "pt": "Avaliar", "tr": "Değerlendir", "fa": "ارزیابی", "zh": "评估", "ja": "評価"},
    "common.export": {"en": "Export", "hy": "Արտահանել", "ru": "Экспорт", "fr": "Exporter", "de": "Exportieren", "es": "Exportar", "it": "Esporta", "pt": "Exportar", "tr": "Dışa aktar", "fa": "خروجی گرفتن", "zh": "导出", "ja": "エクスポート"},
    "common.delete": {"en": "Delete", "hy": "Ջնջել", "ru": "Удалить", "fr": "Supprimer", "de": "Löschen", "es": "Eliminar", "it": "Elimina", "pt": "Excluir", "tr": "Sil", "fa": "حذف", "zh": "删除", "ja": "削除"},
    "common.save": {"en": "Save", "hy": "Պահպանել", "ru": "Сохранить", "fr": "Enregistrer", "de": "Speichern", "es": "Guardar", "it": "Salva", "pt": "Salvar", "tr": "Kaydet", "fa": "ذخیره", "zh": "保存", "ja": "保存"},
    "common.running": {"en": "Running", "hy": "Ընթացքում", "ru": "Выполняется", "fr": "En cours", "de": "Läuft", "es": "Ejecutando", "it": "In esecuzione", "pt": "Executando", "tr": "Çalışıyor", "fa": "در حال اجرا", "zh": "运行中", "ja": "実行中"},
    "common.passed": {"en": "Passed", "hy": "Անցած", "ru": "Пройдено", "fr": "Réussi", "de": "Bestanden", "es": "Aprobado", "it": "Superato", "pt": "Aprovado", "tr": "Geçti", "fa": "موفق", "zh": "通过", "ja": "合格"},
    "common.failed": {"en": "Failed", "hy": "Ձախողված", "ru": "Провалено", "fr": "Échoué", "de": "Fehlgeschlagen", "es": "Fallido", "it": "Fallito", "pt": "Falhou", "tr": "Başarısız", "fa": "ناموفق", "zh": "失败", "ja": "失敗"},
    "common.language": {"en": "Language", "hy": "Լեզու", "ru": "Язык", "fr": "Langue", "de": "Sprache", "es": "Idioma", "it": "Lingua", "pt": "Idioma", "tr": "Dil", "fa": "زبان", "zh": "语言", "ja": "言語"},
    "common.recommendation": {"en": "Recommendation", "hy": "Առաջարկություն", "ru": "Рекомендация", "fr": "Recommandation", "de": "Empfehlung", "es": "Recomendación", "it": "Raccomandazione", "pt": "Recomendaçao", "tr": "Öneri", "fa": "پیشنهاد", "zh": "推荐", "ja": "推奨"},
    "common.accuracy": {"en": "Accuracy", "hy": "Ճշգրտություն", "ru": "Точность", "fr": "Précision", "de": "Genauigkeit", "es": "Precisión", "it": "Precisione", "pt": "Precisão", "tr": "Doğruluk", "fa": "دقت", "zh": "准确率", "ja": "精度"},
    "common.loss": {"en": "Loss", "hy": "Կորուստ", "ru": "Потери", "fr": "Perte", "de": "Verlust", "es": "Pérdida", "it": "Perdita", "pt": "Perda", "tr": "Kayıp", "fa": "زیان", "zh": "损失", "ja": "損失"},
    "common.epochs": {"en": "Epochs", "hy": "Դարաշրջաններ", "ru": "Эпохи", "fr": "Époques", "de": "Epochen", "es": "Épocas", "it": "Epoche", "pt": "Épocas", "tr": "Devirler", "fa": "دوره‌ها", "zh": "轮次", "ja": "エポック"},
    "common.model_type": {"en": "Model type", "hy": "Մոդելի տեսակ", "ru": "Тип модели", "fr": "Type de modèle", "de": "Modelltyp", "es": "Tipo de modelo", "it": "Tipo di modello", "pt": "Tipo de modelo", "tr": "Model tipi", "fa": "نوع مدل", "zh": "模型类型", "ja": "モデルタイプ"},
    "common.name": {"en": "Name", "hy": "Անուն", "ru": "Имя", "fr": "Nom", "de": "Name", "es": "Nombre", "it": "Nome", "pt": "Nome", "tr": "Ad", "fa": "نام", "zh": "名称", "ja": "名前"},
    "common.actions": {"en": "Actions", "hy": "Գործողություններ", "ru": "Действия", "fr": "Actions", "de": "Aktionen", "es": "Acciones", "it": "Azioni", "pt": "Ações", "tr": "Eylemler", "fa": "عملیات", "zh": "操作", "ja": "操作"},
    "common.rag": {"en": "RAG", "hy": "RAG", "ru": "RAG", "fr": "RAG", "de": "RAG", "es": "RAG", "it": "RAG", "pt": "RAG", "tr": "RAG", "fa": "RAG", "zh": "RAG", "ja": "RAG"},
    "common.hardware": {"en": "Hardware", "hy": "Սարքավորում", "ru": "Оборудование", "fr": "Matériel", "de": "Hardware", "es": "Hardware", "it": "Hardware", "pt": "Hardware", "tr": "Donanım", "fa": "سخت‌افزار", "zh": "硬件", "ja": "ハードウェア"},
    "common.model_zoo_desc": {
        "en": "All trained models in the factory",
        "hy": "Բոլոր մարզված մոդելները ֆաբրիկայում",
        "ru": "Все обученные модели в фабрике",
        "fr": "Tous les modèles entraînés de l'usine",
        "de": "Alle trainierten Modelle der Fabrik",
        "es": "Todos los modelos entrenados en la fábrica",
        "it": "Tutti i modelli addestrati in fabbrica",
        "pt": "Todos os modelos treinados na fábrica",
        "tr": "Fabrikadaki tüm eğitilmiş modeller",
        "fa": "همه مدل‌های آموزش‌دیده در کارخانه",
        "zh": "工厂中所有已训练的模型",
        "ja": "工場のすべての学習済みモデル",
    },
    "dashboard.autotest_running": {
        "en": "Autotest is running…",
        "hy": "Ավտոթեստն ընթացքի մեջ է…",
        "ru": "Автотест выполняется…",
        "fr": "Autotest en cours…",
        "de": "Autotest läuft…",
        "es": "Autotest en ejecución…",
        "it": "Autotest in esecuzione…",
        "pt": "Autoteste em execução…",
        "tr": "Otomatik test çalışıyor…",
        "fa": "آزمون خودکار در حال اجرا…",
        "zh": "自动测试运行中…",
        "ja": "自動テスト実行中…",
    },
    "dashboard.autotest_done": {
        "en": "Autotest finished: {passed}/{total} passed",
        "hy": "Ավտոթեստն ավարտվեց՝ {passed}/{total} անցավ",
        "ru": "Автотест завершён: {passed}/{total} пройдено",
        "fr": "Autotest terminé : {passed}/{total} réussis",
        "de": "Autotest beendet: {passed}/{total} bestanden",
        "es": "Autotest finalizado: {passed}/{total} aprobados",
        "it": "Autotest terminato: {passed}/{total} superati",
        "pt": "Autoteste concluído: {passed}/{total} aprovados",
        "tr": "Otomatik test bitti: {passed}/{total} geçti",
        "fa": "آزمون خودکار تمام شد: {passed}/{total} موفق",
        "zh": "自动测试完成：{passed}/{total} 通过",
        "ja": "自動テスト完了：{passed}/{total} 合格",
    },
    "wizard.title": {
        "en": "Create & train a model",
        "hy": "Ստեղծել և մարզել մոդել",
        "ru": "Создать и обучить модель",
        "fr": "Créer et entraîner un modèle",
        "de": "Modell erstellen & trainieren",
        "es": "Crear y entrenar un modelo",
        "it": "Crea e addestra un modello",
        "pt": "Criar e treinar um modelo",
        "tr": "Model oluştur ve eğit",
        "fa": "ایجاد و آموزش مدل",
        "zh": "创建并训练模型",
        "ja": "モデルを作成して学習",
    },
}

LANGUAGES: Dict[str, str] = {
    "en": "English",
    "hy": "Հայերեն",
    "ru": "Русский",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
    "it": "Italiano",
    "pt": "Português",
    "tr": "Türkçe",
    "fa": "فارسی",
    "zh": "中文",
    "ja": "日本語",
}

_assets: Optional[Dict[str, Dict[str, str]]] = None


def _load_assets() -> Dict[str, Dict[str, str]]:
    global _assets
    if _assets is not None:
        return _assets
    from pathlib import Path

    p = Path(__file__).parent / "i18n_assets.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _assets = data
                return _assets
        except Exception:
            pass
    _assets = _STATIC
    return _assets


def languages() -> List[str]:
    return list(LANGUAGES.keys())


def language_names() -> Dict[str, str]:
    return dict(LANGUAGES)


def get_language() -> str:
    return get_config().get("language", "en")


def set_language(code: str) -> None:
    if code not in LANGUAGES:
        raise ValueError(f"Unsupported language: {code}. Available: {list(LANGUAGES.keys())}")
    get_config().set("language", code)


def t(key: str, lang: Optional[str] = None, **kwargs) -> str:
    """Translate a key; falls back to English, then to the key itself."""
    lang = lang or get_language()
    assets = _load_assets()
    entry = assets.get(key, _STATIC.get(key, {}))
    if isinstance(entry, dict):
        value = entry.get(lang) or entry.get("en") or key
    else:
        value = entry or key
    if kwargs:
        try:
            value = value.format(**kwargs)
        except Exception:
            pass
    return value


# ---------------------------------------------------------------------------
# Compatibility API (used by tests and external callers)
# ---------------------------------------------------------------------------
available_languages = languages
supported_langs = languages


class Translator:
    """Simple translation helper bound to a language.

    Usage:
        t = Translator("hy")
        t("common.dashboard")   # -> "Վահանակ"
    """

    def __init__(self, lang: Optional[str] = None) -> None:
        self.lang = lang or get_language()

    def __call__(self, key: str, **kwargs) -> str:
        return t(key, lang=self.lang, **kwargs)

    def translate(self, key: str, **kwargs) -> str:
        return self(key, **kwargs)
