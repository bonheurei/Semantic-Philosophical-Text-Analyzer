import json
import random
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from google import genai
import streamlit as st
from langdetect import LangDetectException, detect
from deep_translator import GoogleTranslator, MyMemoryTranslator
from deep_translator.constants import MY_MEMORY_LANGUAGES_TO_CODES
from deep_translator.exceptions import LanguageNotSupportedException, TooManyRequests
from transformers import pipeline

import feedback_memory
import literary_memory

MYMEMORY_INGILIZCE = "en-US"

ISO_MYMEMORY_TERCIH = {
    "ar": "ar-SA",
    "bg": "bg-BG",
    "bn": "bn-BD",
    "ca": "ca-ES",
    "cs": "cs-CZ",
    "da": "da-DK",
    "de": "de-DE",
    "el": "el-GR",
    "en": MYMEMORY_INGILIZCE,
    "es": "es-ES",
    "fa": "fa-IR",
    "fi": "fi-FI",
    "fr": "fr-FR",
    "he": "he-IL",
    "hi": "hi-IN",
    "hr": "hr-HR",
    "hu": "hu-HU",
    "id": "id-ID",
    "it": "it-IT",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "ms": "ms-MY",
    "nl": "nl-NL",
    "no": "no-NO",
    "pl": "pl-PL",
    "pt": "pt-PT",
    "ro": "ro-RO",
    "ru": "ru-RU",
    "sk": "sk-SK",
    "sv": "sv-SE",
    "th": "th-TH",
    "tl": "tl-PH",
    "tr": "tr-TR",
    "uk": "uk-UA",
    "ur": "ur-PK",
    "vi": "vi-VN",
    "zh": "zh-CN",
}

ROBERTA_MODEL_ID = "SamLowe/roberta-base-go_emotions"
ROBERTA_CHUNK_WORDS = 280
ROBERTA_CHUNK_OVERLAP = 80
ROBERTA_MAX_CHUNKS = 6
PHATIC_MAX_WORDS = 12
SALIENT_EMOTION_MIN = 0.18
NEUTRAL_CHALLENGER_MIN = 0.18
PHATIC_CHALLENGER_MIN = 0.35
LITERARY_MIN_WORDS = 18
MIN_CHARS_DEEP = 100
MIN_WORDS_DEEP = 14
PHATIC_KALIP = re.compile(
    r"^\s*(hi+|hello|hey|merhaba|selam|how are you|nasılsın|nasilsin|"
    r"what's up|whats up|iyi misin|gunaydin|günaydın|good morning|"
    r"good (evening|night)|iyi akşamlar|iyi geceler)[\s\?\!\.\,]*$",
    re.IGNORECASE,
)
GEMINI_INSIGHT_PROMPT = (
    "Provide a brief, 2-sentence philosophical and semantic analysis of this text, "
    "focusing on its underlying tone, subtext, or metaphorical meaning. "
    "If the passage is ordinary, conversational, or lacks subtext, say so plainly—"
    "do not invent deep meaning."
)
EDISI_DUYGU_INKAR = re.compile(
    r"(hiçbir şey hisset|hissetmiyor|hissetmedim|duygum yok|duygu yok|"
    r"umursam|kayıtsız|kayitsiz|duygusuz|ilgisiz|"
    r"felt nothing|don't feel|didn't feel|do not feel|no feeling|numb|detached|"
    r"indifferent|indifference|doesn't (matter|care)|does not (matter|care)|"
    r"indifférent|indifference|je m'en fous|cela m'est égal)",
    re.IGNORECASE,
)
EDISI_BEKLENEN_DUYGU = re.compile(
    r"(sanki .{0,48}(hisset|üzül|gibi)|should have felt|ought to feel|"
    r"as if i (should|ought)|üzülmeliymiş|expected to (cry|grieve|feel)|"
    r"supposed to (feel|grieve)|aurait dû (pleurer|ressentir))",
    re.IGNORECASE,
)
EDISI_OLAY_YASI = re.compile(
    r"\b(ölüm|ölmüş|öldü|cenaze|anne.*öl|death|died|dead|funeral|mourning|"
    r"mort|mère.*mort|muerte|tod)\b",
    re.IGNORECASE,
)
GRIEFISH_LABELS = frozenset({"grief", "sadness", "remorse"})
TON_LABELS = frozenset(
    {
        "realization",
        "confusion",
        "disappointment",
        "neutral",
        "nervousness",
        "annoyance",
        "approval",
    }
)
GEMINI_EMOTION_CALIBRATION = (
    "Emotion rules (event ≠ felt tone):\n"
    "- Score the speaker/narrator's lived affect and stance, never plot keywords alone.\n"
    "- Death, illness, loss, or crime in the plot do NOT automatically mean grief or sadness.\n"
    "- If the voice is numb, flat, estranged, indifferent, or notes not feeling expected "
    "sorrow, prefer realization, confusion, disappointment, nervousness, or neutral—"
    "not grief.\n"
    "- Irony and understatement: score the ironic distance, not the literal topic.\n"
    "- Prefer affective labels (joy, sadness, anger, fear, love, admiration, curiosity, "
    "etc.) when a clear mood is present. Use neutral only when the text is truly flat "
    "or informational.\n"
    "Calibration sketches (do not copy; mirror the method):\n"
    "1) Mother died; narrator feels nothing / finds it absurd → realization/confusion/"
    "neutral over grief.\n"
    "2) Breakup described with cold procedural detail → disappointment/neutral over "
    "sadness.\n"
    "3) Warm reunion with clear affection → love/joy (here event and tone align).\n"
)
EDISI_KARSIT = re.compile(
    r"\b(ama|fakat|ancak|lakin|oysa|but|yet|however|although|though)\b",
    re.IGNORECASE,
)
EDISI_BENZETME = re.compile(
    r"(sanki|gibi bir|metafor|metaphor|\bas if\b|\blike a\b)",
    re.IGNORECASE,
)
ANALYSIS_LANGUAGES: dict[str, tuple[str, str]] = {
    "Arabic": ("ar", "Arabic"),
    "Bengali": ("bn", "Bengali"),
    "Bulgarian": ("bg", "Bulgarian"),
    "Catalan": ("ca", "Catalan"),
    "Chinese (Simplified)": ("zh", "Chinese (Simplified)"),
    "Chinese (Traditional)": ("zh-TW", "Chinese (Traditional)"),
    "Croatian": ("hr", "Croatian"),
    "Czech": ("cs", "Czech"),
    "Danish": ("da", "Danish"),
    "Dutch": ("nl", "Dutch"),
    "English": ("en", "English"),
    "Filipino (Tagalog)": ("tl", "Filipino (Tagalog)"),
    "Finnish": ("fi", "Finnish"),
    "French": ("fr", "French"),
    "German": ("de", "German"),
    "Greek": ("el", "Greek"),
    "Hebrew": ("he", "Hebrew"),
    "Hindi": ("hi", "Hindi"),
    "Hungarian": ("hu", "Hungarian"),
    "Indonesian": ("id", "Indonesian"),
    "Italian": ("it", "Italian"),
    "Japanese": ("ja", "Japanese"),
    "Korean": ("ko", "Korean"),
    "Malay": ("ms", "Malay"),
    "Norwegian": ("no", "Norwegian"),
    "Persian (Farsi)": ("fa", "Persian (Farsi)"),
    "Polish": ("pl", "Polish"),
    "Portuguese": ("pt", "Portuguese"),
    "Romanian": ("ro", "Romanian"),
    "Russian": ("ru", "Russian"),
    "Slovak": ("sk", "Slovak"),
    "Spanish": ("es", "Spanish"),
    "Swedish": ("sv", "Swedish"),
    "Thai": ("th", "Thai"),
    "Turkish": ("tr", "Turkish"),
    "Ukrainian": ("uk", "Ukrainian"),
    "Urdu": ("ur", "Urdu"),
    "Vietnamese": ("vi", "Vietnamese"),
}
ANALYSIS_LANGUAGE_CHOICES = {ad: kod for ad, (kod, _) in ANALYSIS_LANGUAGES.items()}
GEMINI_OUTPUT_LANGUAGE_NAMES = {
    kod.lower(): gemini_ad for _, (kod, gemini_ad) in ANALYSIS_LANGUAGES.items()
}


def gemini_output_language_name(output_lang: str) -> str:
    ham = (output_lang or "en").strip().lower()
    if ham in GEMINI_OUTPUT_LANGUAGE_NAMES:
        return GEMINI_OUTPUT_LANGUAGE_NAMES[ham]
    kok = ham.split("-")[0]
    return GEMINI_OUTPUT_LANGUAGE_NAMES.get(kok, "English")
GEMINI_INSIGHT_REQUEST_TIMEOUT_SEC = 28
GEMINI_INSIGHT_TOTAL_TIMEOUT_SEC = 35
GEMINI_INSIGHT_MAX_TEXT_CHARS = 1000
GEMINI_INSIGHT_RETRY_CHARS = 550
# AQ… keys (AI Studio / consumer) expose Gemini 3.x flash family; 2.0 is 404.
GEMINI_INSIGHT_MODEL_PREFERENCES = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
)
GEMINI_CLIENT_API_VERSIONS = ("v1beta",)
GEMINI_CLOUD_MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
)
GEMINI_CLOUD_API_VERSIONS = ("v1beta",)
GEMINI_CLOUD_REQUEST_TIMEOUT_SEC = 25
GEMINI_CLOUD_TOTAL_TIMEOUT_SEC = 32
GEMINI_CLOUD_MAX_TEXT_CHARS = 750
GEMINI_MAX_MODEL_ATTEMPTS = 3
LITERARY_MIN_CHARS = 220


class GeminiConnectionTimeout(Exception):
    """Gemini generate_content did not finish within the allowed wait time."""


class GeminiServiceUnavailable(Exception):
    """Gemini returned overload / capacity errors for all attempted models."""


def _gemini_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "503",
            "504",
            "429",
            "unavailable",
            "high demand",
            "resource exhausted",
            "resource_exhausted",
            "overloaded",
            "deadline_exceeded",
            "deadline expired",
        )
    )


def _gemini_deadline(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "504" in msg
        or "deadline_exceeded" in msg
        or "deadline expired" in msg
    )


def _gemini_model_not_found(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "404" in msg or "not_found" in msg or "is not found" in msg


def _is_streamlit_cloud() -> bool:
    try:
        url = (st.context.url or "").lower()
        return "streamlit.app" in url or "streamlit.cloud" in url
    except Exception:
        return False


def _gemini_api_key() -> str:
    raw = st.secrets["GEMINI_API_KEY"]
    return str(raw).strip().strip('"').strip("'")


def _gemini_insight_profile() -> dict:
    if _is_streamlit_cloud():
        return {
            "request_timeout_sec": GEMINI_CLOUD_REQUEST_TIMEOUT_SEC,
            "total_timeout_sec": GEMINI_CLOUD_TOTAL_TIMEOUT_SEC,
            "max_text_chars": GEMINI_CLOUD_MAX_TEXT_CHARS,
            "api_versions": GEMINI_CLOUD_API_VERSIONS,
            "model_preferences": GEMINI_CLOUD_MODELS,
            "discover_models": True,
        }
    return {
        "request_timeout_sec": GEMINI_INSIGHT_REQUEST_TIMEOUT_SEC,
        "total_timeout_sec": GEMINI_INSIGHT_TOTAL_TIMEOUT_SEC,
        "max_text_chars": GEMINI_INSIGHT_MAX_TEXT_CHARS,
        "api_versions": GEMINI_CLIENT_API_VERSIONS,
        "model_preferences": GEMINI_INSIGHT_MODEL_PREFERENCES,
        "discover_models": True,
    }


def _gemini_insight_clients(profile: dict):
    for api_version in profile["api_versions"]:
        opts: dict = {"timeout": profile["request_timeout_sec"] * 1000}
        if api_version:
            opts["api_version"] = api_version
        yield genai.Client(api_key=_gemini_api_key(), http_options=opts)


def _model_supports_generate_content(model: object) -> bool:
    for attr in ("supported_actions", "supported_generation_methods"):
        actions = getattr(model, attr, None) or []
        if "generateContent" in actions:
            return True
    return False


def _gemini_model_id_variants(model: object) -> list[str]:
    full = (getattr(model, "name", None) or "").strip()
    short = full.split("/")[-1] if full else ""
    variants: list[str] = []
    for candidate in (short, full):
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


@st.cache_resource(show_spinner=False)
def _gemini_discovered_model_ids(api_version: str | None) -> tuple[str, ...]:
    opts: dict = {"timeout": 20_000}
    if api_version:
        opts["api_version"] = api_version
    client = genai.Client(api_key=_gemini_api_key(), http_options=opts)
    discovered: list[str] = []
    try:
        for model in client.models.list():
            if not _model_supports_generate_content(model):
                continue
            for model_id in _gemini_model_id_variants(model):
                low = model_id.lower()
                if "gemini" not in low:
                    continue
                if any(
                    token in low
                    for token in (
                        "tts",
                        "image",
                        "live",
                        "robotics",
                        "computer-use",
                        "embedding",
                        "preview-tts",
                    )
                ):
                    continue
                if model_id not in discovered:
                    discovered.append(model_id)
    except Exception:
        return tuple()
    return tuple(discovered)


def _gemini_insight_model_ids(client, profile: dict) -> list[str]:
    preferences = list(profile["model_preferences"])
    if not profile["discover_models"]:
        return preferences[:GEMINI_MAX_MODEL_ATTEMPTS]

    api_version = None
    try:
        api_version = getattr(
            getattr(client, "_api_client", None), "api_version", None
        )
    except Exception:
        api_version = None
    discovered = [
        mid
        for mid in _gemini_discovered_model_ids(api_version)
        if "gemini-2.0" not in mid.lower()
    ]
    # Prefer known-good AQ / 3.x flash-lite IDs first, even before discovery order.
    ordered: list[str] = []
    for preferred in preferences:
        if preferred not in ordered:
            if (not discovered) or preferred in discovered:
                ordered.append(preferred)
        if len(ordered) >= GEMINI_MAX_MODEL_ATTEMPTS:
            return ordered
    for model_id in discovered:
        low = model_id.lower()
        if model_id in ordered:
            continue
        # Skip slow/heavy preview/pro variants when we already have lite options.
        if any(token in low for token in ("-pro", "argon", "carbon", "hydrogen", "omni", "transcribe", "nano-banana")):
            continue
        ordered.append(model_id)
        if len(ordered) >= GEMINI_MAX_MODEL_ATTEMPTS:
            break
    return ordered[:GEMINI_MAX_MODEL_ATTEMPTS] or preferences[:GEMINI_MAX_MODEL_ATTEMPTS]


def _gemini_metin_ozeti(metin: str, max_chars: int) -> str:
    temiz = metin.strip()
    if len(temiz) <= max_chars:
        return temiz
    # Keep opening + closing so long essays still carry arc / punchline.
    bas = max(220, int(max_chars * 0.62))
    son = max(120, max_chars - bas - 40)
    if bas + son >= len(temiz):
        return temiz[:max_chars]
    return (
        f"{temiz[:bas].rstrip()}\n\n"
        f"[…middle omitted for length…]\n\n"
        f"{temiz[-son:].lstrip()}"
    )


EMOTION_LABELS_EN = {
    "admiration": "Admiration",
    "amusement": "Amusement",
    "anger": "Anger",
    "annoyance": "Annoyance",
    "approval": "Approval",
    "caring": "Caring",
    "confusion": "Confusion",
    "curiosity": "Curiosity",
    "desire": "Desire",
    "disappointment": "Disappointment",
    "disapproval": "Disapproval",
    "disgust": "Disgust",
    "embarrassment": "Embarrassment",
    "excitement": "Excitement",
    "fear": "Fear",
    "gratitude": "Gratitude",
    "grief": "Grief",
    "joy": "Joy",
    "love": "Love",
    "nervousness": "Nervousness",
    "optimism": "Optimism",
    "pride": "Pride",
    "realization": "Realization",
    "relief": "Relief",
    "remorse": "Remorse",
    "sadness": "Sadness",
    "surprise": "Surprise",
    "neutral": "Neutrality",
}

EMOTION_LABELS_TR = {
    "admiration": "Hayranlık",
    "amusement": "Eğlence",
    "anger": "Öfke",
    "annoyance": "Rahatsızlık",
    "approval": "Onay",
    "caring": "Şefkat",
    "confusion": "Kafa Karışıklığı",
    "curiosity": "Merak",
    "desire": "Arzu",
    "disappointment": "Hayal Kırıklığı",
    "disapproval": "Red",
    "disgust": "İğrenme",
    "embarrassment": "Utanç",
    "excitement": "Heyecan",
    "fear": "Korku",
    "gratitude": "Minnet",
    "grief": "Keder",
    "joy": "Sevinç",
    "love": "Sevgi",
    "nervousness": "Gerginlik",
    "optimism": "İyimserlik",
    "pride": "Gurur",
    "realization": "Farkındalık",
    "relief": "Rahatlama",
    "remorse": "Pişmanlık",
    "sadness": "Hüzün",
    "surprise": "Şaşkınlık",
    "neutral": "Tarafsızlık",
}

EMOTION_LABELS_FR = {
    "admiration": "Admiration",
    "amusement": "Amusement",
    "anger": "Colère",
    "annoyance": "Agacement",
    "approval": "Approbation",
    "caring": "Sollicitude",
    "confusion": "Confusion",
    "curiosity": "Curiosité",
    "desire": "Désir",
    "disappointment": "Déception",
    "disapproval": "Désapprobation",
    "disgust": "Dégoût",
    "embarrassment": "Embarras",
    "excitement": "Exaltation",
    "fear": "Peur",
    "gratitude": "Gratitude",
    "grief": "Chagrin",
    "joy": "Joie",
    "love": "Amour",
    "nervousness": "Nervosité",
    "optimism": "Optimisme",
    "pride": "Fierté",
    "realization": "Prise de conscience",
    "relief": "Soulagement",
    "remorse": "Remords",
    "sadness": "Tristesse",
    "surprise": "Surprise",
    "neutral": "Neutralité",
}

EMOTION_LABELS_BY_LANG: dict[str, dict[str, str]] = {
    "en": EMOTION_LABELS_EN,
    "tr": EMOTION_LABELS_TR,
    "fr": EMOTION_LABELS_FR,
}

GOOGLE_EN = "en"

# Site footer keeps Seneca; analysis panel picks another voice at random.
SITE_MOTTO = (
    "“The mind that is free from disturbance is the true wealth.”",
    "— Seneca, Moral Letters to Lucilius",
)
PHILOSOPHER_QUOTES: tuple[tuple[str, str], ...] = (
    SITE_MOTTO,
    (
        "“The unexamined life is not worth living.”",
        "— Socrates, via Plato’s Apology",
    ),
    (
        "“We are what we repeatedly do. Excellence, then, is not an act but a habit.”",
        "— Aristotle (commonly attributed)",
    ),
    (
        "“He who has a why to live can bear almost any how.”",
        "— Friedrich Nietzsche",
    ),
    (
        "“Man is condemned to be free.”",
        "— Jean-Paul Sartre",
    ),
    (
        "“In the depth of winter, I finally learned that within me there lay an invincible summer.”",
        "— Albert Camus",
    ),
    (
        "“The limits of my language mean the limits of my world.”",
        "— Ludwig Wittgenstein",
    ),
    (
        "“One cannot step twice into the same river.”",
        "— Heraclitus",
    ),
    (
        "“I think, therefore I am.”",
        "— René Descartes",
    ),
    (
        "“Happiness is not an ideal of reason but of imagination.”",
        "— Immanuel Kant",
    ),
    (
        "“The life of man [is] solitary, poor, nasty, brutish, and short.”",
        "— Thomas Hobbes",
    ),
    (
        "“God is dead. God remains dead. And we have killed him.”",
        "— Friedrich Nietzsche",
    ),
    (
        "“To be is to be perceived.”",
        "— George Berkeley",
    ),
    (
        "“Entities should not be multiplied beyond necessity.”",
        "— William of Ockham (commonly attributed)",
    ),
    (
        "“The owl of Minerva spreads its wings only with the falling of the dusk.”",
        "— G. W. F. Hegel",
    ),
    (
        "“Whereof one cannot speak, thereof one must be silent.”",
        "— Ludwig Wittgenstein",
    ),
    (
        "“Hell is other people.”",
        "— Jean-Paul Sartre",
    ),
    (
        "“There is nothing either good or bad, but thinking makes it so.”",
        "— William Shakespeare (Hamlet)",
    ),
    (
        "“Know thyself.”",
        "— Delphic maxim",
    ),
    (
        "“The only true wisdom is in knowing you know nothing.”",
        "— Socrates (commonly attributed)",
    ),
    (
        "“Man is born free, and everywhere he is in chains.”",
        "— Jean-Jacques Rousseau",
    ),
    (
        "“That which does not kill us makes us stronger.”",
        "— Friedrich Nietzsche",
    ),
    (
        "“We live in the best of all possible worlds.”",
        "— Gottfried Wilhelm Leibniz (paraphrased)",
    ),
    (
        "“Existence precedes essence.”",
        "— Jean-Paul Sartre",
    ),
    (
        "“The medium is the message.”",
        "— Marshall McLuhan",
    ),
    (
        "“All that is solid melts into air.”",
        "— Karl Marx & Friedrich Engels",
    ),
    (
        "“Time is a created thing. To say ‘I don’t have time’ is to say ‘I don’t want to.’”",
        "— Lao Tzu (commonly attributed)",
    ),
    (
        "“The map is not the territory.”",
        "— Alfred Korzybski",
    ),
    (
        "“One must imagine Sisyphus happy.”",
        "— Albert Camus",
    ),
    (
        "“Freedom is what you do with what’s been done to you.”",
        "— Jean-Paul Sartre (commonly attributed)",
    ),
    (
        "“Doubt is not a pleasant condition, but certainty is absurd.”",
        "— Voltaire",
    ),
    (
        "“We are asleep. Our Life is a dream. But we wake up sometimes, just enough to know that we are dreaming.”",
        "— Ludwig Wittgenstein",
    ),
)


def rastgele_analiz_sozu() -> tuple[str, str]:
    """Quote for the results panel; never repeats the fixed site motto."""
    adaylar = [q for q in PHILOSOPHER_QUOTES if q != SITE_MOTTO]
    return random.choice(adaylar)


UI_EN = {
    "page_title": "Semantic & Philosophical Text Analyzer",
    "kicker": "Philosophy",
    "title": "Semantic & Philosophical Text Analyzer",
    "info_title": "About this analyzer",
    "text_to_analyze": "Text to analyze",
    "analysis_language": "Analysis language",
    "results_language": "Results language",
    "placeholder": (
        "Paste a passage in any language… "
        "(about 100+ characters for philosophical insight)"
    ),
    "analyze_button": "Analyze",
    "warning_empty": "Paste a passage before running the analysis.",
    "warning_too_short": (
        "This text is short for deep analysis. Emotion spectrum still runs; "
        "add roughly {min_chars}+ characters for a philosophical insight."
    ),
    "input_hint": (
        "Reveals the conceptual structure, emotional temper, and "
        "philosophical undertow of a text."
    ),
    "char_count": "{n} characters",
    "results_label": "Results: Emotion spectrum",
    "motto": SITE_MOTTO[0],
    "motto_attr": SITE_MOTTO[1],
    "insight_kicker": "Philosophical reading",

    "spinner": "Translating and mapping emotional tone…",
    "status_detect": "Detecting source language…",
    "status_translate": "Translating to English for the model…",
    "status_model": "Loading Go Emotions model (first run can take 1–2 minutes on the server)…",
    "status_score": "Scoring emotional tone…",
    "status_emotions_done": "Emotion analysis complete",
    "spinner_insight": "Generating philosophical insight (Gemini)…",
    "spinner_emotions": "Scoring emotional tone…",
    "error_prefix": "Analysis could not be completed:",
    "emotion": "Emotion",
    "intensity": "Intensity",
    "score": "Score",
    "dominant": "Dominant emotion",
    "result_heading": "Emotion spectrum",
    "source_language": "Source language",
    "footer_note": (
        "First 512 tokens of the English translation were analyzed. "
        "Scores are multi-label probabilities."
    ),
    "tools_heading": "About this tool",
    "methodology_title": "Methodology & Linguistic Limitations",
    "methodology_body": (
        "This tool uses an English bridge to analyze many non-English passages. "
        "We are linguistically aware that existential and cultural emotional "
        "states unique to a language may undergo semantic shift when rendered "
        "in English. The emotion spectrum aims at a broadly comparable profile "
        "rather than a perfect map of the source language's feeling-vocabulary."
    ),
    "methodology_points": (
        "Very short texts receive emotion labels only; philosophical insight "
        "needs a longer passage so the model does not over-read ordinary lines.",
        "On longer passages, emotions and insight share one Gemini pass on the "
        "original wording (tone and subtext); RoBERTa remains the fallback.",
        "Neutrality is softened unless the line is a true greeting; affective "
        "labels are preferred when any clear mood is present.",
        "For scoring fallback, non-English text may be translated into English; "
        "that bridge can still shift cultural feeling-vocabulary.",
        "Long RoBERTa fallbacks use peak-weighted windows so strong local tone "
        "is not washed out by flat stretches.",
        "Bar lengths show relative share among the top three labels, not "
        "independent probabilities that sum to 100%.",
    ),
    "model_loading": "Loading emotion classifier…",
    "insight_title": "Semantic & Philosophical Insight",
    "insight_unavailable": "Philosophical insight is temporarily unavailable.",
    "feedback_prompt": "Was this insight perceptive?",
    "feedback_up": "Helpful",
    "feedback_down": "Off",
    "feedback_thanks": "Thanks — this will shape future readings.",
    "feedback_saved": "Feedback saved.",
    "feedback_local_only": (
        "Feedback is saving locally right now. Add SUPABASE_URL and "
        "SUPABASE_KEY in secrets for durable multi-user storage."
    ),
}


@st.cache_data(show_spinner=False)
def mymemory_destekli_localeler() -> frozenset[str]:
    return frozenset(MY_MEMORY_LANGUAGES_TO_CODES.values())


def detect_source_lang(metin: str) -> str:
    try:
        return detect(metin).lower()
    except LangDetectException:
        return "en"


def mymemory_locale(kod: str) -> str:
    """ISO dil kodunu MyMemory locale koduna çevir."""
    kod = (kod or "auto").lower()
    if kod == "auto":
        raise LanguageNotSupportedException(
            "auto",
            message="MyMemory requires an explicit source language, not auto.",
        )

    destek = mymemory_destekli_localeler()
    if kod in destek:
        return kod

    if kod.startswith("en"):
        if MYMEMORY_INGILIZCE in destek:
            return MYMEMORY_INGILIZCE
        return next((d for d in ("en-GB", "en-US") if d in destek), "en-US")

    if kod.startswith("zh"):
        tercih = "zh-CN" if "cn" in kod else "zh-TW"
        if tercih in destek:
            return tercih

    kok = kod.split("-")[0]
    tercih = ISO_MYMEMORY_TERCIH.get(kok)
    if tercih and tercih in destek:
        return tercih

    for tam in sorted(destek):
        if tam.split("-")[0].lower() == kok:
            return tam

    raise LanguageNotSupportedException(
        kod,
        message=f"MyMemory bu dil kodunu desteklemiyor: {kod}",
    )


def gtx_translate(temiz: str, kaynak: str, hedef: str) -> str:
    kaynak = (kaynak or "auto").lower()
    hedef = (hedef or GOOGLE_EN).lower()
    sl = "auto" if kaynak == "auto" else kaynak.split("-")[0]
    tl = GOOGLE_EN if hedef.startswith("en") else hedef.split("-")[0]
    sorgu = urllib.parse.urlencode(
        {
            "client": "gtx",
            "sl": sl,
            "tl": tl,
            "dt": "t",
            "q": temiz[:4500],
        }
    )
    istek = urllib.request.Request(
        f"https://translate.googleapis.com/translate_a/single?{sorgu}"
    )
    with urllib.request.urlopen(istek, timeout=25) as yanit:
        if yanit.status == 429:
            raise TooManyRequests()
        parcalar = json.loads(yanit.read().decode())[0]
    return "".join(p[0] for p in parcalar if p and p[0]).strip()


def mymemory_translate(temiz: str, kaynak: str, hedef: str) -> str:
    kaynak = (kaynak or GOOGLE_EN).lower()
    hedef = (hedef or GOOGLE_EN).lower()
    if kaynak.startswith("en") and hedef.startswith("en"):
        return temiz
    mm_kaynak = mymemory_locale(kaynak)
    mm_hedef = mymemory_locale(hedef)
    return str(
        MyMemoryTranslator(source=mm_kaynak, target=mm_hedef).translate(temiz)
    ).strip()


def guvenli_cevir(metin: str, kaynak: str, hedef: str) -> str:
    """Google → MyMemory → gtx (kaynak/hedef ISO; İngilizce için en / en-US)."""
    kaynak = (kaynak or GOOGLE_EN).lower()
    hedef = (hedef or GOOGLE_EN).lower()
    temiz = metin.strip()
    if not temiz:
        return ""
    if kaynak.split("-")[0] == hedef.split("-")[0] or (
        kaynak.startswith("en") and hedef.startswith("en")
    ):
        return temiz

    google_src = "auto" if kaynak == "auto" else kaynak
    google_tgt = GOOGLE_EN if hedef.startswith("en") else hedef
    # Cap outbound translation wait so Analyze never spins forever.
    ceviri_timeout = 12

    google_hata: str | None = None
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            sonuc = pool.submit(
                GoogleTranslator(source=google_src, target=google_tgt).translate,
                temiz,
            ).result(timeout=ceviri_timeout)
        if sonuc and sonuc.strip():
            return sonuc.strip()
    except FuturesTimeoutError:
        google_hata = "timeout"
    except Exception as exc:
        google_hata = str(exc)

    mymemory_hata: str | None = None
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            sonuc = pool.submit(
                mymemory_translate, temiz, kaynak, hedef
            ).result(timeout=ceviri_timeout)
        if sonuc:
            return sonuc
    except FuturesTimeoutError:
        mymemory_hata = "timeout"
    except Exception as exc:
        mymemory_hata = str(exc)

    gtx_hata: str | None = None
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            sonuc = pool.submit(
                gtx_translate, temiz, kaynak, hedef
            ).result(timeout=ceviri_timeout)
        if sonuc:
            return sonuc
    except FuturesTimeoutError:
        gtx_hata = "timeout"
    except Exception as exc:
        gtx_hata = str(exc)

    raise RuntimeError(
        "Translation failed."
        + (f" Google: {google_hata}." if google_hata else "")
        + (f" MyMemory: {mymemory_hata}." if mymemory_hata else "")
        + (f" gtx: {gtx_hata}." if gtx_hata else "")
    )


@st.cache_resource(show_spinner=False)
def load_emotion_classifier():
    return pipeline(
        "text-classification",
        model=ROBERTA_MODEL_ID,
        top_k=None,
    )


def _gemini_generate_text(prompt: str) -> str:
    """Call Gemini on the main thread; rely on HTTP timeouts (no nested pools)."""
    profile = _gemini_insight_profile()
    last_exc: Exception | None = None
    saw_not_found = False
    for client in _gemini_insight_clients(profile):
        model_ids = _gemini_insight_model_ids(client, profile)
        for model_id in model_ids:
            try:
                yanit = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                )
            except Exception as exc:
                exc_name = type(exc).__name__.lower()
                last_exc = exc
                if _gemini_model_not_found(exc):
                    saw_not_found = True
                    continue
                if (
                    "timeout" in exc_name
                    or "timed out" in str(exc).lower()
                    or _gemini_deadline(exc)
                    or _gemini_retryable(exc)
                ):
                    continue
                raise
            metin_yanit = (yanit.text or "").strip()
            if not metin_yanit:
                raise RuntimeError("Gemini returned empty insight text.")
            return metin_yanit
    if last_exc and (
        _gemini_deadline(last_exc)
        or "timed out" in str(last_exc).lower()
        or "timeout" in type(last_exc).__name__.lower()
    ):
        raise GeminiConnectionTimeout from last_exc
    if saw_not_found and last_exc:
        raise RuntimeError(
            "Bu API anahtarı için generateContent destekleyen uygun bir "
            "Gemini modeli bulunamadı (404). Streamlit Cloud → Secrets "
            "içindeki GEMINI_API_KEY değerinin, localhost’taki "
            "secrets.toml ile birebir aynı AQ… veya AIza… anahtarı "
            "olduğundan emin ol (boşluk/tırnak yok)."
        ) from last_exc
    if last_exc and _gemini_retryable(last_exc):
        raise GeminiServiceUnavailable from last_exc
    if last_exc:
        raise last_exc
    raise RuntimeError("Gemini insight çağrısı yapılandırılamadı.")


def _gemini_insight_prompt(
    metin_ozet: str, output_lang: str, method_examples: str = ""
) -> str:
    lang_name = gemini_output_language_name(output_lang)
    ornek_blok = f"\n\n{method_examples}\n" if method_examples else "\n"
    return (
        f"{GEMINI_INSIGHT_PROMPT}\n"
        f"Please provide the entire analysis and response strictly in {lang_name}."
        f"{ornek_blok}"
        f"Text:\n{metin_ozet}"
    )


def _gemini_with_length_fallback(
    metin: str, output_lang: str, prompt_builder
) -> str:
    profile = _gemini_insight_profile()
    limit = profile["max_text_chars"]
    if len(metin.strip()) > profile["max_text_chars"] * 1.4:
        # Long texts: try compact excerpt first to avoid 504.
        limit = min(limit, GEMINI_INSIGHT_RETRY_CHARS + 150)
    ozet = _gemini_metin_ozeti(metin, limit)
    try:
        return _gemini_generate_text(prompt_builder(ozet, output_lang))
    except (GeminiConnectionTimeout, GeminiServiceUnavailable):
        if limit <= GEMINI_INSIGHT_RETRY_CHARS:
            raise
        kisa = _gemini_metin_ozeti(metin, GEMINI_INSIGHT_RETRY_CHARS)
        return _gemini_generate_text(prompt_builder(kisa, output_lang))


def _literary_match_bridge(metin: str) -> str:
    """English text used only for literary-ref matching across languages."""
    temiz = metin.strip()
    if not temiz:
        return ""
    try:
        source_lang = detect_source_lang(temiz)
    except Exception:
        source_lang = "en"
    if (source_lang or "en").lower().startswith("en"):
        return temiz
    try:
        return metni_ingilizceye_cevir(temiz, source_lang) or temiz
    except Exception:
        return temiz


def _method_examples_bundle(metin: str) -> str:
    """Curated refs + user feedback exemplars for Gemini method guidance."""
    match_text = _literary_match_bridge(metin)
    # Prefer one fewer static ref when approved feedback exists (prompt budget).
    up_var = feedback_memory.retrieve_feedback(
        metin, vote="up", k=1, match_text=match_text
    )
    ref_k = 1 if up_var else literary_memory.MAX_EXAMPLES
    curated = literary_memory.method_examples_for(
        metin, k=ref_k, match_text=match_text
    )
    learned = feedback_memory.format_feedback_for_prompt(
        metin, match_text=match_text
    )
    return "\n\n".join(part for part in (curated, learned) if part)


def gemini_philosophical_insight(metin: str, output_lang: str) -> str:
    temiz = metin.strip()
    if not temiz:
        return ""
    method_examples = _method_examples_bundle(temiz)

    def _builder(ozet: str, lang: str) -> str:
        return _gemini_insight_prompt(ozet, lang, method_examples)

    return _gemini_with_length_fallback(temiz, output_lang, _builder)


def _metin_derin_analiz_uygun(metin: str) -> bool:
    """Skip Gemini insight / literary path on thin or phatic inputs."""
    temiz = metin.strip()
    if not temiz or _phatic_metin(temiz):
        return False
    kelime = len(temiz.split())
    if len(temiz) >= MIN_CHARS_DEEP and kelime >= MIN_WORDS_DEEP:
        return True
    # Compact literary numbness / withheld-feeling signals still deserve Gemini.
    if (
        len(temiz) >= 55
        and kelime >= 8
        and (
            EDISI_DUYGU_INKAR.search(temiz)
            or EDISI_BEKLENEN_DUYGU.search(temiz)
        )
    ):
        return True
    return False


def _katmanli_edebi_metin(metin: str) -> bool:
    if not _metin_derin_analiz_uygun(metin):
        return False
    temiz = metin.strip()
    if len(temiz) < LITERARY_MIN_CHARS:
        return False
    kelime = len(temiz.split())
    if EDISI_DUYGU_INKAR.search(temiz) or EDISI_BEKLENEN_DUYGU.search(temiz):
        return True
    if kelime >= LITERARY_MIN_WORDS and (
        EDISI_KARSIT.search(temiz) or EDISI_BENZETME.search(temiz)
    ):
        return True
    if kelime >= 45:
        return True
    return False


def _parse_gemini_analysis_json(ham: str) -> tuple[list[dict], str]:
    metin = ham.strip()
    kod_blogu = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", metin)
    if kod_blogu:
        metin = kod_blogu.group(1).strip()
    baslangic = metin.find("{")
    bitis = metin.rfind("}")
    if baslangic != -1 and bitis != -1:
        metin = metin[baslangic : bitis + 1]
    data = json.loads(metin)
    ham_duygular = data.get("emotions") or []
    insight = (data.get("insight") or "").strip()
    if not insight:
        raise RuntimeError("Gemini returned empty insight text.")

    gecerli = set(EMOTION_LABELS_EN.keys())
    duygular: list[dict] = []
    for kayit in ham_duygular:
        if not isinstance(kayit, dict):
            continue
        etiket = str(kayit.get("label", "")).strip().lower().replace(" ", "_")
        if etiket not in gecerli:
            continue
        try:
            skor = float(kayit.get("score", 0))
        except (TypeError, ValueError):
            continue
        duygular.append(
            {"label": etiket, "score": max(0.0, min(1.0, skor))}
        )
    duygular.sort(key=lambda x: x["score"], reverse=True)
    if len(duygular) < 1:
        raise RuntimeError("Gemini returned no valid emotion labels.")
    return duygular[:3], insight


def _gemini_birlesik_prompt(
    metin_ozet: str, output_lang: str, method_examples: str = ""
) -> str:
    lang_name = gemini_output_language_name(output_lang)
    etiketler = ", ".join(sorted(EMOTION_LABELS_EN.keys()))
    ornek_blok = f"\n{method_examples}\n\n" if method_examples else "\n"
    return (
        "Analyze the following text in its original language—including "
        "everyday prose, literature, poetry, irony, and philosophical writing.\n\n"
        f"{GEMINI_INSIGHT_PROMPT}\n"
        f"Write the insight field strictly in {lang_name}.\n\n"
        f"{GEMINI_EMOTION_CALIBRATION}\n"
        f"{ornek_blok}"
        "Return exactly three salient Go Emotions labels with confidence "
        f"scores between 0 and 1 (independent). Allowed labels: {etiketler}\n\n"
        "Respond with ONLY valid JSON (no markdown):\n"
        '{"emotions": [{"label": "joy", "score": 0.55}, '
        '{"label": "optimism", "score": 0.30}, {"label": "love", "score": 0.15}], '
        f'"insight": "..."}}\n\n'
        f"Text:\n{metin_ozet}"
    )


def _anlatici_ton_mu(metin: str) -> bool:
    """True when the voice signals numbness / withheld expected feeling."""
    temiz = (metin or "").strip()
    if not temiz:
        return False
    if EDISI_DUYGU_INKAR.search(temiz) or EDISI_BEKLENEN_DUYGU.search(temiz):
        return True
    # Event of loss + cool distancing words often implies tone ≠ grief.
    if EDISI_OLAY_YASI.search(temiz) and re.search(
        r"\b(stranger|alien|foreign|absurd|mechanical|ordinary|nothing|"
        r"yabancı|yabanci|saçma|mekanik|sıradan|siradan|rien|absurde)\b",
        temiz,
        re.IGNORECASE,
    ):
        return True
    return False


def _duygu_olay_yanliligi_duzelt(
    skorlar: list[dict], metin: str
) -> list[dict]:
    """If loss is topical but voice is numb, demote griefish labels."""
    if not skorlar or not _anlatici_ton_mu(metin):
        return skorlar
    if not any(
        s["label"].strip().lower() in GRIEFISH_LABELS for s in skorlar
    ):
        return skorlar
    ayarlanmis: list[dict] = []
    for kayit in skorlar:
        etiket = kayit["label"].strip().lower()
        skor = float(kayit["score"])
        if etiket in GRIEFISH_LABELS:
            skor *= 0.45
        elif etiket in TON_LABELS:
            skor = min(1.0, skor * 1.12 + 0.04)
        ayarlanmis.append({"label": kayit["label"], "score": skor})
    ayarlanmis.sort(key=lambda x: x["score"], reverse=True)
    return ayarlanmis


def gemini_birlesik_analiz(
    metin: str, output_lang: str
) -> tuple[list[dict], str]:
    """Single Gemini pass: top-3 emotions + philosophical insight."""
    temiz = metin.strip()
    if not temiz:
        return [], ""
    # Matching may use an English bridge; Gemini still reads the original text.
    # Approved/rejected user feedback is mixed into method guidance over time.
    method_examples = _method_examples_bundle(temiz)

    def _builder(ozet: str, lang: str) -> str:
        return _gemini_birlesik_prompt(ozet, lang, method_examples)

    ham = _gemini_with_length_fallback(temiz, output_lang, _builder)
    duygular, insight = _parse_gemini_analysis_json(ham)
    # Event≠tone correction runs inside ust_uc_duzenle.
    return ust_uc_duzenle(duygular, temiz), insight


def gemini_edebi_birlesik(
    metin: str, output_lang: str
) -> tuple[list[dict], str]:
    """Backward-compatible alias for unified deep analysis."""
    return gemini_birlesik_analiz(metin, output_lang)


def insight_in_source_language(insight_en: str, source_lang: str) -> str:
    source_lang = (source_lang or "en").lower()
    if source_lang.startswith("en"):
        return insight_en
    return guvenli_cevir(insight_en, GOOGLE_EN, source_lang)


def metni_ingilizceye_cevir(metin: str, source_lang: str) -> str:
    temiz = metin.strip()
    if not temiz:
        return ""
    source_lang = (source_lang or "en").lower()
    if source_lang.startswith("en"):
        return temiz
    return guvenli_cevir(temiz, source_lang, GOOGLE_EN)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 30)
def emotion_labels_for_lang(lang_code: str) -> dict[str, str]:
    ham = (lang_code or "en").strip()
    kok = ham.lower().split("-")[0]
    if kok == "en":
        return EMOTION_LABELS_EN
    if kok in EMOTION_LABELS_BY_LANG:
        return EMOTION_LABELS_BY_LANG[kok]
    hedef = ham if "-" in ham else kok
    sozluk: dict[str, str] = {}
    for anahtar, ingilizce in EMOTION_LABELS_EN.items():
        try:
            cevirilmis = guvenli_cevir(ingilizce, GOOGLE_EN, hedef)
            sozluk[anahtar] = cevirilmis or ingilizce
        except Exception:
            sozluk[anahtar] = ingilizce
    return sozluk


def emotion_display(label: str, source_lang: str) -> str:
    key = label.strip().lower()
    sozluk = emotion_labels_for_lang(source_lang)
    return sozluk.get(key, EMOTION_LABELS_EN.get(key, label.replace("_", " ").title()))


def ust_uc_gemini(skorlar: list[dict]) -> list[dict]:
    sirali = sorted(skorlar, key=lambda x: x["score"], reverse=True)
    return _gosterim_uctus_normallestir(sirali[:3])


def _gosterim_uctus_normallestir(secilen: list[dict]) -> list[dict]:
    top = secilen[:3]
    toplam = sum(kayit["score"] for kayit in top) or 1.0
    return [
        {"label": kayit["label"], "score": kayit["score"] / toplam}
        for kayit in top
    ]


def _phatic_metin(metin: str) -> bool:
    temiz = " ".join(metin.strip().split())
    if not temiz:
        return False
    if len(temiz.split()) > PHATIC_MAX_WORDS:
        return False
    if PHATIC_KALIP.match(temiz):
        return True
    # Soft phatic: very short greeting-like openers.
    return bool(
        re.match(
            r"^(hi|hello|hey|merhaba|selam)\b",
            temiz,
            re.IGNORECASE,
        )
        and len(temiz.split()) <= 6
    )


def _top3_notr_sonra(not_notr: list[dict], sirali: list[dict]) -> list[dict]:
    """Prefer non-neutral first; keep neutral in top-3 if needed for fill."""
    secilen: list[dict] = []
    for kayit in not_notr + sirali:
        if kayit in secilen:
            continue
        secilen.append(kayit)
        if len(secilen) == 3:
            break
    return _gosterim_uctus_normallestir(secilen)


def ust_uc_duzenle(skorlar: list[dict], metin: str = "") -> list[dict]:
    """Pick top-3; soften neutral dominance except true phatic greetings."""
    if metin:
        skorlar = _duygu_olay_yanliligi_duzelt(skorlar, metin)
    sirali = sorted(skorlar, key=lambda x: x["score"], reverse=True)
    if not sirali:
        return []
    not_notr = [
        s for s in sirali if s["label"].strip().lower() != "neutral"
    ]
    en_yuksek_notr = not_notr[0]["score"] if not_notr else 0.0
    neutral = next(
        (s for s in sirali if s["label"].strip().lower() == "neutral"), None
    )
    birinci_notr = sirali[0]["label"].strip().lower() == "neutral"
    phatic = _phatic_metin(metin)

    if phatic and birinci_notr and neutral:
        # Keep neutrality for greetings unless a challenger is clearly strong.
        if en_yuksek_notr >= PHATIC_CHALLENGER_MIN:
            return _top3_notr_sonra(not_notr, sirali)
        return _gosterim_uctus_normallestir(sirali)

    if birinci_notr and en_yuksek_notr >= NEUTRAL_CHALLENGER_MIN:
        # Soft C: neutral may remain in the table, but not as dominant.
        return _top3_notr_sonra(not_notr, sirali)

    if en_yuksek_notr >= SALIENT_EMOTION_MIN and len(not_notr) >= 3:
        return _gosterim_uctus_normallestir(not_notr[:3])

    if en_yuksek_notr >= SALIENT_EMOTION_MIN:
        return _top3_notr_sonra(not_notr, sirali)

    return _gosterim_uctus_normallestir(sirali[:3])


def duygu_tablosu_html(ust_uc: list[dict], source_lang: str) -> str:
    """Single-line HTML so Streamlit markdown does not break the block."""
    satirlar: list[str] = []
    for i, kayit in enumerate(ust_uc, start=1):
        ad = emotion_display(kayit["label"], source_lang)
        yuzde = kayit["score"] * 100
        genislik = min(100.0, max(0.0, yuzde))
        satirlar.append(
            f'<div class="emotion-row">'
            f'<span class="emotion-name">{ad}</span>'
            f'<div class="emotion-bar-track" aria-hidden="true">'
            f'<div class="emotion-bar-fill rank-{i}" style="width: {genislik:.1f}%;"></div>'
            f"</div>"
            f'<span class="emotion-pct">{yuzde:.1f}%</span>'
            f"</div>"
        )
    return '<div class="emotion-table">' + "".join(satirlar) + "</div>"


def _sprig_svg(css_class: str = "panel-sprig") -> str:
    return (
        f'<svg class="{css_class}" viewBox="0 0 64 64" aria-hidden="true">'
        '<path d="M32 58 C32 40 18 32 12 18 C22 20 30 28 32 40 '
        'C34 28 42 20 52 18 C46 32 32 40 32 58 Z" '
        'fill="none" stroke="currentColor" stroke-width="1.4"/>'
        '<path d="M32 58 V28" fill="none" stroke="currentColor" '
        'stroke-width="1.2"/>'
        "</svg>"
    )


def sonuc_paneli_html(
    baskın: str, tablo: str, soz: str, soz_kaynak: str
) -> str:
    return (
        f'<div class="result-panel">'
        f'<p class="results-kicker">{UI_EN["results_label"]}</p>'
        f'<div class="result-top">'
        f'<div class="dominant-block">'
        f'<p class="dominant-label">{UI_EN["dominant"]}</p>'
        f'<p class="dominant-emotion">{baskın}</p>'
        f"</div>"
        f'<div class="spectrum-block">{tablo}</div>'
        f"</div>"
        f'<div class="result-foot">'
        f'<div class="motto-block">'
        f'<p class="motto">{soz}</p>'
        f'<p class="motto-attr">{soz_kaynak}</p>'
        f"</div>"
        f"{_sprig_svg()}"
        f"</div>"
        f"</div>"
    )


def insight_paneli_html(metin: str) -> str:
    temiz = (metin or "").strip()
    if not temiz:
        return ""
    # Escape minimal HTML risk from model output while keeping paragraphs.
    guvenli = (
        temiz.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    paragraflar = [
        f"<p>{p.strip()}</p>"
        for p in re.split(r"\n\s*\n", guvenli)
        if p.strip()
    ]
    if not paragraflar:
        paragraflar = [f"<p>{guvenli}</p>"]
    return (
        f'<article class="insight-panel">'
        f'<div class="insight-head">'
        f'<h2 class="insight-title">{UI_EN["insight_title"]}</h2>'
        f"{_sprig_svg('insight-sprig')}"
        f"</div>"
        f'<div class="insight-ornament" aria-hidden="true">'
        f'<span class="insight-ornament-line"></span>'
        f'<span class="insight-ornament-dot"></span>'
        f'<span class="insight-ornament-line"></span>'
        f"</div>"
        f'<div class="insight-body">{"".join(paragraflar)}</div>'
        f"</article>"
    )


def _metin_roberta_girdisi(metin: str, source_lang: str) -> tuple[str, bool]:
    """RoBERTa Go Emotions is English-only; return model input and bridge flag."""
    temiz = metin.strip()
    if not temiz:
        return "", False
    kod = (source_lang or "en").lower().split("-")[0]
    if kod == "en":
        return temiz, False
    metin_en = metni_ingilizceye_cevir(temiz, source_lang)
    return (metin_en or temiz), bool(metin_en)


def _roberta_parcalari(metin: str) -> list[str]:
    kelimeler = metin.split()
    if len(kelimeler) <= ROBERTA_CHUNK_WORDS:
        return [metin]
    adim = max(1, ROBERTA_CHUNK_WORDS - ROBERTA_CHUNK_OVERLAP)
    parcalar: list[str] = []
    for bas in range(0, len(kelimeler), adim):
        parca = " ".join(kelimeler[bas : bas + ROBERTA_CHUNK_WORDS]).strip()
        if parca:
            parcalar.append(parca)
        if bas + ROBERTA_CHUNK_WORDS >= len(kelimeler):
            break
        if len(parcalar) >= ROBERTA_MAX_CHUNKS:
            break
    return parcalar or [metin]


def _roberta_skorlari_birlestir(parcalar_skor: list[list[dict]]) -> list[dict]:
    """Peak-weighted merge: emphatic chunks outweigh flat/neutral averages."""
    if not parcalar_skor:
        return []
    etiketler = {kayit["label"] for skorlar in parcalar_skor for kayit in skorlar}
    birlesik: list[dict] = []
    for etiket in etiketler:
        degerler: list[float] = []
        for skorlar in parcalar_skor:
            skor = next(
                (
                    float(k["score"])
                    for k in skorlar
                    if k["label"] == etiket
                ),
                0.0,
            )
            degerler.append(skor)
        if not degerler:
            continue
        ortalama = sum(degerler) / len(degerler)
        zirve = max(degerler)
        # Soft peak blend: keep presence of strong local emotion.
        birlesik.append(
            {"label": etiket, "score": 0.45 * ortalama + 0.55 * zirve}
        )
    birlesik.sort(key=lambda x: x["score"], reverse=True)
    return birlesik


def _roberta_emotion_scores(metin_model: str) -> list[dict]:
    classifier = load_emotion_classifier()
    parcalar = _roberta_parcalari(metin_model.strip())
    if len(parcalar) == 1:
        return classifier(
            parcalar[0],
            truncation=True,
            max_length=512,
        )[0]
    tum: list[list[dict]] = []
    for parca in parcalar:
        tum.append(
            classifier(
                parca,
                truncation=True,
                max_length=512,
            )[0]
        )
    return _roberta_skorlari_birlestir(tum)


def _analyze_emotions(
    metin: str, results_lang: str
) -> tuple[str, list[dict], str, bool]:
    source_lang = detect_source_lang(metin)
    metin_model, cevirildi = _metin_roberta_girdisi(metin, source_lang)
    skorlar = _roberta_emotion_scores(metin_model)
    ust_uc = ust_uc_duzenle(skorlar, metin)
    baskın = emotion_display(ust_uc[0]["label"], results_lang)
    return source_lang, ust_uc, baskın, cevirildi


def _fetch_insight(metin: str, results_lang: str) -> str:
    if "GEMINI_API_KEY" not in st.secrets:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it under Streamlit Cloud → "
            "Settings → Secrets (or .streamlit/secrets.toml locally)."
        )
    return gemini_philosophical_insight(metin, results_lang)


def _insight_hata_mesaji(exc: Exception) -> tuple[str, str] | None:
    if isinstance(exc, GeminiConnectionTimeout):
        return (
            "error",
            "Bağlantı zaman aşımına uğradı (504). Duygu analizi tamam; "
            "felsefi özet için metnin daha kısa bir özeti deneniyor veya "
            "biraz sonra tekrar Analyze Text’e basın.",
        )
    if isinstance(exc, GeminiServiceUnavailable):
        return (
            "warning",
            "Gemini şu an çok yoğun (503). Bu geçicidir; birkaç dakika "
            "sonra tekrar Analyze Text’e basın.",
        )
    if _gemini_deadline(exc):
        return (
            "error",
            "Bağlantı zaman aşımına uğradı (504). Daha kısa bir metin "
            "deneyin veya biraz sonra tekrar Analyze Text’e basın.",
        )
    if _gemini_retryable(exc):
        return (
            "warning",
            "Gemini şu an çok yoğun (503). Bu geçicidir; birkaç dakika "
            "sonra tekrar Analyze Text’e basın.",
        )
    return ("error", f"Hata Detayı: {exc}")


st.set_page_config(
    page_title="Semantic & Philosophical Text Analyzer",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_emotion_classifier()


@st.dialog(UI_EN["methodology_title"], width="large")
def methodology_dialog():
    st.markdown(UI_EN["methodology_body"])
    st.markdown(
        "\n".join(f"- {madde}" for madde in UI_EN["methodology_points"])
    )


st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --ink: #f3eee6;
            --muted: #a39e94;
            --quiet: #757068;
            --canvas: #07090c;
            --surface: #12161c;
            --surface-2: #181e26;
            --line: #343c48;
            --teal: #3d9b8f;
            --teal-deep: #1f6f67;
            --copper: #c9a66b;
            --serif: "Libre Baskerville", Georgia, serif;
            --sans: "IBM Plex Sans", "Source Sans 3", sans-serif;
        }

        #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
            display: none !important;
            visibility: hidden !important;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(ellipse 50% 32% at 8% 0%, rgba(61, 155, 143, 0.07) 0%, transparent 55%),
                linear-gradient(180deg, #0f1419 0%, var(--canvas) 45%, #06080a 100%) !important;
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 760px;
            padding-top: 2.2rem;
            padding-bottom: 3.5rem;
        }

        .masthead {
            border-bottom: none;
            padding-bottom: 0.4rem;
            margin-bottom: 1.1rem;
            animation: mast-in 0.5s ease-out both;
        }

        @keyframes mast-in {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes panel-in {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .kicker {
            display: inline-block;
            font-family: var(--sans);
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.68rem;
            font-weight: 500;
            color: var(--copper);
            background: rgba(201, 166, 107, 0.08);
            padding: 0.28rem 0.62rem;
            border: 1px solid rgba(201, 166, 107, 0.28);
            border-radius: 3px;
            margin: 0 0 1rem 0;
        }

        h1.app-title {
            font-family: var(--serif);
            font-weight: 700;
            font-size: clamp(2.55rem, 5.4vw, 3.45rem);
            line-height: 1.08;
            color: #faf6ef;
            margin: 0;
            max-width: 13ch;
        }

        .lede {
            font-family: var(--sans);
            color: var(--muted);
            font-size: 1.05rem;
            margin: 1.05rem 0 0 0;
            max-width: 34rem;
            line-height: 1.6;
        }

        .char-meter {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 1rem;
            margin: 0.15rem 0 0.85rem 0;
            font-family: var(--sans);
            font-size: 0.76rem;
            color: var(--quiet);
        }

        .char-meter .ready { color: var(--teal); }
        .char-meter .short { color: var(--copper); }

        label[data-testid="stWidgetLabel"] p,
        [data-testid="stTextArea"] label p {
            font-family: var(--sans) !important;
            color: #cfc9bf !important;
            font-weight: 500 !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.72rem !important;
        }

        textarea {
            background-color: #0c1015 !important;
            color: var(--ink) !important;
            border: 1px solid #3a4452 !important;
            border-radius: 8px !important;
            font-family: var(--serif) !important;
            font-size: 1.02rem !important;
            line-height: 1.75 !important;
            min-height: 300px !important;
        }

        textarea:focus {
            border-color: rgba(61, 155, 143, 0.65) !important;
            box-shadow: 0 0 0 1px rgba(61, 155, 143, 0.3) !important;
        }

        div.stButton > button[kind="primary"],
        div.stButton > button[data-testid="stBaseButton-primary"] {
            background: transparent !important;
            color: var(--copper) !important;
            border: 1.5px solid var(--copper) !important;
            border-radius: 5px !important;
            font-family: var(--sans) !important;
            font-weight: 600 !important;
            letter-spacing: 0.14em !important;
            text-transform: uppercase !important;
            font-size: 0.8rem !important;
            padding: 0.72rem 1.7rem !important;
            box-shadow: none !important;
            transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
        }

        div.stButton > button[kind="primary"]:hover,
        div.stButton > button[data-testid="stBaseButton-primary"]:hover {
            background: rgba(201, 166, 107, 0.14) !important;
            color: #f0dfbc !important;
            border-color: #dfc08a !important;
        }

        div.stButton > button[kind="primary"]:focus,
        div.stButton > button[data-testid="stBaseButton-primary"]:focus {
            box-shadow: 0 0 0 2px var(--canvas), 0 0 0 4px rgba(61, 155, 143, 0.45) !important;
        }

        .page-motto {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 1rem;
            margin: 1.6rem 0 0.4rem 0;
            padding: 1.1rem 0.2rem 0.2rem 0.2rem;
            border-top: 1px solid var(--line);
        }

        .page-motto .motto {
            font-family: var(--serif);
            font-style: italic;
            font-size: 1.08rem;
            color: #cfc8bc;
            margin: 0 0 0.3rem 0;
            max-width: 28rem;
            line-height: 1.45;
        }

        .page-motto .motto-attr {
            font-family: var(--sans);
            font-size: 0.74rem;
            color: var(--quiet);
            margin: 0;
        }

        .insight-panel {
            position: relative;
            margin-top: 1.55rem;
            padding: 1.45rem 1.55rem 1.5rem 1.55rem;
            border: 1px solid #323b48;
            border-radius: 8px;
            background:
                linear-gradient(180deg, #181e26 0%, #13181f 100%);
            box-shadow: none;
            animation: panel-in 0.35s ease-out both;
        }

        .insight-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
        }

        .insight-title {
            font-family: var(--serif);
            font-weight: 700;
            font-size: 1.48rem;
            color: #f2ede4;
            margin: 0;
            line-height: 1.25;
            max-width: 18ch;
        }

        .insight-sprig {
            width: 1.85rem;
            height: 1.85rem;
            color: var(--copper);
            opacity: 0.8;
            flex-shrink: 0;
            margin-top: 0.15rem;
        }

        .insight-ornament {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin: 1rem 0 1.15rem 0;
            max-width: 11rem;
        }

        .insight-ornament-line {
            flex: 1;
            height: 1px;
            background: rgba(201, 166, 107, 0.45);
        }

        .insight-ornament-dot {
            width: 0.28rem;
            height: 0.28rem;
            border-radius: 50%;
            background: var(--copper);
            opacity: 0.85;
            flex-shrink: 0;
        }

        .insight-body {
            font-family: var(--serif);
            font-size: 1.04rem;
            line-height: 1.8;
            color: #d2cbbf;
            max-width: 39rem;
        }

        .insight-body p {
            margin: 0 0 0.95rem 0;
        }

        .insight-body p:last-child {
            margin-bottom: 0;
        }

        .feedback-row {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem 1rem;
            margin: 0.85rem 0 0.2rem 0;
            padding: 0.85rem 0 0.15rem 0;
            border-top: 1px solid #2c3440;
        }

        .feedback-prompt {
            font-family: var(--sans);
            font-size: 0.78rem;
            letter-spacing: 0.06em;
            color: var(--quiet);
            margin: 0;
        }

        .result-panel {
            position: relative;
            margin-top: 1.35rem;
            padding: 1.4rem 1.4rem 1.2rem 1.4rem;
            background: #151b23;
            border: 1px solid #2f3844;
            border-radius: 8px;
            box-shadow: none;
            animation: panel-in 0.4s ease-out both;
        }

        .results-kicker {
            font-family: var(--sans);
            font-size: 0.64rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--quiet);
            margin: 0 0 1.05rem 0;
        }

        .result-top {
            display: grid;
            grid-template-columns: minmax(8.5rem, 0.9fr) 1.35fr;
            gap: 1.45rem;
            align-items: start;
        }

        @media (max-width: 720px) {
            .result-top { grid-template-columns: 1fr; }
        }

        .dominant-label {
            font-family: var(--sans);
            font-size: 0.64rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--quiet);
            margin: 0 0 0.4rem 0;
        }

        .dominant-emotion {
            font-family: var(--serif);
            font-size: clamp(1.85rem, 3.6vw, 2.35rem);
            font-style: italic;
            color: var(--teal);
            margin: 0;
            line-height: 1.12;
            letter-spacing: 0.01em;
        }

        .emotion-table {
            display: flex;
            flex-direction: column;
            gap: 0.62rem;
        }

        .emotion-head { display: none; }

        .emotion-row {
            display: grid;
            grid-template-columns: 1.15fr 2fr 3.2rem;
            align-items: center;
            column-gap: 0.7rem;
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0.1rem 0;
        }

        .emotion-rank { display: none; }

        .emotion-name {
            font-family: var(--sans);
            font-size: 0.88rem;
            font-weight: 500;
            color: #d6d0c5;
        }

        .emotion-bar-track {
            height: 0.28rem;
            background: #252c35;
            border-radius: 2px;
            overflow: hidden;
        }

        .emotion-bar-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.35s ease;
        }

        .emotion-bar-fill.rank-1 { background: var(--teal); }
        .emotion-bar-fill.rank-2 { background: #a8926f; }
        .emotion-bar-fill.rank-3 { background: #6a7380; }

        .emotion-pct {
            font-family: var(--sans);
            font-size: 0.8rem;
            font-weight: 500;
            color: #a8a296;
            text-align: right;
            font-variant-numeric: tabular-nums;
        }

        .result-foot {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 1rem;
            margin-top: 1.3rem;
            padding-top: 1rem;
            border-top: 1px solid #2c3440;
        }

        .motto {
            font-family: var(--serif);
            font-style: italic;
            font-size: 1.02rem;
            color: #c5bfb4;
            margin: 0 0 0.28rem 0;
            max-width: 30rem;
            line-height: 1.42;
        }

        .motto-attr {
            font-family: var(--sans);
            font-size: 0.72rem;
            color: var(--quiet);
            margin: 0;
            letter-spacing: 0.02em;
        }

        .panel-sprig {
            width: 2.2rem;
            height: 2.2rem;
            color: var(--copper);
            opacity: 0.75;
            flex-shrink: 0;
        }

        [data-testid="stExpander"] {
            margin-top: 1.2rem;
            border: 1px solid var(--line);
            border-radius: 10px;
            background: var(--surface);
        }

        [data-testid="stExpander"] summary {
            font-family: var(--serif);
            font-size: 1.12rem;
            color: #e4dfd6;
        }

        [data-testid="stExpander"] p {
            font-family: var(--sans);
            color: var(--muted);
            line-height: 1.7;
            font-size: 0.95rem;
        }

        [data-testid="column"]:has(.methodology-info-anchor) {
            display: flex;
            justify-content: flex-end;
            align-items: flex-start;
            padding-top: 0.2rem;
        }

        [data-testid="column"]:has(.methodology-info-anchor) [data-testid="stButton"] {
            width: auto !important;
        }

        [data-testid="column"]:has(.methodology-info-anchor) button {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            min-width: 2.7rem !important;
            width: 2.7rem !important;
            height: 2.7rem !important;
            padding: 0 !important;
            border-radius: 50% !important;
            border: 1px solid rgba(61, 155, 143, 0.45) !important;
            background: var(--surface) !important;
            box-shadow: none !important;
        }

        [data-testid="column"]:has(.methodology-info-anchor) button p {
            margin: 0 !important;
            font-family: var(--serif) !important;
            font-style: italic !important;
            font-weight: 700 !important;
            font-size: 1.35rem !important;
            color: var(--teal) !important;
        }

        [data-testid="column"]:has(.methodology-info-anchor) button:hover {
            border-color: var(--teal) !important;
            background: rgba(61, 155, 143, 0.1) !important;
        }

        [data-testid="stDialog"] h2 {
            font-family: var(--serif) !important;
            color: #f4f0e8 !important;
        }

        [data-testid="stDialog"] p,
        [data-testid="stDialog"] li {
            font-family: var(--sans) !important;
            color: var(--muted) !important;
            line-height: 1.7;
        }

        textarea::placeholder {
            color: #8a857c !important;
            opacity: 1 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

mast_col, info_col = st.columns([11, 1], vertical_alignment="top")
with mast_col:
    st.markdown(
        f"""
        <div class="masthead">
            <h1 class="app-title">{UI_EN["title"]}</h1>
            <p class="lede">{UI_EN["input_hint"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with info_col:
    st.markdown(
        '<span class="methodology-info-anchor" aria-hidden="true"></span>',
        unsafe_allow_html=True,
    )
    if st.button(
        "i",
        type="secondary",
        key="methodology_info_btn",
        help=UI_EN["methodology_title"],
        use_container_width=False,
    ):
        methodology_dialog()

metin = st.text_area(
    UI_EN["text_to_analyze"],
    height=320,
    placeholder=UI_EN["placeholder"],
)
karakter_sayisi = len(metin.strip())
derin_hazir = _metin_derin_analiz_uygun(metin)
metre_sinif = "ready" if derin_hazir else "short"
metre_durum = (
    "Ready for philosophical insight"
    if derin_hazir
    else f"Insight unlocks at ~{MIN_CHARS_DEEP}+ characters"
)
st.markdown(
    f'<div class="char-meter">'
    f'<span>{UI_EN["char_count"].format(n=karakter_sayisi)}</span>'
    f'<span class="{metre_sinif}">{metre_durum}</span>'
    f"</div>",
    unsafe_allow_html=True,
)

lang_col, btn_col = st.columns([2, 1], vertical_alignment="bottom")
with lang_col:
    analysis_language_label = st.selectbox(
        UI_EN["analysis_language"],
        options=sorted(ANALYSIS_LANGUAGE_CHOICES.keys()),
        index=sorted(ANALYSIS_LANGUAGE_CHOICES.keys()).index("English"),
    )
with btn_col:
    analiz_et = st.button(UI_EN["analyze_button"], type="primary")
results_lang = ANALYSIS_LANGUAGE_CHOICES[analysis_language_label]

st.markdown(
    f"""
    <div class="page-motto">
      <div>
        <p class="motto">{UI_EN["motto"]}</p>
        <p class="motto-attr">{UI_EN["motto_attr"]}</p>
      </div>
      <svg class="panel-sprig" viewBox="0 0 64 64" aria-hidden="true">
        <path d="M32 58 C32 40 18 32 12 18 C22 20 30 28 32 40 C34 28 42 20 52 18 C46 32 32 40 32 58 Z" fill="none" stroke="currentColor" stroke-width="1.4"/>
        <path d="M32 58 V28" fill="none" stroke="currentColor" stroke-width="1.2"/>
      </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

if analiz_et:
    if not metin.strip():
        st.warning(UI_EN["warning_empty"])
        st.session_state.pop("last_analysis", None)
    else:
        try:
            insight_metni = ""
            temiz_metin = metin.strip()
            derin_analiz = _metin_derin_analiz_uygun(temiz_metin)
            gemini_anahtar = "GEMINI_API_KEY" in st.secrets
            ust_uc: list[dict] = []
            baskın = ""
            # Deep path: Gemini is primary for emotions+insight; RoBERTa is fallback.
            if gemini_anahtar and derin_analiz:
                with st.spinner(""):
                    try:
                        gemini_uc, insight_metni = gemini_birlesik_analiz(
                            temiz_metin, results_lang
                        )
                        if gemini_uc:
                            ust_uc = gemini_uc
                            baskın = emotion_display(
                                ust_uc[0]["label"], results_lang
                            )
                    except Exception as insight_exc:
                        mesaj = _insight_hata_mesaji(insight_exc)
                        if mesaj:
                            seviye, metin_mesaj = mesaj
                            if seviye == "warning":
                                st.warning(metin_mesaj)
                            else:
                                st.error(metin_mesaj)
                        insight_metni = ""
                if not ust_uc:
                    with st.spinner(""):
                        _, ust_uc, baskın, _ = _analyze_emotions(
                            temiz_metin, results_lang
                        )
            else:
                with st.spinner(""):
                    _, ust_uc, baskın, _ = _analyze_emotions(
                        temiz_metin, results_lang
                    )

            soz, soz_kaynak = rastgele_analiz_sozu()
            st.session_state["last_analysis"] = {
                "text": temiz_metin,
                "insight": insight_metni,
                "ust_uc": ust_uc,
                "baskin": baskın,
                "results_lang": results_lang,
                "deep": derin_analiz,
                "quote": soz,
                "quote_attr": soz_kaynak,
                "feedback_vote": None,
            }
            st.components.v1.html(
                "<script>window.parent.scrollTo({top: window.parent.document.body.scrollHeight, behavior: 'smooth'});</script>",
                height=0,
            )
        except Exception as exc:
            st.session_state.pop("last_analysis", None)
            st.error(f'{UI_EN["error_prefix"]} {exc}')

sonuc = st.session_state.get("last_analysis")
if sonuc:
    ust_uc = sonuc.get("ust_uc") or []
    baskın = sonuc.get("baskin") or ""
    insight_metni = sonuc.get("insight") or ""
    results_lang_sonuc = sonuc.get("results_lang") or results_lang
    tablo = duygu_tablosu_html(ust_uc, results_lang_sonuc) if ust_uc else ""

    if not sonuc.get("deep"):
        st.info(UI_EN["warning_too_short"].format(min_chars=MIN_CHARS_DEEP))
    elif insight_metni:
        st.markdown(
            insight_paneli_html(insight_metni),
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="feedback-row">'
            f'<p class="feedback-prompt">{UI_EN["feedback_prompt"]}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
        onceki_oy = sonuc.get("feedback_vote")
        if onceki_oy:
            st.caption(UI_EN["feedback_thanks"])
        else:
            if feedback_memory.storage_mode() != "supabase":
                st.caption(UI_EN["feedback_local_only"])
            up_col, down_col, _ = st.columns([1, 1, 4])
            with up_col:
                if st.button(
                    UI_EN["feedback_up"],
                    key="feedback_up_btn",
                    use_container_width=True,
                ):
                    try:
                        feedback_memory.save_feedback(
                            vote="up",
                            text=sonuc.get("text") or "",
                            insight=insight_metni,
                            emotions=ust_uc,
                            results_lang=results_lang_sonuc,
                        )
                        st.session_state["last_analysis"]["feedback_vote"] = "up"
                        st.success(UI_EN["feedback_saved"])
                        st.rerun()
                    except Exception as fb_exc:
                        st.warning(f"Feedback could not be saved: {fb_exc}")
            with down_col:
                if st.button(
                    UI_EN["feedback_down"],
                    key="feedback_down_btn",
                    use_container_width=True,
                ):
                    try:
                        feedback_memory.save_feedback(
                            vote="down",
                            text=sonuc.get("text") or "",
                            insight=insight_metni,
                            emotions=ust_uc,
                            results_lang=results_lang_sonuc,
                        )
                        st.session_state["last_analysis"]["feedback_vote"] = "down"
                        st.success(UI_EN["feedback_saved"])
                        st.rerun()
                    except Exception as fb_exc:
                        st.warning(f"Feedback could not be saved: {fb_exc}")

    if ust_uc and tablo:
        st.markdown(
            sonuc_paneli_html(
                baskın,
                tablo,
                sonuc.get("quote") or SITE_MOTTO[0],
                sonuc.get("quote_attr") or SITE_MOTTO[1],
            ),
            unsafe_allow_html=True,
        )
