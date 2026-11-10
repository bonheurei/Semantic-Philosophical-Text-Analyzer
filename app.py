import concurrent.futures
import json
import re
import urllib.parse
import urllib.request

from google import genai
import streamlit as st
from langdetect import LangDetectException, detect
from deep_translator import GoogleTranslator, MyMemoryTranslator
from deep_translator.constants import MY_MEMORY_LANGUAGES_TO_CODES
from deep_translator.exceptions import LanguageNotSupportedException, TooManyRequests
from transformers import pipeline

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
ROBERTA_CHUNK_WORDS = 320
ROBERTA_CHUNK_OVERLAP = 60
ROBERTA_MAX_CHUNKS = 4
PHATIC_MAX_WORDS = 12
SALIENT_EMOTION_MIN = 0.22
LITERARY_MIN_WORDS = 18
EDISI_DUYGU_INKAR = re.compile(
    r"(hiçbir şey hisset|hissetmiyor|hissetmedim|duygum yok|duygu yok|"
    r"felt nothing|don't feel|didn't feel|do not feel|no feeling|numb|detached)",
    re.IGNORECASE,
)
EDISI_BEKLENEN_DUYGU = re.compile(
    r"(sanki .{0,48}(hisset|üzül|gibi)|should have felt|ought to feel|"
    r"as if i (should|ought)|üzülmeliymiş)",
    re.IGNORECASE,
)
EDISI_KARSIT = re.compile(
    r"\b(ama|fakat|ancak|lakin|oysa|but|yet|however|although|though)\b",
    re.IGNORECASE,
)
EDISI_BENZETME = re.compile(
    r"(sanki|gibi bir|metafor|metaphor|\bas if\b|\blike a\b)",
    re.IGNORECASE,
)
GEMINI_INSIGHT_PROMPT = (
    "Provide a brief, 2-sentence philosophical and semantic analysis of this text, "
    "focusing on its underlying tone, subtext, or metaphorical meaning."
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
GEMINI_INSIGHT_REQUEST_TIMEOUT_SEC = 60
GEMINI_INSIGHT_TOTAL_TIMEOUT_SEC = 120
GEMINI_INSIGHT_MAX_TEXT_CHARS = 2400
GEMINI_INSIGHT_MODEL_PREFERENCES = (
    "gemini-3.8-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
)
GEMINI_CLIENT_API_VERSIONS = ("v1beta", "v1", None)
GEMINI_CLOUD_MODELS = (
    "gemini-3.8-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
)
GEMINI_CLOUD_API_VERSIONS = ("v1beta", "v1")
GEMINI_CLOUD_REQUEST_TIMEOUT_SEC = 45
GEMINI_CLOUD_TOTAL_TIMEOUT_SEC = 75
GEMINI_CLOUD_MAX_TEXT_CHARS = 1200


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


def _gemini_insight_model_ids(client, profile: dict) -> list[str]:
    if not profile["discover_models"]:
        return list(profile["model_preferences"])

    preferences = profile["model_preferences"]
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
        pass

    if discovered:
        ordered: list[str] = []
        for preferred in preferences:
            if preferred in discovered and preferred not in ordered:
                ordered.append(preferred)
        for model_id in discovered:
            if model_id not in ordered:
                ordered.append(model_id)
        return ordered
    return list(preferences)


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

UI_EN = {
    "page_title": "Semantic & Philosophical Text Analyzer",
    "kicker": "Scholarly text review",
    "title": "Semantic & Philosophical Text Analyzer",
    "info_title": "About this analyzer",
    "text_to_analyze": "Text to analyze",
    "analysis_language": "Analysis language",
    "results_language": "Results language",
    "placeholder": "Paste a passage in any language…",
    "analyze_button": "Analyze Text",
    "warning_empty": "Paste a passage before running the analysis.",
    "spinner": "Translating and mapping emotional tone…",
    "status_detect": "Detecting source language…",
    "status_translate": "Translating to English for the model…",
    "status_model": "Loading Go Emotions model (first run can take 1–2 minutes on the server)…",
    "status_score": "Scoring emotional tone…",
    "status_emotions_done": "Emotion analysis complete",
    "spinner_insight": "Generating philosophical insight (Gemini)…",
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
        "For scoring, your text may be translated into English; results can "
        "reflect that bridge as much as the original voice.",
        "Metaphor, tone, and wordplay often flatten in translation, which can "
        "skew or dull emotional signal.",
        "Only a limited span of the bridge is scored at once; very long "
        "passages are read in several averaged windows.",
        "Bar lengths show relative share among the top three labels, not "
        "independent probabilities that sum to 100%.",
        "Where irony, negated feeling, or layered subtext is clear, analysis "
        "may lean on the original wording more than on a literal bridge.",
    ),
    "model_loading": "Loading emotion classifier…",
    "insight_title": "Semantic & Philosophical Insight",
    "insight_unavailable": "Philosophical insight is temporarily unavailable.",
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

    google_hata: str | None = None
    try:
        sonuc = GoogleTranslator(source=google_src, target=google_tgt).translate(temiz)
        if sonuc and sonuc.strip():
            return sonuc.strip()
    except Exception as exc:
        google_hata = str(exc)

    mymemory_hata: str | None = None
    try:
        sonuc = mymemory_translate(temiz, kaynak, hedef)
        if sonuc:
            return sonuc
    except Exception as exc:
        mymemory_hata = str(exc)

    gtx_hata: str | None = None
    try:
        sonuc = gtx_translate(temiz, kaynak, hedef)
        if sonuc:
            return sonuc
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
    profile = _gemini_insight_profile()

    def _generate() -> str:
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

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_generate).result(
                timeout=profile["total_timeout_sec"]
            )
    except concurrent.futures.TimeoutError as exc:
        raise GeminiConnectionTimeout from exc


def gemini_philosophical_insight(metin: str, output_lang: str) -> str:
    temiz = metin.strip()
    if not temiz:
        return ""
    profile = _gemini_insight_profile()
    metin_ozet = temiz[: profile["max_text_chars"]]
    lang_name = gemini_output_language_name(output_lang)
    prompt = (
        f"{GEMINI_INSIGHT_PROMPT}\n"
        f"Please provide the entire analysis and response strictly in {lang_name}.\n\n"
        f"Text:\n{metin_ozet}"
    )
    return _gemini_generate_text(prompt)


def _katmanli_edebi_metin(metin: str) -> bool:
    temiz = metin.strip()
    if not temiz:
        return False
    kelime = len(temiz.split())
    if kelime < 10:
        return False
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


def gemini_edebi_birlesik(
    metin: str, output_lang: str
) -> tuple[list[dict], str]:
    temiz = metin.strip()
    if not temiz:
        return [], ""
    profile = _gemini_insight_profile()
    metin_ozet = temiz[: profile["max_text_chars"]]
    lang_name = gemini_output_language_name(output_lang)
    etiketler = ", ".join(sorted(EMOTION_LABELS_EN.keys()))
    prompt = (
        "Analyze the following text in its original language—including "
        "literature, poetry, irony, and philosophical prose.\n\n"
        f"{GEMINI_INSIGHT_PROMPT}\n"
        f"Write the insight field strictly in {lang_name}.\n\n"
        "For emotions, score the narrator's lived tone and subtext, not plot "
        "keywords alone. When the speaker reports numbness, estrangement, or "
        "not feeling expected grief, prefer neutral, realization, confusion, "
        "or disappointment over sadness/grief driven only by events.\n\n"
        "Return exactly three salient Go Emotions labels with confidence "
        f"scores between 0 and 1 (independent). Allowed labels: {etiketler}\n\n"
        "Respond with ONLY valid JSON (no markdown):\n"
        '{"emotions": [{"label": "realization", "score": 0.55}, '
        '{"label": "neutral", "score": 0.30}, {"label": "confusion", "score": 0.15}], '
        f'"insight": "..."}}\n\n'
        f"Text:\n{metin_ozet}"
    )
    ham = _gemini_generate_text(prompt)
    duygular, insight = _parse_gemini_analysis_json(ham)
    return ust_uc_gemini(duygular), insight


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


def ust_uc_duzenle(skorlar: list[dict], metin: str = "") -> list[dict]:
    """Pick top-3 labels; normalize bars; neutral only leads on short phatic text."""
    sirali = sorted(skorlar, key=lambda x: x["score"], reverse=True)
    if not sirali:
        return []
    kelime = len(metin.split())
    not_notr = [
        s for s in sirali if s["label"].strip().lower() != "neutral"
    ]
    en_yuksek_notr = not_notr[0]["score"] if not_notr else 0.0
    neutral = next(
        (s for s in sirali if s["label"].strip().lower() == "neutral"), None
    )

    if (
        kelime <= PHATIC_MAX_WORDS
        and neutral
        and sirali[0]["label"].strip().lower() == "neutral"
    ):
        ikinci = sirali[1]["score"] if len(sirali) > 1 else 0.0
        if ikinci < 0.40 or neutral["score"] - ikinci >= 0.15:
            return _gosterim_uctus_normallestir(sirali)

    if kelime > PHATIC_MAX_WORDS and en_yuksek_notr >= SALIENT_EMOTION_MIN:
        if len(not_notr) >= 3:
            return _gosterim_uctus_normallestir(not_notr)
        secilen: list[dict] = []
        for kayit in not_notr + sirali:
            if kayit in secilen:
                continue
            secilen.append(kayit)
            if len(secilen) == 3:
                break
        return _gosterim_uctus_normallestir(secilen)

    if en_yuksek_notr < 0.40:
        return _gosterim_uctus_normallestir(sirali)

    if len(not_notr) >= 3:
        return _gosterim_uctus_normallestir(not_notr)
    secilen = []
    for kayit in not_notr + sirali:
        if kayit in secilen:
            continue
        secilen.append(kayit)
        if len(secilen) == 3:
            break
    return _gosterim_uctus_normallestir(secilen)


def duygu_tablosu_html(ust_uc: list[dict], source_lang: str) -> str:
    """Single-line HTML so Streamlit markdown does not break the block."""
    satirlar: list[str] = []
    for i, kayit in enumerate(ust_uc, start=1):
        ad = emotion_display(kayit["label"], source_lang)
        yuzde = kayit["score"] * 100
        genislik = min(100.0, max(0.0, yuzde))
        satirlar.append(
            f'<div class="emotion-row">'
            f'<span class="emotion-rank">{i}</span>'
            f'<span class="emotion-name">{ad}</span>'
            f'<div class="emotion-bar-track" aria-hidden="true">'
            f'<div class="emotion-bar-fill rank-{i}" style="width: {genislik:.1f}%;"></div>'
            f"</div>"
            f'<span class="emotion-pct">{yuzde:.1f}%</span>'
            f"</div>"
        )
    return (
        '<div class="emotion-table">'
        '<div class="emotion-head">'
        f"<span></span><span>{UI_EN['emotion']}</span>"
        f"<span>{UI_EN['intensity']}</span><span>{UI_EN['score']}</span>"
        "</div>"
        + "".join(satirlar)
        + "</div>"
    )


def sonuc_paneli_html(baskın: str, tablo: str) -> str:
    return (
        f'<div class="result-panel">'
        f'<p class="result-heading">{UI_EN["result_heading"]}</p>'
        f'<p class="dominant-emotion">{UI_EN["dominant"]}: {baskın}</p>'
        f"{tablo}"
        f"</div>"
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
    biriken: dict[str, list[float]] = {}
    for skorlar in parcalar_skor:
        for kayit in skorlar:
            etiket = kayit["label"]
            biriken.setdefault(etiket, []).append(float(kayit["score"]))
    birlesik = [
        {"label": etiket, "score": sum(degerler) / len(degerler)}
        for etiket, degerler in biriken.items()
    ]
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
            "felsefi özet gelmedi. Daha kısa metin deneyin veya biraz "
            "sonra tekrar Analyze Text’e basın.",
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
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Source+Sans+3:wght@400;500;600&display=swap');

        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #0f1216;
            color: #e8e4dc;
        }

        [data-testid="stHeader"] {
            background: rgba(15, 18, 22, 0.85);
        }

        .block-container {
            max-width: 920px;
            padding-top: 3.2rem;
            padding-bottom: 4rem;
        }

        .masthead {
            border-bottom: 1px solid #2c313a;
            padding-bottom: 1.4rem;
            margin-bottom: 1.8rem;
        }

        .kicker {
            font-family: "Source Sans 3", sans-serif;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            font-size: 0.72rem;
            color: #b9a27a;
            margin: 0 0 0.55rem 0;
        }

        h1.app-title {
            font-family: "Cormorant Garamond", Georgia, serif;
            font-weight: 600;
            font-size: 2.7rem;
            line-height: 1.15;
            color: #f4f0e8;
            margin: 0;
        }

        .lede {
            font-family: "Source Sans 3", sans-serif;
            color: #a8a39a;
            font-size: 1.02rem;
            margin: 0.75rem 0 0 0;
            max-width: 48rem;
        }

        label[data-testid="stWidgetLabel"] p,
        [data-testid="stTextArea"] label p {
            font-family: "Source Sans 3", sans-serif !important;
            color: #d8d3c8 !important;
            font-weight: 500 !important;
            letter-spacing: 0.01em;
        }

        textarea {
            background-color: #171b21 !important;
            color: #efeae2 !important;
            border: 1px solid #323844 !important;
            border-radius: 8px !important;
            font-family: "Source Sans 3", sans-serif !important;
            font-size: 1rem !important;
            line-height: 1.65 !important;
        }

        textarea:focus {
            border-color: #b9a27a !important;
            box-shadow: 0 0 0 1px #b9a27a !important;
        }

        div.stButton > button {
            background: #c4ae86;
            color: #1a160f;
            border: none;
            border-radius: 6px;
            font-family: "Source Sans 3", sans-serif;
            font-weight: 600;
            letter-spacing: 0.04em;
            padding: 0.55rem 1.35rem;
            transition: background 0.15s ease;
        }

        div.stButton > button:hover {
            background: #d4c19a;
            color: #1a160f;
            border: none;
        }

        div.stButton > button:focus {
            box-shadow: 0 0 0 2px #0f1216, 0 0 0 4px #b9a27a;
        }

        .result-panel {
            margin-top: 1.6rem;
            background: linear-gradient(165deg, #181c23 0%, #14181e 100%);
            border: 1px solid #2c313a;
            border-radius: 12px;
            padding: 1.5rem 1.5rem 1.35rem 1.5rem;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
        }

        .result-heading {
            font-family: "Cormorant Garamond", Georgia, serif;
            font-size: 1.55rem;
            color: #f4f0e8;
            margin: 0 0 0.15rem 0;
        }

        .dominant-emotion {
            font-family: "Cormorant Garamond", Georgia, serif;
            font-size: 2.05rem;
            color: #c4ae86;
            margin: 0 0 1.25rem 0;
            letter-spacing: 0.02em;
        }

        .emotion-table {
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
        }

        .emotion-head,
        .emotion-row {
            display: grid;
            grid-template-columns: 2.2rem 1fr 2.2fr 4.5rem;
            align-items: center;
            column-gap: 0.85rem;
        }

        .emotion-head {
            font-family: "Source Sans 3", sans-serif;
            font-size: 0.68rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #7a756e;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid #2a2f38;
            margin-bottom: 0.15rem;
        }

        .emotion-row {
            background: rgba(12, 14, 18, 0.55);
            border: 1px solid #2a3039;
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
        }

        .emotion-rank {
            font-family: "Cormorant Garamond", Georgia, serif;
            font-size: 1.35rem;
            color: #b9a27a;
            text-align: center;
        }

        .emotion-name {
            font-family: "Source Sans 3", sans-serif;
            font-size: 1.02rem;
            font-weight: 500;
            color: #ebe6dd;
        }

        .emotion-bar-track {
            height: 0.45rem;
            background: #252a33;
            border-radius: 999px;
            overflow: hidden;
        }

        .emotion-bar-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.35s ease;
        }

        .emotion-bar-fill.rank-1 { background: linear-gradient(90deg, #c4ae86, #e0cba0); }
        .emotion-bar-fill.rank-2 { background: linear-gradient(90deg, #8f9aa8, #b5c0cc); }
        .emotion-bar-fill.rank-3 { background: linear-gradient(90deg, #6b7a8c, #8a97a8); }

        .emotion-pct {
            font-family: "Source Sans 3", sans-serif;
            font-size: 0.95rem;
            font-weight: 600;
            color: #d4cfc6;
            text-align: right;
        }

        .pipeline-note {
            font-family: "Source Sans 3", sans-serif;
            color: #9a958d;
            font-size: 0.82rem;
            margin: 0 0 0.75rem 0;
            padding: 0.55rem 0.65rem;
            border-left: 2px solid #5c5345;
            background: rgba(10, 12, 15, 0.45);
            border-radius: 0 6px 6px 0;
        }

        [data-testid="stExpander"],
        [data-testid="stExpander"].insight-expander {
            margin-top: 1.25rem;
            border: 1px solid #2c313a;
            border-radius: 10px;
            background: #14181e;
        }

        [data-testid="stExpander"].insight-expander {
            margin-bottom: 0.5rem;
            border-color: #3d4a3a;
            background: linear-gradient(165deg, #1a2218 0%, #14181e 100%);
        }

        [data-testid="stExpander"] summary {
            font-family: "Cormorant Garamond", Georgia, serif;
            font-size: 1.15rem;
            color: #d8d3c8;
        }

        [data-testid="stExpander"] p {
            font-family: "Source Sans 3", sans-serif;
            color: #a8a39a;
            line-height: 1.7;
            font-size: 0.95rem;
        }

        [data-testid="column"]:has(.methodology-info-anchor) {
            display: flex;
            justify-content: flex-end;
            align-items: flex-start;
            padding-top: 0.35rem;
        }

        [data-testid="column"]:has(.methodology-info-anchor) [data-testid="stButton"] {
            width: auto !important;
        }

        [data-testid="column"]:has(.methodology-info-anchor) button {
            position: relative;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-sizing: border-box !important;
            min-width: 3rem !important;
            width: 3rem !important;
            height: 3rem !important;
            max-width: 3rem !important;
            max-height: 3rem !important;
            padding: 0 !important;
            margin: 0 !important;
            border-radius: 50% !important;
            border: 1px solid rgba(185, 162, 122, 0.45) !important;
            background:
                radial-gradient(
                    circle at 32% 28%,
                    rgba(196, 174, 134, 0.18) 0%,
                    transparent 56%
                ),
                linear-gradient(158deg, #262d37 0%, #14181e 100%) !important;
            box-shadow:
                0 8px 22px rgba(0, 0, 0, 0.32),
                inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
            transition:
                border-color 0.2s ease,
                transform 0.2s ease,
                box-shadow 0.2s ease,
                background 0.2s ease;
        }

        [data-testid="column"]:has(.methodology-info-anchor) button::before {
            content: "";
            position: absolute;
            inset: 7px;
            border-radius: 50%;
            border: 1px solid rgba(196, 174, 134, 0.22);
            pointer-events: none;
        }

        [data-testid="column"]:has(.methodology-info-anchor) button p {
            margin: 0 !important;
            padding: 0 !important;
            font-family: "Cormorant Garamond", Georgia, serif !important;
            font-style: italic !important;
            font-weight: 600 !important;
            font-size: 1.55rem !important;
            line-height: 1 !important;
            color: #c9b38a !important;
            transform: translateY(-1px);
        }

        [data-testid="column"]:has(.methodology-info-anchor) button:hover {
            border-color: rgba(216, 193, 148, 0.8) !important;
            transform: translateY(-2px);
            box-shadow:
                0 12px 28px rgba(0, 0, 0, 0.38),
                0 0 0 1px rgba(196, 174, 134, 0.14),
                inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        }

        [data-testid="column"]:has(.methodology-info-anchor) button:hover p {
            color: #ede3cf !important;
        }

        [data-testid="column"]:has(.methodology-info-anchor) button:focus {
            outline: none !important;
            box-shadow:
                0 0 0 2px #0f1216,
                0 0 0 4px rgba(185, 162, 122, 0.55) !important;
        }

        [data-testid="stDialog"] h2 {
            font-family: "Cormorant Garamond", Georgia, serif !important;
            color: #f4f0e8 !important;
            font-weight: 600 !important;
        }

        [data-testid="stDialog"] p,
        [data-testid="stDialog"] li {
            font-family: "Source Sans 3", sans-serif !important;
            color: #a8a39a !important;
            line-height: 1.7;
        }

        [data-testid="stDialog"] ul {
            padding-left: 1.15rem;
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
            <p class="kicker">{UI_EN["kicker"]}</p>
            <h1 class="app-title">{UI_EN["title"]}</h1>
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

st.markdown(
    '<style> textarea::placeholder { color: #b0b0b0 !important; opacity: 1 !important; } </style>',
    unsafe_allow_html=True,
)

metin = st.text_area(
    UI_EN["text_to_analyze"],
    height=320,
    placeholder=UI_EN["placeholder"],
)

analysis_language_label = st.selectbox(
    UI_EN["analysis_language"],
    options=sorted(ANALYSIS_LANGUAGE_CHOICES.keys()),
    index=sorted(ANALYSIS_LANGUAGE_CHOICES.keys()).index("English"),
)
results_lang = ANALYSIS_LANGUAGE_CHOICES[analysis_language_label]

analiz_et = st.button(UI_EN["analyze_button"], type="primary")

if analiz_et:
    if not metin.strip():
        st.warning(UI_EN["warning_empty"])
    else:
        try:
            insight_metni = ""
            temiz_metin = metin.strip()
            edebi_metin = _katmanli_edebi_metin(temiz_metin)
            gemini_anahtar = "GEMINI_API_KEY" in st.secrets
            edebi_gemini = edebi_metin and gemini_anahtar
            with st.spinner(""):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    fut_roberta = pool.submit(
                        _analyze_emotions, temiz_metin, results_lang
                    )
                    fut_gemini = None
                    if edebi_gemini:
                        fut_gemini = pool.submit(
                            gemini_edebi_birlesik, temiz_metin, results_lang
                        )
                    elif gemini_anahtar:
                        fut_gemini = pool.submit(
                            _fetch_insight, temiz_metin, results_lang
                        )
                    _, ust_uc, baskın, _ = fut_roberta.result()
                    if edebi_gemini and fut_gemini is not None:
                        try:
                            ust_uc, insight_metni = fut_gemini.result()
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
                    elif fut_gemini is not None:
                        try:
                            insight_metni = fut_gemini.result()
                        except Exception as insight_exc:
                            mesaj = _insight_hata_mesaji(insight_exc)
                            if mesaj:
                                seviye, metin_mesaj = mesaj
                                if seviye == "warning":
                                    st.warning(metin_mesaj)
                                else:
                                    st.error(metin_mesaj)
                            insight_metni = ""
                    else:
                        insight_metni = ""

            tablo = duygu_tablosu_html(ust_uc, results_lang)

            with st.expander(UI_EN["insight_title"], expanded=True):
                if insight_metni:
                    st.markdown(insight_metni)

            st.markdown(
                sonuc_paneli_html(baskın, tablo),
                unsafe_allow_html=True,
            )

            st.components.v1.html(
                "<script>window.parent.scrollTo({top: window.parent.document.body.scrollHeight, behavior: 'smooth'});</script>",
                height=0,
            )
        except Exception as exc:
            st.error(f'{UI_EN["error_prefix"]} {exc}')
