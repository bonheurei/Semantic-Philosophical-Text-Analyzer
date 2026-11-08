import json
import os
import urllib.parse
import urllib.request

import google.generativeai as genai
import streamlit as st
from langdetect import LangDetectException, detect
from deep_translator import GoogleTranslator, MyMemoryTranslator
from deep_translator.constants import MY_MEMORY_LANGUAGES_TO_CODES
from deep_translator.exceptions import LanguageNotSupportedException, TooManyRequests
from transformers import pipeline

MYMEMORY_INGILIZCE = "en-US"

ISO_MYMEMORY_TERCIH = {
    "ar": "ar-SA",
    "de": "de-DE",
    "en": MYMEMORY_INGILIZCE,
    "es": "es-ES",
    "fr": "fr-FR",
    "it": "it-IT",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "pt": "pt-PT",
    "ru": "ru-RU",
    "tr": "tr-TR",
    "zh": "zh-CN",
}

MODEL_ID = "SamLowe/roberta-base-go_emotions"
GEMINI_INSIGHT_PROMPT = (
    "Provide a brief, 2-sentence philosophical and semantic analysis of this text, "
    "focusing on its underlying tone, subtext, or metaphorical meaning."
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

UI_EN = {
    "page_title": "Semantic & Philosophical Text Analyzer",
    "kicker": "Scholarly text review",
    "title": "Semantic & Philosophical Text Analyzer",
    "lede": (
        "Your passage is translated into English for the model. Emotion names "
        "in the results match the language of your text; this interface stays in English."
    ),
    "text_to_analyze": "Text to analyze",
    "placeholder": "Paste a passage in any language…",
    "analyze_button": "Analyze Text",
    "warning_empty": "Paste a passage before running the analysis.",
    "spinner": "Translating and mapping emotional tone…",
    "error_prefix": "Analysis could not be completed:",
    "emotion": "Emotion",
    "intensity": "Intensity",
    "score": "Score",
    "dominant": "Dominant emotion",
    "result_heading": "Emotion spectrum",
    "result_note": "Transformers · {model} · deep-translator · top three labels",
    "bridge_note": "Analysis was run on the English translation of your text",
    "source_language": "Source language",
    "footer_note": (
        "First 512 tokens of the English translation were analyzed. "
        "Scores are multi-label probabilities."
    ),
    "methodology_title": "Methodology & Linguistic Limitations",
    "methodology_body": (
        "This tool uses an English bridge to analyze original texts. However, we are "
        "linguistically aware that existential and cultural emotional states unique to "
        "languages such as French may undergo structural semantic loss (semantic shift) "
        "when translated into English. This tool aims to measure the universal emotion "
        "spectrum of texts."
    ),
    "model_loading": "Loading Go Emotions model…",
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


@st.cache_resource(show_spinner=UI_EN["model_loading"])
def load_emotion_classifier():
    return pipeline(
        "text-classification",
        model=MODEL_ID,
        top_k=None,
    )


def gemini_philosophical_insight(metin: str) -> str:
    genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
    model = genai.GenerativeModel('gemini-3.8-flash')
    prompt = f"{GEMINI_INSIGHT_PROMPT}\n\nText:\n{metin.strip()}"
    yanit = model.generate_content(prompt)
    if not yanit.text:
        raise RuntimeError("Gemini returned empty insight text.")
    return yanit.text.strip()


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


def emotion_display(label: str, source_lang: str) -> str:
    key = label.strip().lower()
    kok = (source_lang or "en").lower().split("-")[0]
    sozluk = EMOTION_LABELS_BY_LANG.get(kok, EMOTION_LABELS_EN)
    return sozluk.get(key, EMOTION_LABELS_EN.get(key, label.replace("_", " ").title()))


def ust_uc_duzenle(skorlar: list[dict]) -> list[dict]:
    sirali = sorted(skorlar, key=lambda x: x["score"], reverse=True)
    return sirali[:3]


def duygu_tablosu_html(ust_uc: list[dict], source_lang: str) -> str:
    satirlar = []
    for i, kayit in enumerate(ust_uc, start=1):
        ad = emotion_display(kayit["label"], source_lang)
        yuzde = kayit["score"] * 100
        genislik = min(100.0, max(0.0, yuzde))
        satirlar.append(
            f"""
            <div class="emotion-row">
                <span class="emotion-rank">{i}</span>
                <span class="emotion-name">{ad}</span>
                <div class="emotion-bar-track" aria-hidden="true">
                    <div class="emotion-bar-fill rank-{i}" style="width: {genislik:.1f}%;"></div>
                </div>
                <span class="emotion-pct">%{yuzde:.1f}</span>
            </div>
            """
        )
    return (
        '<div class="emotion-table">'
        '<div class="emotion-head">'
        f'<span></span><span>{UI_EN["emotion"]}</span>'
        f'<span>{UI_EN["intensity"]}</span><span>{UI_EN["score"]}</span>'
        "</div>"
        + "".join(satirlar)
        + "</div>"
    )


st.set_page_config(
    page_title="Semantic & Philosophical Text Analyzer",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="collapsed",
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

        .result-note {
            font-family: "Source Sans 3", sans-serif;
            color: #8f8a82;
            font-size: 0.86rem;
            margin: 0 0 1.1rem 0;
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="masthead">
        <p class="kicker">{UI_EN["kicker"]}</p>
        <h1 class="app-title">{UI_EN["title"]}</h1>
        <p class="lede">{UI_EN["lede"]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metin = st.text_area(
    UI_EN["text_to_analyze"],
    height=320,
    placeholder=UI_EN["placeholder"],
)

analiz_et = st.button(UI_EN["analyze_button"], type="primary")

if analiz_et:
    if not metin.strip():
        st.warning(UI_EN["warning_empty"])
    else:
        try:
            with st.spinner(UI_EN["spinner"]):
                source_lang = detect_source_lang(metin.strip())
                metin_en = metni_ingilizceye_cevir(metin.strip(), source_lang)

                classifier = load_emotion_classifier()
                skorlar = classifier(
                    metin_en,
                    truncation=True,
                    max_length=512,
                )[0]
                ust_uc = ust_uc_duzenle(skorlar)
                baskın = emotion_display(ust_uc[0]["label"], source_lang)

                insight_metni = ""
                try:
                    insight_en = gemini_philosophical_insight(metin.strip())
                    insight_metni = insight_in_source_language(insight_en, source_lang)
                except Exception as e:
                    st.error(f"Hata Detayı: {e}")

            with st.expander("Semantic & Philosophical Insight", expanded=True):
                if insight_metni:
                    st.markdown(insight_metni)

            tablo = duygu_tablosu_html(ust_uc, source_lang)
            cevirildi = not source_lang.startswith("en")
            pipe_notu = (
                f'{UI_EN["bridge_note"]} · {UI_EN["source_language"]}: {source_lang.upper()}'
                if cevirildi
                else f'{UI_EN["source_language"]}: {source_lang.upper()}'
            )

            st.markdown(
                f"""
                <div class="result-panel">
                    <p class="result-heading">{UI_EN["result_heading"]}</p>
                    <p class="result-note">{UI_EN["result_note"].format(model=MODEL_ID)}</p>
                    <p class="pipeline-note">{pipe_notu}</p>
                    <p class="dominant-emotion">{UI_EN["dominant"]}: {baskın}</p>
                    {tablo}
                    <p class="result-note" style="margin-top: 1rem; margin-bottom: 0;">
                        {UI_EN["footer_note"]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f'{UI_EN["error_prefix"]} {exc}')

with st.expander(UI_EN["methodology_title"]):
    st.markdown(UI_EN["methodology_body"])
