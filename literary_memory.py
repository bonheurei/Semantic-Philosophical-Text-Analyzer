"""Literary reference memory for Gemini prompting.

Multilingual retrieval (no vector DB yet):
1) Expand query tokens via a compact multilingual theme lexicon.
2) Bridge to English when possible (Google → MyMemory), then score.
3) Fall back to curated method anchors if overlap stays weak.

Callers may pass match_text= (already-English bridge) to avoid a second
translation when the app already translated for RoBERTa.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "literary_refs.json"
MAX_EXAMPLES = 2
BRIDGE_CHARS = 1200
MIN_ACCEPT_SCORE = 0.08

FALLBACK_IDS = (
    "dostoevsky-underground",
    "kafka-law",
    "chekhov-unsaid",
    "seneca-disturbance",
)

# Maps surface forms (many languages) → English literary theme tokens.
LEXICON_TO_THEMES: dict[str, set[str]] = {
    # death / mourning
    "death": {"death", "mortality"},
    "dead": {"death", "mortality"},
    "die": {"death", "mortality"},
    "dying": {"death", "mortality"},
    "mortality": {"death", "mortality"},
    "funeral": {"death", "mortality", "grief"},
    "grief": {"grief", "death"},
    "mourning": {"grief", "death"},
    "ölüm": {"death", "mortality"},
    "olum": {"death", "mortality"},
    "ölmüş": {"death", "mortality"},
    "olmus": {"death", "mortality"},
    "öldü": {"death", "mortality"},
    "oldu": {"death", "mortality"},
    "cenaze": {"death", "mortality", "grief"},
    "yas": {"grief", "death"},
    "mort": {"death", "mortality"},
    "muerte": {"death", "mortality"},
    "tod": {"death", "mortality"},
    "morte": {"death", "mortality"},
    "смерть": {"death", "mortality"},
    "死": {"death", "mortality"},
    "死亡": {"death", "mortality"},
    # mother / family
    "mother": {"mother", "family"},
    "maternal": {"mother", "family"},
    "anne": {"mother", "family"},
    "annesi": {"mother", "family"},
    "annem": {"mother", "family"},
    "mère": {"mother", "family"},
    "mere": {"mother", "family"},
    "madre": {"mother", "family"},
    "mutter": {"mother", "family"},
    # alienation / indifference
    "alienation": {"alienation", "estrangement"},
    "estrangement": {"alienation", "estrangement"},
    "stranger": {"alienation", "estrangement", "outsider"},
    "outsider": {"alienation", "estrangement"},
    "indifference": {"indifference", "apathy", "numbness"},
    "indifferent": {"indifference", "apathy"},
    "apathy": {"indifference", "apathy"},
    "numb": {"numbness", "indifference"},
    "numbness": {"numbness", "indifference"},
    "detached": {"detachment", "alienation"},
    "yabancı": {"alienation", "estrangement", "stranger"},
    "yabanci": {"alienation", "estrangement", "stranger"},
    "yabancılık": {"alienation", "estrangement"},
    "umursamıyor": {"indifference", "apathy"},
    "umursamiyor": {"indifference", "apathy"},
    "umursamaz": {"indifference", "apathy"},
    "kayıtsız": {"indifference", "apathy"},
    "kayitsiz": {"indifference", "apathy"},
    "duygusuz": {"numbness", "indifference"},
    "duygusuzluk": {"numbness", "indifference"},
    "étranger": {"alienation", "estrangement", "stranger"},
    "etranger": {"alienation", "estrangement", "stranger"},
    "indifférence": {"indifference", "apathy"},
    "indifference": {"indifference", "apathy"},
    "extranjero": {"alienation", "estrangement"},
    "alienacion": {"alienation", "estrangement"},
    "alienación": {"alienation", "estrangement"},
    "indiferencia": {"indifference", "apathy"},
    "fremd": {"alienation", "estrangement"},
    "чужой": {"alienation", "estrangement"},
    "равнодушие": {"indifference", "apathy"},
    # freedom / control
    "freedom": {"freedom", "agency"},
    "free": {"freedom", "agency"},
    "liberty": {"freedom", "agency"},
    "özgürlük": {"freedom", "agency"},
    "ozgurluk": {"freedom", "agency"},
    "liberté": {"freedom", "agency"},
    "libertad": {"freedom", "agency"},
    "control": {"control", "agency"},
    "kontrol": {"control", "agency"},
    # guilt / conscience
    "guilt": {"guilt", "conscience"},
    "shame": {"guilt", "shame"},
    "conscience": {"conscience", "guilt"},
    "suçluluk": {"guilt", "conscience"},
    "sucluluk": {"guilt", "conscience"},
    "vicdan": {"conscience", "guilt"},
    # love / desire
    "love": {"love", "desire"},
    "aşk": {"love", "desire"},
    "ask": {"love", "desire"},
    "amour": {"love", "desire"},
    "amor": {"love", "desire"},
    # fear / absurd / void
    "fear": {"fear", "terror"},
    "korku": {"fear", "terror"},
    "peur": {"fear", "terror"},
    "absurd": {"absurd", "void"},
    "absurde": {"absurd", "void"},
    "anlamsız": {"absurd", "void", "meaningless"},
    "anlamsiz": {"absurd", "void", "meaningless"},
    "void": {"void", "absurd"},
    # waiting / bureaucracy / power
    "waiting": {"waiting", "bureaucracy"},
    "wait": {"waiting", "bureaucracy"},
    "beklemek": {"waiting", "bureaucracy"},
    "bekleyiş": {"waiting", "bureaucracy"},
    "law": {"authority", "bureaucracy"},
    "yasa": {"authority", "bureaucracy"},
    "power": {"power", "authority"},
    "güç": {"power", "authority"},
    "guc": {"power", "authority"},
    # self / truth / time / nature
    "self": {"self", "identity", "consciousness"},
    "benlik": {"self", "identity"},
    "truth": {"truth", "reality", "illusion"},
    "hakikat": {"truth", "reality"},
    "time": {"time", "impermanence"},
    "zaman": {"time", "impermanence"},
    "nature": {"nature"},
    "doğa": {"nature"},
    "doga": {"nature"},
    "peace": {"tranquility", "disturbance"},
    "huzur": {"tranquility", "disturbance"},
    "disturbance": {"disturbance", "tranquility"},
    "wealth": {"wealth", "tranquility"},
}

_TOKEN = re.compile(
    r"[a-zA-ZçğıöşüÇĞİÖŞÜâîûÂÎÛа-яА-ЯёЁ\u3040-\u30ff\u3400-\u9fff']+",
    re.UNICODE,
)


@lru_cache(maxsize=1)
def load_refs() -> tuple[dict, ...]:
    if not DATA_PATH.is_file():
        return tuple()
    with DATA_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries") or []
    temiz: list[dict] = []
    for kayit in entries:
        if not isinstance(kayit, dict):
            continue
        if not (kayit.get("analysis") or "").strip():
            continue
        temiz.append(kayit)
    return tuple(temiz)


def _tokenize(metin: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(metin or "") if len(t) > 2}


def _lexicon_themes(tokens: set[str]) -> set[str]:
    themes: set[str] = set()
    for tok in tokens:
        hit = LEXICON_TO_THEMES.get(tok)
        if hit:
            themes.update(hit)
        # Soft prefix for Turkish verb forms: umursamiyor → umursam
        if len(tok) >= 6:
            for n in (6, 7, 8):
                hit = LEXICON_TO_THEMES.get(tok[:n])
                if hit:
                    themes.update(hit)
    return themes


def _query_token_set(metin: str, match_text: str | None = None) -> set[str]:
    ham = _tokenize(metin)
    if match_text:
        ham |= _tokenize(match_text)
    else:
        bridged = _bridge_to_english(metin)
        if bridged:
            ham |= _tokenize(bridged)
    return ham | _lexicon_themes(ham)


def _translate_once(parca: str, source: str) -> str:
    from deep_translator import GoogleTranslator, MyMemoryTranslator

    try:
        if source == "auto":
            out = GoogleTranslator(source="auto", target="en").translate(parca)
        else:
            out = GoogleTranslator(source=source, target="en").translate(parca)
        if out and out.strip() and out.strip() != parca.strip():
            return out.strip()
    except Exception:
        pass
    try:
        out = MyMemoryTranslator(source=source if source != "auto" else "tr", target="en-US").translate(parca)
        if out and out.strip():
            return out.strip()
    except Exception:
        pass
    return ""


@lru_cache(maxsize=64)
def _bridge_to_english(metin: str) -> str:
    """Best-effort English bridge for matching; safe no-op on failure."""
    temiz = (metin or "").strip()
    if not temiz:
        return ""
    parca = temiz[:BRIDGE_CHARS]
    try:
        from langdetect import detect

        dil = detect(parca).lower().split("-")[0]
    except Exception:
        dil = "en"
    if dil == "en":
        return parca
    ceviri = _translate_once(parca, "auto") or _translate_once(parca, dil)
    return ceviri or parca


def _entry_bags(entry: dict) -> tuple[set[str], set[str], set[str]]:
    themes: set[str] = set()
    for parca in entry.get("themes") or []:
        themes.update(_tokenize(str(parca).replace("_", " ")))
    keywords: set[str] = set()
    for parca in entry.get("keywords") or []:
        keywords.update(_tokenize(str(parca).replace("_", " ")))
    soft = set()
    soft.update(_tokenize(entry.get("work") or ""))
    soft.update(_tokenize(entry.get("author") or ""))
    soft.update(_tokenize(entry.get("passage_hint") or ""))
    soft.update(_tokenize(entry.get("analysis") or ""))
    return themes, keywords, soft


@lru_cache(maxsize=1)
def _ref_bags() -> tuple[tuple[str, set[str], set[str], set[str]], ...]:
    out = []
    for e in load_refs():
        themes, keywords, soft = _entry_bags(e)
        out.append((e.get("id") or "", themes, keywords, soft))
    return tuple(out)


def _score_entry(
    query_tokens: set[str],
    themes: set[str],
    keywords: set[str],
    soft: set[str],
) -> float:
    if not query_tokens:
        return 0.0
    theme_hit = len(query_tokens & themes)
    key_hit = len(query_tokens & keywords)
    soft_hit = len(query_tokens & soft)
    if theme_hit + key_hit + soft_hit == 0:
        return 0.0
    ham = (3.0 * theme_hit) + (2.5 * key_hit) + (0.6 * soft_hit)
    norm = (len(query_tokens) ** 0.5) + 1.0
    return ham / norm


def _fallback_entries(refs: tuple[dict, ...], k: int) -> list[dict]:
    by_id = {e.get("id"): e for e in refs}
    secilen = [by_id[i] for i in FALLBACK_IDS if i in by_id]
    if len(secilen) >= k:
        return secilen[:k]
    for e in refs:
        if e not in secilen:
            secilen.append(e)
        if len(secilen) >= k:
            break
    return secilen[:k]


def retrieve(
    metin: str,
    k: int = MAX_EXAMPLES,
    match_text: str | None = None,
) -> list[dict]:
    """Return up to k reference analyses most related to the user text."""
    refs = load_refs()
    if not refs or k <= 0:
        return []

    query_tokens = _query_token_set(metin, match_text=match_text)
    if not query_tokens:
        return _fallback_entries(refs, k)

    by_id = {e.get("id"): e for e in refs}
    skorlu: list[tuple[float, dict]] = []
    for entry_id, themes, keywords, soft in _ref_bags():
        entry = by_id.get(entry_id)
        if not entry:
            continue
        skor = _score_entry(query_tokens, themes, keywords, soft)
        if skor >= MIN_ACCEPT_SCORE:
            skorlu.append((skor, entry))

    skorlu.sort(key=lambda x: x[0], reverse=True)
    if not skorlu:
        return _fallback_entries(refs, k)
    return [e for _, e in skorlu[:k]]


def format_for_prompt(entries: list[dict]) -> str:
    if not entries:
        return ""
    bloklar: list[str] = []
    for i, e in enumerate(entries, start=1):
        yazar = (e.get("author") or "Unknown").strip()
        eser = (e.get("work") or "Untitled").strip()
        analiz = (e.get("analysis") or "").strip()
        ipucu = (e.get("passage_hint") or "").strip()
        bloklar.append(
            f"Example {i} — method sample on {yazar}, {eser}:\n"
            f"Focus of sample: {ipucu}\n"
            f"How a careful reading proceeds: {analiz}"
        )
    return (
        "Use the following licensed close-reading samples ONLY as methodological "
        "guidance for tone, depth, and how to attend to subtext. "
        "Do NOT claim the user's text is from these works. "
        "Do NOT copy their sentences. Write an original insight about the user's text.\n\n"
        + "\n\n".join(bloklar)
    )


def method_examples_for(
    metin: str,
    k: int = MAX_EXAMPLES,
    match_text: str | None = None,
) -> str:
    """Convenience: retrieve + format for Gemini prompts."""
    return format_for_prompt(retrieve(metin, k=k, match_text=match_text))
