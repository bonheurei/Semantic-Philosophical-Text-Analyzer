"""User feedback loop with durable remote storage (Supabase) + local fallback.

Upvotes → preferred close-reading exemplars for later Gemini prompts.
Downvotes → short 'avoid this failure mode' notes.

Remote (multi-user / Streamlit Cloud):
  secrets: SUPABASE_URL, SUPABASE_KEY
  table: insight_feedback  (see data/feedback_schema.sql)

Local fallback when remote secrets are missing: data/feedback.jsonl
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import literary_memory

DATA_PATH = Path(__file__).resolve().parent / "data" / "feedback.jsonl"
TABLE_NAME = "insight_feedback"
MAX_UP_EXAMPLES = 2
MAX_DOWN_NOTES = 2
TEXT_STORE_CHARS = 1200
INSIGHT_STORE_CHARS = 900
REMOTE_FETCH_LIMIT = 400
_TOKEN = re.compile(
    r"[a-zA-ZçğıöşüÇĞİÖŞÜâîûÂÎÛа-яА-ЯёЁ\u3040-\u30ff\u3400-\u9fff']+",
    re.UNICODE,
)


def _tokenize(metin: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(metin or "") if len(t) > 2}


def _ensure_parent() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)


def _secret_get(name: str) -> str:
    env = (os.environ.get(name) or "").strip()
    if env:
        return env
    try:
        import streamlit as st

        raw = st.secrets.get(name, "")
        return str(raw).strip().strip('"').strip("'")
    except Exception:
        return ""


def remote_enabled() -> bool:
    return bool(_secret_get("SUPABASE_URL") and _secret_get("SUPABASE_KEY"))


def storage_mode() -> str:
    return "supabase" if remote_enabled() else "local"


def _supabase_headers(*, write: bool = False) -> dict[str, str]:
    key = _secret_get("SUPABASE_KEY")
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    if write:
        headers["Content-Type"] = "application/json"
        headers["Prefer"] = "return=representation"
    return headers


def _supabase_url(path_query: str) -> str:
    base = _secret_get("SUPABASE_URL").rstrip("/")
    return f"{base}/rest/v1/{path_query.lstrip('/')}"


def _http_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    body: dict | None = None,
    timeout: int = 20,
) -> object:
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                return None
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Supabase HTTP {exc.code}: {detail[:300]}"
        ) from exc


def _normalize_row(row: dict) -> dict | None:
    if not isinstance(row, dict):
        return None
    vote = row.get("vote")
    if vote not in {"up", "down"}:
        return None
    emotions = row.get("emotions") or []
    if isinstance(emotions, str):
        try:
            emotions = json.loads(emotions)
        except json.JSONDecodeError:
            emotions = []
    return {
        "id": str(row.get("id") or ""),
        "ts": int(row.get("ts") or 0),
        "vote": vote,
        "text": str(row.get("text") or ""),
        "insight": str(row.get("insight") or ""),
        "emotions": emotions if isinstance(emotions, list) else [],
        "results_lang": str(row.get("results_lang") or "en"),
    }


def _load_local(limit: int = 400) -> list[dict]:
    if not DATA_PATH.is_file():
        return []
    kayitlar: list[dict] = []
    with DATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            norm = _normalize_row(row)
            if norm:
                kayitlar.append(norm)
    return kayitlar[-limit:]


def _save_local(kayit: dict) -> None:
    _ensure_parent()
    with DATA_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kayit, ensure_ascii=False) + "\n")


def _load_remote(limit: int = REMOTE_FETCH_LIMIT) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "select": "id,ts,vote,text,insight,emotions,results_lang",
            "order": "ts.desc",
            "limit": str(limit),
        }
    )
    url = _supabase_url(f"{TABLE_NAME}?{query}")
    payload = _http_json("GET", url, headers=_supabase_headers())
    if not isinstance(payload, list):
        return []
    kayitlar: list[dict] = []
    for row in payload:
        norm = _normalize_row(row)
        if norm:
            kayitlar.append(norm)
    # Oldest → newest for fallback slicing consistency with local.
    kayitlar.reverse()
    return kayitlar


def _save_remote(kayit: dict) -> None:
    url = _supabase_url(TABLE_NAME)
    _http_json(
        "POST",
        url,
        headers=_supabase_headers(write=True),
        body=kayit,
    )


def load_feedback(limit: int = 400) -> list[dict]:
    if remote_enabled():
        try:
            return _load_remote(limit=limit)
        except Exception:
            # Prefer continuity over hard failure during retrieval.
            return _load_local(limit=limit)
    return _load_local(limit=limit)


def save_feedback(
    *,
    vote: str,
    text: str,
    insight: str,
    emotions: list[dict] | None = None,
    results_lang: str = "en",
) -> dict:
    if vote not in {"up", "down"}:
        raise ValueError("vote must be 'up' or 'down'")
    temiz_text = (text or "").strip()[:TEXT_STORE_CHARS]
    temiz_insight = (insight or "").strip()[:INSIGHT_STORE_CHARS]
    if not temiz_text or not temiz_insight:
        raise ValueError("text and insight are required")
    kayit = {
        "id": uuid.uuid4().hex[:12],
        "ts": int(time.time()),
        "vote": vote,
        "text": temiz_text,
        "insight": temiz_insight,
        "emotions": emotions or [],
        "results_lang": results_lang,
    }
    if remote_enabled():
        _save_remote(kayit)
        return kayit
    _save_local(kayit)
    return kayit


def _score_against_query(query_tokens: set[str], row: dict) -> float:
    if not query_tokens:
        return 0.0
    bag = _tokenize(row.get("text") or "") | _tokenize(row.get("insight") or "")
    try:
        bag |= literary_memory._lexicon_themes(_tokenize(row.get("text") or ""))
        query_tokens = query_tokens | literary_memory._lexicon_themes(query_tokens)
    except Exception:
        pass
    if not bag:
        return 0.0
    overlap = query_tokens & bag
    if not overlap:
        return 0.0
    return len(overlap) / ((len(query_tokens) ** 0.5) + 1.0)


def retrieve_feedback(
    metin: str,
    *,
    vote: str,
    k: int = 2,
    match_text: str | None = None,
) -> list[dict]:
    rows = [r for r in load_feedback() if r.get("vote") == vote]
    if not rows or k <= 0:
        return []
    query = _tokenize(metin)
    if match_text:
        query |= _tokenize(match_text)
    try:
        query |= literary_memory._lexicon_themes(query)
        if not match_text:
            bridged = literary_memory._bridge_to_english(metin)
            query |= _tokenize(bridged)
            query |= literary_memory._lexicon_themes(_tokenize(bridged))
    except Exception:
        pass
    if not query:
        return rows[-k:]

    skorlu: list[tuple[float, dict]] = []
    for row in rows:
        skor = _score_against_query(query, row)
        if skor > 0:
            skorlu.append((skor, row))
    skorlu.sort(key=lambda x: x[0], reverse=True)
    if skorlu:
        return [r for _, r in skorlu[:k]]
    return rows[-k:]


def format_feedback_for_prompt(
    metin: str, match_text: str | None = None
) -> str:
    ups = retrieve_feedback(
        metin, vote="up", k=MAX_UP_EXAMPLES, match_text=match_text
    )
    downs = retrieve_feedback(
        metin, vote="down", k=MAX_DOWN_NOTES, match_text=match_text
    )
    if not ups and not downs:
        return ""

    bloklar: list[str] = []
    if ups:
        parcalar = []
        for i, row in enumerate(ups, start=1):
            parcalar.append(
                f"Approved example {i}:\n"
                f"Source passage (excerpt): {(row.get('text') or '')[:280]}\n"
                f"Approved insight: {(row.get('insight') or '').strip()}"
            )
        bloklar.append(
            "User-approved insights for similar writing. Prefer this depth, "
            "restraint, and attention to lived tone—do not copy wording:\n\n"
            + "\n\n".join(parcalar)
        )
    if downs:
        parcalar = []
        for i, row in enumerate(downs, start=1):
            parcalar.append(
                f"Rejected example {i}:\n"
                f"{(row.get('insight') or '').strip()[:320]}"
            )
        bloklar.append(
            "User-rejected insights for similar writing. Avoid the same "
            "failure modes (overclaiming, wrong tone, plot-driven emotion, "
            "generic filler):\n\n"
            + "\n\n".join(parcalar)
        )
    return "\n\n".join(bloklar)
