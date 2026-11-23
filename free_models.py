"""
Zero-cost mode for the bot.

When no paid LLM key (OpenRouter / OpenAI / Anthropic) is configured, the bot
runs on GitHub Models, the rate-limited free inference endpoint that every
GitHub account gets, authenticated with the GITHUB_TOKEN of the Actions run
(workflow permission `models: read`). Nothing is billed unless paid usage is
switched on in the GitHub account, which this project never does.

Two pieces live here:

1. build_free_llms(): picks a small ensemble of different forecasting models
   plus a cheap parser model from the live model catalog. Free limits are
   per model and per day, so spreading the calls over several models both
   diversifies the forecast and multiplies the daily budget.
2. fetch_news(): a key-less research step. It reads recent headlines from the
   Google News RSS search feed so the forecasters see dated, current
   information instead of relying on training data alone.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from forecasting_tools import GeneralLlm

logger = logging.getLogger(__name__)

GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"
GITHUB_MODELS_CATALOG_URL = "https://models.github.ai/catalog/models"
TOKEN_ENV = "GITHUB_MODELS_TOKEN"
PAID_KEY_ENVS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")

# Ordered preferences, matched against catalog ids (case-insensitive). At most
# one model per line is taken, so the ensemble mixes model families.
FORECASTER_PREFERENCES = [
    r"^openai/gpt-4\.1$",
    r"deepseek-v3",
    r"^xai/grok-3$",
    r"^openai/gpt-4o$",
    r"llama-4-maverick|llama-3\.3-70b",
    r"mistral-(large|medium)",
    r"^microsoft/phi-4$",
]
PARSER_PREFERENCES = [
    r"^openai/gpt-4\.1-mini$",
    r"^openai/gpt-4o-mini$",
    r"^openai/gpt-4\.1-nano$",
    r"mistral-small",
    r"^microsoft/phi-4-mini",
]
STATIC_FORECASTERS = ["openai/gpt-4.1", "openai/gpt-4o"]
STATIC_PARSERS = ["openai/gpt-4.1-mini", "openai/gpt-4o-mini"]
NUM_FORECASTERS = 3
NUM_PARSERS = 2


def _is_set(name: str) -> bool:
    value = (os.getenv(name) or "").strip()
    return bool(value) and not value.lower().startswith(("your_", "replace", "xxx"))


def free_mode() -> bool:
    """True when the run should use GitHub Models instead of a paid key."""
    return _is_set(TOKEN_ENV) and not any(_is_set(k) for k in PAID_KEY_ENVS)


class RotatingLlm(GeneralLlm):
    """
    A GeneralLlm that spreads calls over a pool of models (round robin) and
    falls through to the next model when one fails, e.g. on a rate limit.
    """

    def __init__(self, pool: list[GeneralLlm]) -> None:
        if not pool:
            raise ValueError("RotatingLlm needs at least one model")
        first = pool[0]
        super().__init__(
            model=first.model,
            temperature=first.litellm_kwargs.get("temperature"),
            timeout=first.litellm_kwargs.get("timeout"),
            allowed_tries=1,
            base_url=first.litellm_kwargs.get("base_url"),
            api_key=first.litellm_kwargs.get("api_key"),
        )
        self._pool = pool
        self._next_index = 0

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["pool"] = [llm.model for llm in self._pool]
        return data

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:
        start = self._next_index
        self._next_index += 1
        last_error: Exception | None = None
        for offset in range(len(self._pool)):
            llm = self._pool[(start + offset) % len(self._pool)]
            try:
                return await llm.invoke(prompt, system_prompt)
            except Exception as error:  # noqa: BLE001 - any failure moves on
                last_error = error
                logger.warning(
                    f"Model {llm.model} failed ({type(error).__name__}: "
                    f"{str(error)[:200]}); trying the next model in the pool"
                )
        assert last_error is not None
        raise last_error


def _fetch_catalog(token: str) -> list[dict]:
    request = urllib.request.Request(
        GITHUB_MODELS_CATALOG_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "metac-bot-free-mode",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data if isinstance(data, list) else []


def _is_text_chat_model(entry: dict) -> bool:
    model_id = str(entry.get("id", "")).lower()
    tier = str(entry.get("rate_limit_tier", "")).lower()
    outputs = [str(m).lower() for m in entry.get("supported_output_modalities") or []]
    if not model_id or "embed" in model_id or tier == "embeddings":
        return False
    return not outputs or "text" in outputs


def _pick(
    catalog: list[dict],
    preferences: list[str],
    count: int,
    tiers: tuple[str, ...],
    exclude: set[str],
) -> list[str]:
    usable = [
        str(e["id"])
        for e in catalog
        if _is_text_chat_model(e)
        and str(e.get("rate_limit_tier", "")).lower() in tiers
        and str(e["id"]) not in exclude
    ]
    chosen: list[str] = []
    for pattern in preferences:
        if len(chosen) >= count:
            break
        match = next(
            (m for m in usable if re.search(pattern, m, re.I) and m not in chosen),
            None,
        )
        if match:
            chosen.append(match)
    # Top up with other models of the wanted tiers, one per publisher first.
    seen_publishers = {m.split("/")[0] for m in chosen}
    for model_id in usable:
        if len(chosen) >= count:
            break
        publisher = model_id.split("/")[0]
        if model_id not in chosen and publisher not in seen_publishers:
            chosen.append(model_id)
            seen_publishers.add(publisher)
    for model_id in usable:
        if len(chosen) >= count:
            break
        if model_id not in chosen:
            chosen.append(model_id)
    return chosen


def _github_llm(model_id: str, token: str, temperature: float) -> GeneralLlm:
    # GitHub Models speaks the OpenAI chat-completions protocol, so litellm's
    # generic "openai/" provider is pointed at it with a custom base URL.
    return GeneralLlm(
        model=f"openai/{model_id}",
        temperature=temperature,
        timeout=90,
        allowed_tries=2,
        base_url=GITHUB_MODELS_BASE_URL,
        api_key=token,
    )


def build_free_llms() -> dict[str, GeneralLlm | str]:
    """The `llms` dict for the bot when running on GitHub Models."""
    token = os.environ[TOKEN_ENV]
    try:
        catalog = _fetch_catalog(token)
        print("GitHub Models catalog (id | tier | max input | max output):")
        for entry in catalog:
            limits = entry.get("limits") or {}
            print(
                f"    {entry.get('id')} | {entry.get('rate_limit_tier')} | "
                f"{limits.get('max_input_tokens')} | {limits.get('max_output_tokens')}"
            )
    except Exception as error:  # noqa: BLE001
        print(f"Could not read the GitHub Models catalog ({error}); using defaults")
        catalog = []

    forecasters = _pick(
        catalog, FORECASTER_PREFERENCES, NUM_FORECASTERS, ("high",), set()
    )
    if len(forecasters) < NUM_FORECASTERS:
        forecasters += _pick(
            catalog,
            FORECASTER_PREFERENCES,
            NUM_FORECASTERS - len(forecasters),
            ("low",),
            set(forecasters),
        )
    if not forecasters:
        forecasters = STATIC_FORECASTERS
    parsers = _pick(catalog, PARSER_PREFERENCES, NUM_PARSERS, ("low",), set(forecasters))
    if not parsers:
        parsers = STATIC_PARSERS

    print(f"Free mode. Forecasting models: {forecasters}. Parser models: {parsers}.")
    forecaster = RotatingLlm([_github_llm(m, token, 0.3) for m in forecasters])
    parser = RotatingLlm([_github_llm(m, token, 0.0) for m in parsers])
    return {
        "default": forecaster,
        "summarizer": parser,
        "researcher": "no_research",
        "parser": parser,
    }


def predictions_per_question() -> int:
    return NUM_FORECASTERS


# ----------------------------------------------------------------- research

_STOPWORDS = {
    "will", "the", "a", "an", "be", "by", "of", "in", "on", "at", "to", "for",
    "before", "after", "between", "and", "or", "is", "are", "was", "were",
    "what", "which", "who", "how", "many", "much", "when", "does", "do", "did",
    "have", "has", "had", "than", "that", "this", "these", "those", "any",
    "its", "their", "with", "from", "as", "least", "most", "more", "less",
    "according", "there", "happen", "occur", "following",
}
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december"
)


def search_queries(question_text: str) -> list[str]:
    """Two key-less search queries derived from the question title."""
    text = re.sub(r"\s+", " ", question_text).strip().rstrip("?")
    without_dates = re.sub(
        rf"\b(({_MONTHS})\s+\d{{1,2}}(,\s*\d{{4}})?|\d{{4}}|({_MONTHS}))\b",
        " ",
        text,
        flags=re.I,
    )
    words = [w for w in re.findall(r"[A-Za-z0-9$%.'&-]+", without_dates)]
    keywords = [w for w in words if w.lower() not in _STOPWORDS and len(w) > 1]
    compact = " ".join(keywords[:9])
    queries = [q for q in (compact, text[:150]) if q]
    return list(dict.fromkeys(queries))


def _read_feed(query: str, max_age_days: int) -> list[tuple[datetime, str, str]]:
    url = (
        "https://news.google.com/rss/search?"
        + urllib.parse.urlencode(
            {"q": f"{query} when:{max_age_days}d", "hl": "en-US", "gl": "US", "ceid": "US:en"}
        )
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=15) as response:
        root = ET.fromstring(response.read())
    items: list[tuple[datetime, str, str]] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "")
        except (TypeError, ValueError):
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if title:
            items.append((published, source, title))
    return items


def _collect_headlines(question_text: str, max_items: int, max_age_days: int) -> str:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    seen: set[str] = set()
    collected: list[tuple[datetime, str, str]] = []
    for query in search_queries(question_text):
        try:
            # Feed order is relevance; keep the best-ranked items per query.
            for published, source, title in _read_feed(query, max_age_days)[:25]:
                key = re.sub(r"\W+", "", title.lower())[:80]
                if published >= cutoff and key not in seen:
                    seen.add(key)
                    collected.append((published, source, title))
        except Exception as error:  # noqa: BLE001
            logger.warning(f"News lookup failed for '{query}': {error}")
    if not collected:
        return ""
    collected.sort(key=lambda entry: entry[0], reverse=True)
    lines = [
        f"- {published:%Y-%m-%d} | {source or 'unknown source'} | {title}"
        for published, source, title in collected[:max_items]
    ]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        f"Recent news headlines (retrieved {today}, newest first; headlines only, "
        "so treat them as leads, not as verified facts):\n" + "\n".join(lines)
    )


async def fetch_news(
    question_text: str, max_items: int = 18, max_age_days: int = 45
) -> str:
    """Recent dated headlines for a question, or an empty string."""
    return await asyncio.to_thread(
        _collect_headlines, question_text, max_items, max_age_days
    )
