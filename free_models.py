"""
Zero-cost mode for the bot.

When OPENROUTER_FREE_ONLY is not set to "false", the bot only ever calls
OpenRouter models whose id ends in ":free". Those cost nothing, need no
credits on the account, and are rate limited by OpenRouter (a small number
of requests per day), so this module is built to spend very few calls:

1. build_free_llms(): picks two different free models from OpenRouter's live
   public model list and rotates between them (falling through to the other
   one when a model is unavailable).
2. fetch_news(): a key-less research step. It reads recent headlines from the
   Google News RSS search feed so the forecasters see dated, current
   information instead of relying on training data alone.
3. probability_from_text(): reads the final "Probability: ZZ%" line of a
   binary forecast directly, so no extra parsing call is needed.

Set the repository variable OPENROUTER_FREE_ONLY to "false" to go back to
the template's paid default models (e.g. once donated credits are on the key).
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

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
KEY_ENV = "OPENROUTER_API_KEY"
FREE_ONLY_ENV = "OPENROUTER_FREE_ONLY"

# Ordered preferences, matched against free model ids (case-insensitive). One
# model per line at most, so the pool mixes model families.
FORECASTER_PREFERENCES = [
    r"nemotron-3-ultra",
    r"gemma-4-31b",
    r"nemotron-3\.5-super",
    r"deepseek",
    r"gpt-oss-120b",
    r"llama-(4|3\.3-70b)",
    r"qwen",
    r"gemma",
    r"mistral",
    r"ernie",
]
STATIC_FORECASTERS = [
    "deepseek/deepseek-chat-v3-0324:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]
NUM_FORECASTERS = 2  # forecasts per question
POOL_SIZE = 4  # free models are often throttled upstream, so keep spares
MIN_CONTEXT_TOKENS = 32000
TEST_QUESTION_LIMIT = 3


# Paid mode (OPENROUTER_FREE_ONLY=false): fixed, mid-priced models so the cost
# per question stays predictable (roughly 4-6 US cents at Nov 2026 prices).
PAID_FORECASTERS = [
    "openai/gpt-6.1-terra",
    "google/gemini-4-hydrogen",
    "anthropic/claude-sonnet-5.5",
]
PAID_PARSER = "anthropic/claude-haiku-5.5"
PAID_RESEARCHER = "perplexity/sonar"  # searches the web itself
PAID_MAX_OUTPUT_TOKENS = 2500


def _is_set(name: str) -> bool:
    value = (os.getenv(name) or "").strip()
    return bool(value) and not value.lower().startswith(("your_", "replace", "xxx"))


def free_mode() -> bool:
    """True unless paid models were explicitly allowed for this repository."""
    free_only = (os.getenv(FREE_ONLY_ENV) or "").strip().lower() != "false"
    return _is_set(KEY_ENV) and free_only


DAILY_LIMIT_MARKER = "free-models-per-day"
_daily_limit_reached = False


class DailyLimitReached(RuntimeError):
    """OpenRouter's free daily limit is used up; no further calls this run."""


def paid_mode() -> bool:
    """True when paid OpenRouter models were explicitly switched on."""
    return _is_set(KEY_ENV) and not free_mode()


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
        )
        self._pool = pool
        self._next_index = 0

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["pool"] = [llm.model for llm in self._pool]
        return data

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:
        global _daily_limit_reached
        if _daily_limit_reached:
            raise DailyLimitReached("Free daily limit reached earlier in this run")
        start = self._next_index
        self._next_index += 1
        last_error: Exception | None = None
        for offset in range(len(self._pool)):
            llm = self._pool[(start + offset) % len(self._pool)]
            try:
                return await llm.invoke(prompt, system_prompt)
            except Exception as error:  # noqa: BLE001 - any failure moves on
                last_error = error
                if DAILY_LIMIT_MARKER in str(error):
                    # Every further call would fail and still hit the API.
                    _daily_limit_reached = True
                    raise DailyLimitReached(
                        "OpenRouter free daily limit reached; the remaining "
                        "questions are picked up by a later run"
                    ) from error
                logger.warning(
                    f"Model {llm.model} failed ({type(error).__name__}: "
                    f"{str(error)[:200]}); trying the next model in the pool"
                )
        assert last_error is not None
        raise last_error


def _fetch_free_models() -> list[str]:
    request = urllib.request.Request(
        OPENROUTER_MODELS_URL, headers={"User-Agent": "metac-bot-free-mode"}
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
    free: list[str] = []
    for entry in data.get("data", []):
        model_id = str(entry.get("id", ""))
        outputs = (entry.get("architecture") or {}).get("output_modalities") or ["text"]
        context = entry.get("context_length") or 0
        if (
            model_id.endswith(":free")
            and "text" in outputs
            and context >= MIN_CONTEXT_TOKENS
        ):
            free.append(model_id)
    return free


def _pick(usable: list[str], preferences: list[str], count: int) -> list[str]:
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
    for model_id in usable:
        if len(chosen) >= count:
            break
        if model_id not in chosen:
            chosen.append(model_id)
    return chosen


def _free_llm(model_id: str, temperature: float) -> GeneralLlm:
    assert model_id.endswith(":free"), f"{model_id} is not a free model"
    return GeneralLlm(
        model=f"openrouter/{model_id}",
        temperature=temperature,
        timeout=120,
        allowed_tries=1,  # failed calls count against the daily free limit
    )


def build_free_llms() -> dict[str, GeneralLlm | str]:
    """The `llms` dict for the bot when running on free OpenRouter models."""
    try:
        free_models = _fetch_free_models()
        print(f"Free OpenRouter models right now: {free_models}")
    except Exception as error:  # noqa: BLE001
        print(f"Could not read the OpenRouter model list ({error}); using defaults")
        free_models = []
    forecasters = _pick(free_models, FORECASTER_PREFERENCES, POOL_SIZE)
    if not forecasters:
        forecasters = STATIC_FORECASTERS
    print(f"Free mode. Models in rotation: {forecasters}.")
    forecaster = RotatingLlm([_free_llm(m, 0.3) for m in forecasters])
    parser = RotatingLlm([_free_llm(m, 0.0) for m in reversed(forecasters)])
    return {
        "default": forecaster,
        "summarizer": parser,
        "researcher": "no_research",
        "parser": parser,
    }


def _paid_llm(model_id: str) -> GeneralLlm:
    return GeneralLlm(
        model=f"openrouter/{model_id}",
        temperature=None,  # several of these models reject a temperature
        timeout=180,
        allowed_tries=2,
        max_tokens=PAID_MAX_OUTPUT_TOKENS,
    )


def build_paid_llms() -> dict[str, GeneralLlm | str]:
    """The `llms` dict for the bot when paid OpenRouter models are allowed."""
    print(
        f"Paid mode. Forecasters: {PAID_FORECASTERS}. Parser: {PAID_PARSER}. "
        f"Researcher: {PAID_RESEARCHER}."
    )
    parser = _paid_llm(PAID_PARSER)
    return {
        "default": RotatingLlm([_paid_llm(m) for m in PAID_FORECASTERS]),
        "summarizer": parser,
        "researcher": "no_research",  # research is done in web_research()
        "parser": parser,
    }


async def web_research(question_text: str, resolution_criteria: str) -> str:
    """A short, dated news rundown from a web-searching model plus headlines."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = (
        "You are a research assistant to a professional forecaster. Today is "
        f"{today}. Search the web and write a concise rundown (at most 350 words) "
        "of the most relevant and most recent facts for the question below. Give "
        "dates for every fact, say what the current status is, and say plainly "
        "when you found nothing recent. Do not give a forecast.\n\n"
        f"Question: {question_text}\n\nResolution criteria: {resolution_criteria}"
    )
    rundown = ""
    try:
        rundown = await _paid_llm(PAID_RESEARCHER).invoke(prompt)
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Web research failed: {type(error).__name__}: {str(error)[:200]}")
    headlines = await fetch_news(question_text, max_items=10)
    return "\n\n".join(part for part in (rundown, headlines) if part)


def predictions_per_question() -> int:
    return len(PAID_FORECASTERS) if paid_mode() else NUM_FORECASTERS


def probability_from_text(reasoning: str) -> float | None:
    """
    The decimal probability from the last "Probability: ZZ%" line of a binary
    forecast, or None when the text has no unambiguous one.
    """
    matches = re.findall(
        r"probability\s*[:=]\s*\**\s*(\d{1,3}(?:\.\d+)?)\s*%", reasoning, flags=re.I
    )
    if not matches:
        return None
    value = float(matches[-1])
    if not 0 <= value <= 100:
        return None
    return value / 100


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
            for published, source, title in _read_feed(query, max_age_days)[:10]:
                key = re.sub(r"\W+", "", title.lower())[:80]
                if published >= cutoff and key not in seen:
                    seen.add(key)
                    collected.append((published, source, title))
        except Exception as error:  # noqa: BLE001
            logger.warning(f"News lookup failed for '{query}': {error}")
    if not collected:
        return ""
    lines = [
        f"- {published:%Y-%m-%d} | {source or 'unknown source'} | {title}"
        for published, source, title in collected[:max_items]
    ]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        f"Recent news headlines (retrieved {today}, most relevant first; headlines only, "
        "so treat them as leads, not as verified facts):\n" + "\n".join(lines)
    )


async def fetch_news(
    question_text: str, max_items: int = 14, max_age_days: int = 45
) -> str:
    """Recent dated headlines for a question, or an empty string."""
    return await asyncio.to_thread(
        _collect_headlines, question_text, max_items, max_age_days
    )
