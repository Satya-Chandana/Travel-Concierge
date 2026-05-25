"""
Intent classifier.

Replaces the brittle keyword-matching ``detect_intent()`` function with a
structured LLM call that returns a typed result.

Why this matters
----------------
Keyword lists break on paraphrases ("how do I get from A to B?" misses
"route" / "directions"), on negation ("not a weather question"), and on
multi-intent queries ("places to eat near my hotel route").

The classifier prompts the LLM to produce a strict JSON object, then parses
it.  A fast, cheap model (gpt-3.5-turbo) is used so latency is < 200 ms.

Fallback
--------
If the LLM call fails for any reason (network, quota, bad JSON) the module
falls back to the original keyword heuristic so the app never crashes.
"""
from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Optional

import requests

from config import settings

log = logging.getLogger(__name__)


class Intent(str, Enum):
    PLACES = "places"
    ROUTE = "route"
    WEATHER = "weather"
    ITINERARY = "itinerary"
    RESTAURANT = "restaurant"
    GENERAL = "general"


_SYSTEM_PROMPT = """\
You are a travel intent classifier.
Given a user message, output a JSON object with exactly these keys:
  - "intent": one of ["places","route","weather","itinerary","restaurant","general"]
  - "locations": a list of location strings mentioned (empty list if none)
  - "days": integer number of days if mentioned, else null
  - "transport_mode": string if a transport mode is mentioned, else null
  - "interests": list of interest keywords if mentioned, else []

Rules:
- "itinerary" = multi-day trip plan request
- "places" = looking for attractions / things to do
- "route" = directions / how to get somewhere
- "weather" = current or forecast weather
- "restaurant" = food / dining / eating
- "general" = anything else

Respond with ONLY the JSON object. No markdown, no explanation.\
"""


def classify(user_message: str) -> dict:
    """
    Classify user intent using GPT-3.5-turbo with a structured JSON schema.

    Returns a dict with keys: intent, locations, days, transport_mode, interests.
    Falls back to heuristic classification on any error.
    """
    try:
        resp = requests.post(
            "https://api.fireworks.ai/inference/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "accounts/fireworks/models/gpt-oss-20b",
                "temperature": 0,
                "max_tokens": 200,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            },
            timeout=5,
        )
        resp.raise_for_status()
        raw_content = resp.json()["choices"][0]["message"]["content"]
        # Strip accidental markdown fences
        raw_content = re.sub(r"```[a-z]*\n?", "", raw_content).strip()
        parsed = json.loads(raw_content)
        # Enforce valid intent value
        parsed["intent"] = Intent(parsed.get("intent", "general")).value
        log.debug("Classified %r → %s", user_message[:60], parsed["intent"])
        return parsed
    except Exception as exc:
        log.warning("Intent classifier LLM call failed (%s). Using heuristic fallback.", exc)
        return _heuristic_classify(user_message)


def _heuristic_classify(text: str) -> dict:
    """Keyword-based fallback — same logic as the original detect_intent."""
    t = text.lower()

    intent = Intent.GENERAL
    if any(w in t for w in ["itinerary", "trip plan", "travel plan", "schedule", "day plan"]):
        intent = Intent.ITINERARY
    elif any(w in t for w in ["restaurant", "eat", "food", "lunch", "dinner", "breakfast", "cafe"]):
        intent = Intent.RESTAURANT
    elif any(w in t for w in ["attraction", "places", "visit", "sightseeing", "tourist", "things to do"]):
        intent = Intent.PLACES
    elif any(w in t for w in ["route", "how to get", "directions", "transport", "travel from", "way to"]):
        intent = Intent.ROUTE
    elif any(w in t for w in ["weather", "forecast", "temperature", "rain", "climate"]):
        intent = Intent.WEATHER

    # Extract locations heuristically
    location_match = re.findall(r"(?:in|to|from|at|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
    days_match = re.search(r"(\d+)\s*(?:-\s*)?day", t)

    return {
        "intent": intent.value,
        "locations": location_match,
        "days": int(days_match.group(1)) if days_match else None,
        "transport_mode": None,
        "interests": [],
    }
