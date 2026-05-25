"""
Central configuration — loaded once at startup.
All API keys and tunable constants live here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ── LLM ────────────────────────────────────────────────────────────────
    openai_api_key: str = field(default_factory=lambda: os.getenv("FIREWORKS_API_KEY", ""))
    llm_model: str = "accounts/fireworks/models/gpt-oss-20b"
    llm_temperature: float = 0.0
    summary_memory_token_limit: int = 1500

    # ── Maps / Places ───────────────────────────────────────────────────────
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    places_default_radius_m: int = 5_000   # metres
    places_max_results: int = 10

    # ── Weather ─────────────────────────────────────────────────────────────
    openweather_api_key: str = field(default_factory=lambda: os.getenv("OPENWEATHER_API_KEY", ""))
    weather_default_forecast_days: int = 3
    weather_max_forecast_days: int = 7

    # ── Auth (Firebase) ──────────────────────────────────────────────────────
    firebase_api_key: str = field(default_factory=lambda: os.getenv("FIREBASE_API_KEY", "AIzaSyAW_g9Kb16cY6XvQzTX49N_SBG84DdK1Ss"))
    firebase_auth_domain: str = "travelbot-69f23.firebaseapp.com"
    firebase_project_id: str = "travelbot-69f23"
    firebase_storage_bucket: str = "travelbot-69f23.appspot.com"
    firebase_messaging_sender_id: str = "656502753345"
    firebase_app_id: str = "1:656502753345:web:93cd5c754ddd3be1719ec0"

    # ── Observability ────────────────────────────────────────────────────────
    langsmith_tracing: bool = field(
        default_factory=lambda: os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    )
    langsmith_api_key: str = field(default_factory=lambda: os.getenv("LANGCHAIN_API_KEY", ""))
    langsmith_project: str = field(
        default_factory=lambda: os.getenv("LANGCHAIN_PROJECT", "travel-concierge")
    )

    # ── Route optimisation ───────────────────────────────────────────────────
    tsp_max_waypoints: int = 12   # greedy nearest-neighbour cap


# Singleton — import this everywhere instead of calling os.getenv directly.
settings = Settings()
