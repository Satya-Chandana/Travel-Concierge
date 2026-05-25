"""
Travel Concierge — FastAPI backend.

Endpoints
---------
POST /api/chat              — agent chat
POST /api/places            — search + LTR re-rank places
POST /api/weather           — weather / forecast
POST /api/route             — directions
POST /api/itinerary         — DP-optimised multi-day schedule
POST /api/feedback          — record accept/skip for LTR training
GET  /api/health            — health check
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
from ml.ranking import FeedbackStore, LambdaMARTRanker

_feedback_store = FeedbackStore(Path("data/feedback_store.json"))
_ranker = LambdaMARTRanker(Path("data/ltr_model.pkl"))
_agents: dict[str, Any] = {}


def _get_agent(session_id: str):
    if session_id not in _agents:
        from agents.tools import build_tools
        from agents.travel_agent import TravelAgent
        tools = build_tools()
        try:
            from langchain_community.agent_toolkits.load_tools import load_tools
            from langchain_openai import ChatOpenAI
            from config import settings
            _llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,
                              openai_api_key=settings.openai_api_key)
            wiki = load_tools(["wikipedia"], llm=_llm)
            tools.extend(wiki)
        except Exception as e:
            log.warning("Wikipedia tool skipped: %s", e)
        _agents[session_id] = TravelAgent(tools=tools, verbose=False)
    return _agents[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Travel Concierge API v2 starting up.")
    yield
    log.info("Shutting down.")


app = FastAPI(title="Travel Concierge API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str
    place: Optional[str] = None
    interests: Optional[str] = None

class PlacesRequest(BaseModel):
    session_id: str = "default"
    location: str
    interest: str = "tourist attractions"
    centre_lat: Optional[float] = None
    centre_lon: Optional[float] = None

class WeatherRequest(BaseModel):
    location: str
    days: int = 3

class RouteRequest(BaseModel):
    locations: list[str]
    transport_mode: str = "driving"
    city_context: str = ""

class ItineraryRequest(BaseModel):
    session_id: str = "default"
    location: str
    interests: str = ""
    num_days: int = 3

class FeedbackRequest(BaseModel):
    session_id: str
    place_name: str
    place_types: list[str]
    features: list[float]
    accepted: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        from agents.intent_classifier import classify, Intent
        classification = classify(req.message)
        intent = classification.get("intent", "general")
        is_itinerary = intent == "itinerary"

        context = ""
        if req.interests:
            context += f"My interests: {req.interests}.\n"
        if req.place:
            context += f"I am planning a trip to {req.place}.\n"

        full_prompt = (context + req.message).strip()
        agent = _get_agent(req.session_id)
        response = agent.chat(full_prompt, run_critique=is_itinerary)
        return {"response": response, "intent": intent, "session_id": req.session_id}
    except Exception as e:
        log.error("Chat error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/places")
async def places(req: PlacesRequest):
    """Search places and re-rank using LambdaMART personalisation."""
    try:
        from services.geocoding import geocode
        from services.places import search_nearby

        coord = geocode(req.location)
        centre_lat = req.centre_lat or coord.latitude
        centre_lon = req.centre_lon or coord.longitude

        raw_places = search_nearby(coord.latitude, coord.longitude, req.interest)
        place_dicts = [
            {
                "name": p.name, "address": p.address, "rating": p.rating,
                "user_ratings_total": p.user_ratings_total,
                "lat": p.latitude, "lon": p.longitude,
                "types": p.types, "price_level": p.price_level,
            }
            for p in raw_places
        ]

        # LTR re-ranking
        from ml.ranking import rank_places
        ranked = rank_places(
            place_dicts, req.session_id, req.interest,
            centre_lat, centre_lon, _feedback_store, _ranker
        )

        return {
            "places": ranked,
            "ranking_method": "lambdamart" if _ranker._model else "heuristic_coldstart",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/itinerary")
async def itinerary(req: ItineraryRequest):
    """
    Build a DP-optimised, constraint-aware multi-day itinerary.
    Places are first LTR-ranked, then scheduled with Weighted Interval Scheduling.
    """
    try:
        from services.geocoding import geocode
        from services.places import search_nearby
        from ml.ranking import rank_places
        from ml.scheduler import schedule_multi_day

        coord = geocode(req.location)

        # Fetch more candidates for multi-day
        raw_places = search_nearby(
            coord.latitude, coord.longitude, req.interests or "tourist attractions",
            max_results=min(30, req.num_days * 8),
        )
        place_dicts = [
            {
                "name": p.name, "address": p.address, "rating": p.rating,
                "user_ratings_total": p.user_ratings_total,
                "lat": p.latitude, "lon": p.longitude,
                "types": p.types, "price_level": p.price_level,
            }
            for p in raw_places
        ]

        # Step 1: LTR re-rank
        ranked = rank_places(
            place_dicts, req.session_id, req.interests,
            coord.latitude, coord.longitude, _feedback_store, _ranker
        )

        # Step 2: DP schedule across days
        days = schedule_multi_day(ranked, req.num_days, city=req.location)

        return {
            "location": req.location,
            "num_days": req.num_days,
            "days": [
                {
                    "label": d.date_label,
                    "formatted": d.format(),
                    "visits": [
                        {
                            "name": v.place_name,
                            "address": v.address,
                            "arrival": v.arrival_str(),
                            "departure": v.departure_str(),
                            "lat": v.lat,
                            "lon": v.lon,
                        }
                        for v in d.visits
                    ],
                    "meals": [
                        {"type": m.meal_type, "start": m.start_str(),
                         "end": m.end_str(), "suggestion": m.suggestion}
                        for m in d.meals
                    ],
                    "total_stops": d.total_stops,
                    "note": d.optimality_note,
                }
                for d in days
            ],
            "ranking_method": "lambdamart" if _ranker._model else "heuristic_coldstart",
        }
    except Exception as e:
        log.error("Itinerary error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    """Record user accept/skip for LTR model training."""
    try:
        from ml.ranking import FeedbackEvent
        event = FeedbackEvent(
            session_id=req.session_id,
            place_name=req.place_name,
            place_types=req.place_types,
            features=req.features,
            accepted=1 if req.accepted else 0,
        )
        _feedback_store.record(event)

        # Retrain if we have enough data
        trained = _ranker.train(_feedback_store)
        return {"recorded": True, "model_retrained": trained,
                "total_feedback": len(_feedback_store.events)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/weather")
async def weather(req: WeatherRequest):
    try:
        from services.geocoding import geocode
        from services.weather import get_current, get_forecast, format_forecast
        coord = geocode(req.location)
        if req.days <= 1:
            w = get_current(coord.latitude, coord.longitude)
            return {"formatted": w.format(), "type": "current"}
        else:
            forecast = get_forecast(coord.latitude, coord.longitude, days=req.days)
            return {"formatted": format_forecast(forecast, req.location), "type": "forecast",
                    "days": [{"date": d.date, "description": d.description,
                               "max": d.temp_max_c, "min": d.temp_min_c} for d in forecast]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/route")
async def route(req: RouteRequest):
    try:
        from services.routing import get_route
        result = get_route(req.locations, req.transport_mode,
                           city_context=req.city_context, optimise=True)
        return {"directions": result.format_directions(),
                "geocode_points": result.geocode_points,
                "distance_km": result.total_distance_km,
                "duration_min": result.total_duration_min,
                "optimised": result.optimised}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in _agents:
        _agents[session_id].clear_memory()
        del _agents[session_id]
    return {"cleared": True}
