"""
LangChain tools used by the ReAct travel agent.

Each tool:
  - Has a single, clearly described responsibility.
  - Delegates to a service module (services/) — no business logic here.
  - Returns a plain string so the agent can reason over it.
  - Handles all exceptions internally and returns a user-friendly error
    message rather than crashing the agent loop.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Union

from langchain.tools import BaseTool

from services.geocoding import geocode, GeocodingError, reverse_geocode
from services.places import search_nearby, format_places, normalise_interest, search_restaurants
from services.routing import get_route, normalise_mode
from services.weather import get_current, get_forecast, format_forecast, WeatherServiceError

log = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_input(raw: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Normalise tool input to a dict regardless of how the agent serialised it."""
    if isinstance(raw, dict):
        return raw
    raw = str(raw).strip()
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return {"location": raw}


# ── Places Of Interest ────────────────────────────────────────────────────────

class PlacesOfInterestTool(BaseTool):
    """Return a ranked list of nearby attractions for a given city and interest."""

    name: str = "places_of_interest"
    description: str = (
        "Find popular nearby places for a city and interest category. "
        "Input: JSON with 'location' (city name) and 'interest' (e.g. 'museums', 'beaches'). "
        "Returns a numbered list of places with ratings and addresses."
    )

    def _run(self, raw_input: Union[str, Dict[str, Any]] = "") -> str:
        data = _parse_input(raw_input)
        location_name = data.get("location", "").strip()
        interest = data.get("interest", "tourist attractions").strip()

        if not location_name:
            return "⚠️ Please provide a location name."

        try:
            coord = geocode(location_name)
        except GeocodingError as exc:
            return f"⚠️ {exc}"

        try:
            interest_norm = normalise_interest(interest)
            places = search_nearby(coord.latitude, coord.longitude, interest_norm)
            return format_places(places)
        except Exception as exc:
            log.error("PlacesOfInterestTool error: %s", exc)
            return "⚠️ Could not retrieve places of interest. Please try again."

    def _arun(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Async not supported.")


# ── Route Retriever ───────────────────────────────────────────────────────────

class RouteRetrieverTool(BaseTool):
    """Get turn-by-turn directions between two or more locations."""

    name: str = "route_retriever"
    description: str = (
        "Get directions between locations. "
        "Input: JSON with 'locations' (list of 2+ place names) and 'transport_mode' "
        "(e.g. 'driving', 'walking', 'transit', 'bicycling'). "
        "Returns step-by-step directions and total travel time."
    )

    def _run(self, raw_input: Union[str, Dict[str, Any]] = "") -> str:
        data = _parse_input(raw_input)
        locations: List[str] = data.get("locations", [])
        mode: str = data.get("transport_mode", data.get("transportation_mode", "driving"))

        if len(locations) < 2:
            return "⚠️ Please provide at least two locations (origin and destination)."

        try:
            result = get_route(locations, mode, optimise=len(locations) > 2)
            return result.format_directions()
        except ValueError as exc:
            return f"⚠️ {exc}"
        except GeocodingError as exc:
            return f"⚠️ Could not locate one of the places: {exc}"
        except Exception as exc:
            log.error("RouteRetrieverTool error: %s", exc)
            return "⚠️ Could not retrieve directions. Please check the location names and try again."

    def _arun(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Async not supported.")


# ── Weather Tool ──────────────────────────────────────────────────────────────

class WeatherTool(BaseTool):
    """Fetch current weather or a multi-day forecast for a location."""

    name: str = "weather"
    description: str = (
        "Get current weather or a multi-day forecast for a city. "
        "Input: JSON with 'location' (city name) and optionally 'days' (integer, default 3, max 7). "
        "Returns temperature, conditions, and humidity."
    )

    def _run(self, raw_input: Union[str, Dict[str, Any]] = "") -> str:
        data = _parse_input(raw_input)
        location_name = data.get("location", "").strip()
        days = int(data.get("days", 3))

        if not location_name:
            return "⚠️ Please provide a location."

        try:
            coord = geocode(location_name)
        except GeocodingError as exc:
            return f"⚠️ {exc}"

        try:
            if days <= 1:
                current = get_current(coord.latitude, coord.longitude)
                return current.format()
            else:
                forecast = get_forecast(coord.latitude, coord.longitude, days=days)
                return format_forecast(forecast, location_name=location_name)
        except WeatherServiceError as exc:
            return f"⚠️ {exc}"
        except Exception as exc:
            log.error("WeatherTool error: %s", exc)
            return "⚠️ Could not retrieve weather data. Please try again."

    def _arun(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Async not supported.")


# ── Restaurant Tool ───────────────────────────────────────────────────────────

class RestaurantTool(BaseTool):
    """Find highly rated nearby restaurants for a given meal."""

    name: str = "restaurant_finder"
    description: str = (
        "Find top-rated nearby restaurants for a location and meal type. "
        "Input: JSON with 'location' (city or neighbourhood) and 'meal_type' "
        "('breakfast', 'lunch', or 'dinner'). "
        "Returns up to 3 restaurants with ratings and addresses."
    )

    def _run(self, raw_input: Union[str, Dict[str, Any]] = "") -> str:
        data = _parse_input(raw_input)
        location_name = data.get("location", "").strip()
        meal_type = data.get("meal_type", "lunch")

        if not location_name:
            return "⚠️ Please provide a location."

        try:
            coord = geocode(location_name)
        except GeocodingError as exc:
            return f"⚠️ {exc}"

        return search_restaurants(coord.latitude, coord.longitude, meal_type=meal_type)

    def _arun(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Async not supported.")


# ── Exported list of all tools ────────────────────────────────────────────────

def build_tools() -> list[BaseTool]:
    """Instantiate and return all available tools."""
    return [
        PlacesOfInterestTool(),
        RouteRetrieverTool(),
        WeatherTool(),
        RestaurantTool(),
    ]
