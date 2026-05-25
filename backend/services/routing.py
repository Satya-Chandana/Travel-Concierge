"""
Routing service.

Features
--------
* Google Directions API for turn-by-turn navigation.
* Greedy nearest-neighbour TSP optimiser to reorder waypoints
  geographically, minimising total travel distance.
* Transport mode normalisation.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional

import requests

from config import settings
from services.geocoding import geocode, Coordinate, GeocodingError

log = logging.getLogger(__name__)


# ── Transport mode normalisation ────────────────────────────────────────────

_MODE_MAP: dict[str, str] = {
    "foot": "walking",
    "walk": "walking",
    "walking": "walking",
    "cycle": "bicycling",
    "bike": "bicycling",
    "bicycle": "bicycling",
    "bicycling": "bicycling",
    "bus": "transit",
    "train": "transit",
    "subway": "transit",
    "transit": "transit",
    "public": "transit",
    "car": "driving",
    "driving": "driving",
    "drive": "driving",
}


def normalise_mode(raw: str) -> str:
    """Map free-text transport mode to a Google Directions API mode string."""
    key = raw.lower().strip()
    for token, mode in _MODE_MAP.items():
        if token in key:
            return mode
    return "driving"


# ── TSP: greedy nearest-neighbour ───────────────────────────────────────────

def _haversine_km(a: Coordinate, b: Coordinate) -> float:
    """Great-circle distance in km between two coordinates."""
    R = 6371.0
    lat1, lon1 = math.radians(a.latitude), math.radians(a.longitude)
    lat2, lon2 = math.radians(b.latitude), math.radians(b.longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def optimise_waypoint_order(coords: list[Coordinate]) -> list[Coordinate]:
    """
    Greedy nearest-neighbour TSP on a list of coordinates.

    The first and last stops are kept fixed (origin / destination).
    Interior waypoints are reordered to minimise total Haversine distance.

    Complexity: O(n²) — acceptable for n ≤ 12 (the configured cap).
    """
    if len(coords) <= 2:
        return coords

    origin, destination = coords[0], coords[-1]
    interior = list(coords[1:-1])

    ordered = [origin]
    while interior:
        last = ordered[-1]
        nearest = min(interior, key=lambda c: _haversine_km(last, c))
        ordered.append(nearest)
        interior.remove(nearest)

    ordered.append(destination)
    return ordered


# ── Directions ───────────────────────────────────────────────────────────────

@dataclass
class RouteResult:
    steps: list[str]
    geocode_points: list[list[float]]   # [[lat, lon], ...]
    total_distance_km: float = 0.0
    total_duration_min: float = 0.0
    optimised: bool = False

    def format_directions(self) -> str:
        if not self.steps:
            return "Turn-by-turn directions are not available for this route."
        lines = [f"🧭 **Route ({self.total_distance_km:.1f} km, ~{int(self.total_duration_min)} min)**\n"]
        lines.extend(f"  {s}" for s in self.steps)
        if self.optimised:
            lines.append("\n_ℹ️ Waypoint order was optimised for shortest total distance._")
        return "\n".join(lines)


def get_route(
    location_names: list[str],
    transport_mode: str = "driving",
    *,
    city_context: str = "",
    optimise: bool = True,
) -> RouteResult:
    """
    Resolve location names → coordinates → Google Directions route.

    Parameters
    ----------
    location_names:
        At least two place names.  e.g. ``["Eiffel Tower", "Louvre Museum"]``.
    transport_mode:
        Free-text mode — will be normalised.
    city_context:
        If set, appended to bare location names for better geocoding
        accuracy  (e.g. ``"Paris"`` → ``"Louvre Museum near Paris"``).
    optimise:
        If True, reorder interior waypoints with the TSP greedy algorithm.

    Raises
    ------
    ValueError
        When fewer than two locations are provided.
    GeocodingError
        When a location cannot be resolved.
    RuntimeError
        When the Directions API call fails.
    """
    if len(location_names) < 2:
        raise ValueError("At least two locations are required.")

    if len(location_names) > settings.tsp_max_waypoints:
        location_names = location_names[: settings.tsp_max_waypoints]
        log.warning("Waypoint list truncated to %d stops.", settings.tsp_max_waypoints)

    mode = normalise_mode(transport_mode)

    def enrich(name: str) -> str:
        if "," not in name and city_context:
            return f"{name} near {city_context}"
        return name

    coords = [geocode(enrich(n)) for n in location_names]

    if optimise and len(coords) > 2:
        coords = optimise_waypoint_order(coords)
        optimised = True
    else:
        optimised = False

    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")

    params: dict = {
        "origin": f"{coords[0].latitude},{coords[0].longitude}",
        "destination": f"{coords[-1].latitude},{coords[-1].longitude}",
        "mode": mode,
        "key": settings.google_api_key,
    }
    if len(coords) > 2:
        waypoints = "|".join(f"{c.latitude},{c.longitude}" for c in coords[1:-1])
        params["waypoints"] = waypoints

    resp = requests.get(
        "https://maps.googleapis.com/maps/api/directions/json",
        params=params,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "OK":
        raise RuntimeError(f"Directions API error: {data.get('status')} — {data.get('error_message', '')}")

    leg = data["routes"][0]["legs"][0]
    steps_raw = []
    geocode_pts: list[list[float]] = []

    for step in data["routes"][0]["legs"][0]["steps"] if len(coords) == 2 else _all_steps(data):
        clean = re.sub(r"<[^>]+>", "", step.get("html_instructions", ""))
        steps_raw.append(f"- {clean}")
        sl = step.get("start_location", {})
        if sl:
            geocode_pts.append([sl["lat"], sl["lng"]])

    end = data["routes"][0]["legs"][-1]["end_location"]
    geocode_pts.append([end["lat"], end["lng"]])

    total_dist = sum(l["distance"]["value"] for l in data["routes"][0]["legs"]) / 1000
    total_dur = sum(l["duration"]["value"] for l in data["routes"][0]["legs"]) / 60

    return RouteResult(
        steps=steps_raw,
        geocode_points=geocode_pts,
        total_distance_km=round(total_dist, 1),
        total_duration_min=round(total_dur, 1),
        optimised=optimised,
    )


def _all_steps(data: dict) -> list[dict]:
    steps = []
    for leg in data["routes"][0]["legs"]:
        steps.extend(leg["steps"])
    return steps
