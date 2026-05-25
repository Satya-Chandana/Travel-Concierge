"""
Places service — wraps Google Places Nearby Search.

All public functions return plain Python objects (dicts / lists) so
the rest of the codebase is not coupled to the HTTP client.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import requests

from config import settings

log = logging.getLogger(__name__)

# ── Interest keyword normalisation ──────────────────────────────────────────
_INTEREST_MAP: dict[str, str] = {
    "attractions": "famous places",
    "tourist attractions": "famous landmarks",
    "museums": "museum",
    "temples": "temple",
    "places to visit": "landmarks",
    "sightseeing": "top sights",
    "monuments": "monument",
    "beaches": "beach",
    "food": "restaurant",
    "shopping": "shopping mall",
    "nightlife": "bar nightclub",
    "parks": "park",
    "hiking": "hiking trail",
    "art": "art gallery",
}


def normalise_interest(raw: str) -> str:
    return _INTEREST_MAP.get(raw.lower().strip(), raw.strip())


@dataclass
class Place:
    name: str
    address: str
    rating: float
    user_ratings_total: int
    latitude: float
    longitude: float
    types: list[str] = field(default_factory=list)
    place_id: str = ""
    price_level: Optional[int] = None   # 0–4, None if unknown

    def format_line(self, idx: int) -> str:
        price = ("$" * self.price_level) if self.price_level else ""
        return (
            f"{idx}. 📍 {self.name}  —  {self.rating}⭐ ({self.user_ratings_total:,} reviews)"
            + (f"  {price}" if price else "")
            + f"\n   🗺️  {self.address}"
        )


def search_nearby(
    lat: float,
    lon: float,
    keyword: str,
    radius_m: int = settings.places_default_radius_m,
    max_results: int = settings.places_max_results,
) -> list[Place]:
    """
    Search Google Places Nearby Search and return a ranked list of Places.

    Sorted by (rating DESC, total_reviews DESC) — same signal a traveller
    would use manually.

    Raises
    ------
    RuntimeError
        When the API key is missing or the HTTP request fails.
    """
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")

    keyword = normalise_interest(keyword)
    params = {
        "location": f"{lat},{lon}",
        "radius": radius_m,
        "keyword": keyword,
        "key": settings.google_api_key,
    }
    resp = requests.get(
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
        params=params,
        timeout=8,
    )
    resp.raise_for_status()
    data = resp.json()

    raw_places = data.get("results", [])
    places: list[Place] = []
    for r in raw_places:
        geo = r.get("geometry", {}).get("location", {})
        places.append(
            Place(
                name=r.get("name", "Unnamed"),
                address=r.get("vicinity", "Address not available"),
                rating=float(r.get("rating", 0)),
                user_ratings_total=int(r.get("user_ratings_total", 0)),
                latitude=geo.get("lat", lat),
                longitude=geo.get("lng", lon),
                types=r.get("types", []),
                place_id=r.get("place_id", ""),
                price_level=r.get("price_level"),
            )
        )

    places.sort(key=lambda p: (p.rating, p.user_ratings_total), reverse=True)
    return places[:max_results]


def format_places(places: list[Place]) -> str:
    if not places:
        return "No places found for that interest."
    return "\n\n".join(p.format_line(i + 1) for i, p in enumerate(places))


def search_restaurants(
    lat: float,
    lon: float,
    meal_type: str = "lunch",
    radius_m: int = 800,
    max_results: int = 3,
) -> str:
    """Return a formatted string of top nearby restaurants."""
    keyword_map = {
        "lunch": "cafe lunch bistro",
        "dinner": "dinner restaurant fine dining",
        "breakfast": "breakfast cafe brunch",
    }
    keyword = keyword_map.get(meal_type.lower(), "restaurant")
    try:
        places = search_nearby(lat, lon, keyword, radius_m=radius_m, max_results=max_results)
        if not places:
            return "No restaurants found nearby."
        return "\n\n".join(
            f"🍽️ {p.name} — {p.rating}⭐\n📍 {p.address}" for p in places
        )
    except Exception as exc:
        log.error("Restaurant search failed: %s", exc)
        return "⚠ Could not fetch nearby restaurants."
