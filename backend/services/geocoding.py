"""
Geocoding service.

Strategy
--------
1. Google Geocoding API  (fast, accurate, paid)
2. Nominatim / OpenStreetMap  (free, slower)
3. Hard-coded known cities  (last resort, never fails for major cities)

All public functions raise ``GeocodingError`` on complete failure so
callers can surface a clean error rather than catching generic exceptions.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Optional

import requests
from geopy.geocoders import Nominatim
from geopy.location import Location

from config import settings

log = logging.getLogger(__name__)

# ── Fallback table for ultra-common cities ──────────────────────────────────
_KNOWN: dict[str, tuple[float, float]] = {
    "chicago": (41.8781, -87.6298),
    "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "paris": (48.8566, 2.3522),
    "london": (51.5074, -0.1278),
    "tokyo": (35.6895, 139.6917),
    "dubai": (25.2048, 55.2708),
    "sydney": (-33.8688, 151.2093),
    "rome": (41.9028, 12.4964),
    "barcelona": (41.3851, 2.1734),
}


class GeocodingError(RuntimeError):
    """Raised when a location cannot be resolved by any strategy."""


@dataclass(frozen=True)
class Coordinate:
    latitude: float
    longitude: float
    address: str = ""

    def as_tuple(self) -> tuple[float, float]:
        return (self.latitude, self.longitude)

    # Geopy-compatible shims so existing code that calls .latitude / .longitude works
    @property
    def lat(self) -> float:
        return self.latitude

    @property
    def lon(self) -> float:
        return self.longitude


def _google_geocode(location: str) -> Optional[Coordinate]:
    """Try Google Geocoding API. Returns None on any failure."""
    if not settings.google_api_key:
        return None
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": location, "key": settings.google_api_key},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("results"):
            r = data["results"][0]
            geo = r["geometry"]["location"]
            return Coordinate(
                latitude=geo["lat"],
                longitude=geo["lng"],
                address=r.get("formatted_address", location),
            )
    except Exception as exc:
        log.debug("Google geocode failed for %r: %s", location, exc)
    return None


def _nominatim_geocode(location: str) -> Optional[Coordinate]:
    """Try Nominatim. Returns None on any failure."""
    try:
        geolocator = Nominatim(user_agent="travel-concierge-v2")
        loc: Optional[Location] = geolocator.geocode(location)
        if loc:
            return Coordinate(latitude=loc.latitude, longitude=loc.longitude, address=str(loc))
    except Exception as exc:
        log.debug("Nominatim geocode failed for %r: %s", location, exc)
    return None


def geocode(location: str, *, retries: int = 2) -> Coordinate:
    """
    Resolve a place name to coordinates.

    Parameters
    ----------
    location:
        Human-readable place name, e.g. ``"Eiffel Tower, Paris"``.
    retries:
        Number of extra attempts on transient network errors.

    Returns
    -------
    Coordinate

    Raises
    ------
    GeocodingError
        When all strategies are exhausted.
    """
    location = location.strip()
    if not location:
        raise GeocodingError("Empty location string provided.")

    for attempt in range(retries + 1):
        coord = _google_geocode(location) or _nominatim_geocode(location)
        if coord:
            return coord
        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))

    # Hard-coded fallback
    key = location.lower().strip()
    for known_key, (lat, lon) in _KNOWN.items():
        if known_key in key:
            log.warning("Using hard-coded coords for %r", location)
            return Coordinate(latitude=lat, longitude=lon, address=location)

    raise GeocodingError(
        f"Could not resolve coordinates for '{location}'. "
        "Please check the spelling or try a more specific name."
    )


def reverse_geocode(lat: float, lon: float) -> str:
    """Return a human-readable city name for coordinates."""
    try:
        geolocator = Nominatim(user_agent="travel-concierge-v2")
        loc = geolocator.reverse((lat, lon), exactly_one=True)
        if loc:
            addr = loc.raw.get("address", {})
            return (
                addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("state")
                or "Unknown Location"
            )
    except Exception as exc:
        log.debug("Reverse geocode failed: %s", exc)
    return "Unknown Location"
