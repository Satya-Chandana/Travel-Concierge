"""
Weather service — OpenWeatherMap wrapper.

Provides current conditions and multi-day forecasts.
All functions return clean Python objects, never raw API dicts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests

from config import settings

log = logging.getLogger(__name__)

_CONDITION_EMOJI: dict[str, str] = {
    "clear sky": "☀️",
    "few clouds": "⛅",
    "scattered clouds": "⛅",
    "broken clouds": "☁️",
    "overcast clouds": "☁️",
    "shower rain": "🌧️",
    "light rain": "🌦️",
    "moderate rain": "🌧️",
    "heavy rain": "🌧️",
    "rain": "🌧️",
    "thunderstorm": "🌩️",
    "snow": "❄️",
    "light snow": "🌨️",
    "mist": "🌫️",
    "fog": "🌫️",
    "haze": "🌫️",
    "drizzle": "🌦️",
}


def _emoji(description: str) -> str:
    return _CONDITION_EMOJI.get(description.lower(), "🌍")


@dataclass(frozen=True)
class CurrentWeather:
    city: str
    description: str
    temp_c: float
    feels_like_c: float
    humidity_pct: int
    wind_kph: float

    def format(self) -> str:
        emoji = _emoji(self.description)
        return (
            f"{emoji} **{self.city}** — {self.description.capitalize()}\n"
            f"🌡️ {self.temp_c:.1f}°C (feels like {self.feels_like_c:.1f}°C)  "
            f"💧 Humidity: {self.humidity_pct}%  "
            f"💨 Wind: {self.wind_kph:.1f} km/h"
        )


@dataclass(frozen=True)
class ForecastDay:
    date: str
    description: str
    temp_max_c: float
    temp_min_c: float

    def format(self) -> str:
        emoji = _emoji(self.description)
        return (
            f"{self.date} — {emoji} {self.description.capitalize()}\n"
            f"  🌡️ High: {self.temp_max_c:.1f}°C  |  ❄️ Low: {self.temp_min_c:.1f}°C"
        )


class WeatherServiceError(RuntimeError):
    pass


def get_current(lat: float, lon: float) -> CurrentWeather:
    """Fetch current weather conditions."""
    if not settings.openweather_api_key:
        raise WeatherServiceError("OPENWEATHER_API_KEY is not configured.")

    resp = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "lat": lat,
            "lon": lon,
            "units": "metric",
            "appid": settings.openweather_api_key,
        },
        timeout=6,
    )
    resp.raise_for_status()
    data = resp.json()

    return CurrentWeather(
        city=data.get("name", "your location"),
        description=data["weather"][0]["description"],
        temp_c=data["main"]["temp"],
        feels_like_c=data["main"]["feels_like"],
        humidity_pct=data["main"]["humidity"],
        wind_kph=data["wind"]["speed"] * 3.6,
    )


def get_forecast(
    lat: float,
    lon: float,
    days: int = settings.weather_default_forecast_days,
) -> list[ForecastDay]:
    """
    Fetch a daily forecast using OpenWeatherMap One Call API.

    Days are capped at ``settings.weather_max_forecast_days``.
    """
    if not settings.openweather_api_key:
        raise WeatherServiceError("OPENWEATHER_API_KEY is not configured.")

    days = min(days, settings.weather_max_forecast_days)

    resp = requests.get(
        "https://api.openweathermap.org/data/2.5/onecall",
        params={
            "lat": lat,
            "lon": lon,
            "exclude": "current,minutely,hourly,alerts",
            "units": "metric",
            "appid": settings.openweather_api_key,
        },
        timeout=6,
    )
    resp.raise_for_status()
    data = resp.json()

    result: list[ForecastDay] = []
    for day in data.get("daily", [])[:days]:
        date_str = datetime.utcfromtimestamp(day["dt"]).strftime("%A, %b %-d")
        result.append(
            ForecastDay(
                date=date_str,
                description=day["weather"][0]["description"],
                temp_max_c=day["temp"]["max"],
                temp_min_c=day["temp"]["min"],
            )
        )
    return result


def format_forecast(forecast: list[ForecastDay], location_name: str = "") -> str:
    header = f"📅 {len(forecast)}-Day Forecast"
    if location_name:
        header += f" for **{location_name.title()}**"
    lines = [header, ""]
    lines.extend(d.format() for d in forecast)
    return "\n".join(lines)
