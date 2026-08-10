"""
weather.py — Weather fetcher using wttr.in (no API key needed).
Caches result in memory for 30 minutes.
"""
import asyncio
import logging
import os
import time
from typing import Optional

log = logging.getLogger("weather")

_cache: Optional[dict] = None
_cache_time: float = 0
CACHE_TTL = 1800  # 30 minutes


async def get_weather(city: Optional[str] = None) -> dict:
    """
    Fetch current weather for city from wttr.in JSON API.
    Returns dict with temp_c, condition, humidity, wind_kmh, city.
    Uses in-memory cache (30min TTL).
    """
    global _cache, _cache_time

    city = city or os.environ.get("WEATHER_CITY", "Delhi")
    now = time.time()

    if _cache and (now - _cache_time) < CACHE_TTL:
        log.debug("Weather: returning cached result")
        return _cache

    result = await asyncio.to_thread(_fetch_weather_sync, city)
    if result:
        _cache = result
        _cache_time = now
    return result or {"error": f"Could not fetch weather for {city}", "city": city}


def _fetch_weather_sync(city: str) -> Optional[dict]:
    """Synchronous wttr.in fetch."""
    import httpx
    import urllib.parse
    try:
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        with httpx.Client(timeout=8) as client:
            resp = client.get(url, headers={"User-Agent": "MrMeeseeks/1.0"})
            resp.raise_for_status()
            data = resp.json()

        current = data["current_condition"][0]
        area = data.get("nearest_area", [{}])[0]
        area_name = area.get("areaName", [{}])[0].get("value", city)

        return {
            "city": area_name,
            "temp_c": int(current["temp_C"]),
            "feels_like_c": int(current["FeelsLikeC"]),
            "condition": current["weatherDesc"][0]["value"],
            "humidity": int(current["humidity"]),
            "wind_kmh": int(current["windspeedKmph"]),
        }
    except Exception as e:
        log.warning(f"Weather fetch failed for {city}: {e}")
        return None


def format_weather_for_brief(w: dict) -> str:
    """Format weather dict into a natural spoken sentence."""
    if "error" in w:
        return "I couldn't get the weather right now."
    return (f"Weather in {w['city']} is {w['temp_c']} degrees Celsius, {w['condition'].lower()}. "
            f"Feels like {w['feels_like_c']} degrees. Humidity {w['humidity']} percent, "
            f"wind at {w['wind_kmh']} kilometres per hour.")
