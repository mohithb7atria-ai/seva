"""
Free geolocation helper.

The vehicle unit already has raw GPS coordinates from its GPS module (no API
needed for that part — it's a hardware fix, not a network call). What this
module adds is *reverse geocoding*: turning (lat, lon) into a human-readable
address/road name for the municipal dashboard, using OpenStreetMap's free
Nominatim API (no API key required, but has a strict usage policy — see
https://operations.osmfoundation.org/policies/nominatim/ — cache results and
respect the 1 req/sec rate limit in production, or self-host Nominatim for
higher volume).
"""

import time
import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "pothole-detection-system/1.0 (municipal-demo)"

_last_request_time = 0.0
_MIN_INTERVAL_S = 1.0  # respect Nominatim's usage policy rate limit
_cache: dict = {}


def reverse_geocode(lat: float, lon: float) -> str:
    """
    Returns a short human-readable address for the given coordinates.
    Falls back to raw coordinates if the API is unreachable or rate-limited,
    so a network hiccup never blocks a pothole report from being saved.
    """
    global _last_request_time
    cache_key = (round(lat, 5), round(lon, 5))
    if cache_key in _cache:
        return _cache[cache_key]

    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - elapsed)

    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 17},
            headers={"User-Agent": USER_AGENT},
            timeout=3,
        )
        _last_request_time = time.time()
        resp.raise_for_status()
        data = resp.json()
        address = data.get("display_name", f"{lat:.5f}, {lon:.5f}")
        # Trim to something dashboard-friendly rather than the full address blob.
        short_address = ", ".join(address.split(",")[:3])
        _cache[cache_key] = short_address
        return short_address
    except Exception:
        fallback = f"{lat:.5f}, {lon:.5f}"
        _cache[cache_key] = fallback
        return fallback
