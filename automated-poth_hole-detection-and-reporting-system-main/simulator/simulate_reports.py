"""
Fleet simulator for demoing the pothole reporting pipeline without real
hardware. Generates plausible detections from a small fleet of virtual buses
and cars driving fixed routes around a city, and POSTs them to the backend
API — exactly as a real onboard unit would.

Run (with backend already running on :5000):
    python simulate_reports.py --count 40 --interval 0.3
"""

import argparse
import random
import time

import requests

API_URL = "http://localhost:5000/api/detections"

# A handful of fixed "known pothole" locations so repeated passes generate
# realistic duplicate-merge behavior, plus randomized locations along routes
# for coverage of genuinely new potholes.
KNOWN_POTHOLE_HOTSPOTS = [
    (8.5241, 76.9366),   # busy junction
    (8.4855, 76.9492),   # near a market road
    (8.5459, 76.9109),   # highway stretch
]

CITY_BOUNDS = {"lat_min": 8.45, "lat_max": 8.58, "lon_min": 76.88, "lon_max": 76.99}

VEHICLES = [f"bus-{i:03d}" for i in range(1, 6)] + [f"car-{i:03d}" for i in range(100, 106)]

SEVERITIES = ["minor", "moderate", "severe"]
SEVERITY_WEIGHTS = [0.5, 0.35, 0.15]  # most potholes are minor; severe ones are rarer


def jitter(lat, lon, meters=8):
    """Nudge coordinates slightly, simulating GPS noise / slightly different lane position."""
    deg_per_m = 1 / 111_000
    return (
        lat + random.uniform(-meters, meters) * deg_per_m,
        lon + random.uniform(-meters, meters) * deg_per_m,
    )


def random_location():
    # 60% chance of a re-pass over a known hotspot (demonstrates dedup),
    # 40% chance of a brand-new location (demonstrates fresh reports).
    if random.random() < 0.6:
        lat, lon = random.choice(KNOWN_POTHOLE_HOTSPOTS)
        return jitter(lat, lon, meters=6)
    return (
        random.uniform(CITY_BOUNDS["lat_min"], CITY_BOUNDS["lat_max"]),
        random.uniform(CITY_BOUNDS["lon_min"], CITY_BOUNDS["lon_max"]),
    )


def generate_report():
    lat, lon = random_location()
    severity = random.choices(SEVERITIES, weights=SEVERITY_WEIGHTS)[0]
    depth_by_severity = {"minor": (0.5, 2.4), "moderate": (2.5, 4.9), "severe": (5.0, 9.0)}
    depth_range = depth_by_severity[severity]

    return {
        "vehicle_id": random.choice(VEHICLES),
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "severity": severity,
        "est_depth_cm": round(random.uniform(*depth_range), 1),
        "confidence": round(random.uniform(0.6, 0.98), 2),
        "image_ref": None,
    }


def run(count: int, interval: float, api_url: str):
    print(f"Simulating {count} vehicle detections -> {api_url}")
    ok, merged, failed = 0, 0, 0

    for i in range(count):
        report = generate_report()
        try:
            resp = requests.post(api_url, json=report, timeout=3)
            resp.raise_for_status()
            result = resp.json()
            was_merged = result.get("merged", False)
            merged += int(was_merged)
            ok += 1
            tag = "MERGED " if was_merged else "NEW    "
            print(f"[{i+1}/{count}] {tag} {report['vehicle_id']:>9} "
                  f"{report['severity']:<8} depth={report['est_depth_cm']}cm "
                  f"@ ({report['latitude']}, {report['longitude']})")
        except requests.RequestException as e:
            failed += 1
            print(f"[{i+1}/{count}] FAILED to submit report: {e}")

        time.sleep(interval)

    print(f"\nDone. {ok} submitted ({merged} merged into existing reports), {failed} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a fleet of vehicles reporting potholes.")
    parser.add_argument("--count", type=int, default=40, help="Number of detections to simulate")
    parser.add_argument("--interval", type=float, default=0.3, help="Seconds between submissions")
    parser.add_argument("--api-url", default=API_URL, help="Backend ingestion endpoint")
    args = parser.parse_args()

    run(args.count, args.interval, args.api_url)
