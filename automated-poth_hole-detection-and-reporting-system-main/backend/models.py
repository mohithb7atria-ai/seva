"""
Database layer for the pothole reporting backend.

Uses SQLite for zero-config local/demo deployment. Swap the connection
string in db.py-style fashion for PostgreSQL in production (schema is
plain SQL, no SQLite-specific syntax used).
"""

import sqlite3
import time
import uuid
from contextlib import contextmanager
from math import radians, sin, cos, sqrt, atan2
from typing import Optional

DB_PATH = "pothole_reports.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id TEXT PRIMARY KEY,
    vehicle_id TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    severity TEXT NOT NULL CHECK(severity IN ('minor','moderate','severe')),
    est_depth_cm REAL,
    confidence REAL,
    image_ref TEXT,
    timestamp REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'reported' CHECK(status IN ('reported','acknowledged','scheduled','repaired')),
    report_count INTEGER NOT NULL DEFAULT 1,
    address TEXT
);

CREATE INDEX IF NOT EXISTS idx_detections_location ON detections(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_detections_status ON detections(status);
"""


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript(SCHEMA)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lon points, in meters."""
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


# Two reports within this distance are treated as the same physical pothole.
DEDUP_RADIUS_M = 15.0


def find_nearby_detection(conn, lat: float, lon: float) -> Optional[sqlite3.Row]:
    """
    Find an existing open (not yet repaired) detection within DEDUP_RADIUS_M
    of the given coordinates. A tighter production implementation would use
    a spatial index (e.g. PostGIS ST_DWithin) rather than scanning all rows,
    but this is sufficient for a city-district-scale demo dataset.
    """
    rows = conn.execute(
        "SELECT * FROM detections WHERE status != 'repaired'"
    ).fetchall()
    for row in rows:
        if haversine_m(lat, lon, row["latitude"], row["longitude"]) <= DEDUP_RADIUS_M:
            return row
    return None


def insert_or_merge_detection(vehicle_id, lat, lon, severity, est_depth_cm,
                               confidence, image_ref, address=None) -> dict:
    """
    Insert a new pothole report, or — if a matching report already exists
    nearby — merge into it (bump the report_count and refresh severity to
    the worse of the two estimates). This is the duplicate-filtering step
    from the system architecture: many vehicles will drive over the same
    pothole and each would otherwise create a separate alert.
    """
    severity_rank = {"minor": 0, "moderate": 1, "severe": 2}
    now = time.time()

    with get_db() as conn:
        existing = find_nearby_detection(conn, lat, lon)
        if existing:
            merged_severity = severity if severity_rank[severity] > severity_rank[existing["severity"]] else existing["severity"]
            conn.execute(
                """UPDATE detections
                   SET report_count = report_count + 1,
                       severity = ?,
                       est_depth_cm = MAX(est_depth_cm, ?),
                       timestamp = ?
                   WHERE id = ?""",
                (merged_severity, est_depth_cm, now, existing["id"]),
            )
            row = conn.execute("SELECT * FROM detections WHERE id = ?", (existing["id"],)).fetchone()
            return {"merged": True, "detection": dict(row)}

        new_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO detections
               (id, vehicle_id, latitude, longitude, severity, est_depth_cm,
                confidence, image_ref, timestamp, status, report_count, address)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'reported', 1, ?)""",
            (new_id, vehicle_id, lat, lon, severity, est_depth_cm, confidence, image_ref, now, address),
        )
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (new_id,)).fetchone()
        return {"merged": False, "detection": dict(row)}
