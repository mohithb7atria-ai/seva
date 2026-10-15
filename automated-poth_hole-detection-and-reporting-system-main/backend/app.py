"""
Pothole Reporting Backend
==========================

REST API that receives detection records from onboard vehicle units,
deduplicates reports of the same physical pothole, reverse-geocodes
locations, and serves aggregated data to the municipal dashboard.

Run:
    pip install -r requirements.txt
    python app.py
    # -> http://localhost:5000
"""

import os
import threading
from flask import Flask, request, jsonify, send_from_directory

from models import init_db, get_db, insert_or_merge_detection
from geolocation import reverse_geocode

app = Flask(__name__)

DASHBOARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard"))


@app.route("/")
@app.route("/dashboard")
def index():
    return send_from_directory(DASHBOARD_DIR, "dashboard.html")



@app.after_request
def add_cors_headers(response):
    # The dashboard is served separately (static HTML/JS) and needs
    # cross-origin access to this API. Kept dependency-free (no flask-cors)
    # so the backend has zero extra install requirements beyond Flask itself.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PATCH, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def cors_preflight(_any):
    return "", 204


ALLOWED_STATUSES = {"reported", "acknowledged", "scheduled", "repaired"}
ALLOWED_SEVERITIES = {"minor", "moderate", "severe"}


def background_geocode(detection_id, lat, lon):
    """Fetches the address in the background so the API doesn't block."""
    try:
        address = reverse_geocode(lat, lon)
        if address:
            with get_db() as conn:
                conn.execute(
                    "UPDATE detections SET address = ? WHERE id = ?", 
                    (address, detection_id)
                )
                conn.commit()
    except Exception as e:
        # In a production app, log this error
        print(f"Geocoding failed for detection {detection_id}: {e}")


@app.route("/api/detections", methods=["POST"])
def submit_detection():
    """
    Body:
    {
      "vehicle_id": "bus-042",
      "latitude": 8.5241,
      "longitude": 76.9366,
      "severity": "moderate",
      "est_depth_cm": 3.2,
      "confidence": 0.87,
      "image_ref": "s3://bucket/frame123.jpg"   # optional
    }
    """
    data = request.get_json(force=True, silent=True) or {}

    required = ["vehicle_id", "latitude", "longitude", "severity"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    if data["severity"] not in ALLOWED_SEVERITIES:
        return jsonify({"error": f"severity must be one of {sorted(ALLOWED_SEVERITIES)}"}), 400

    try:
        lat = float(data["latitude"])
        lon = float(data["longitude"])
    except (TypeError, ValueError):
        return jsonify({"error": "latitude/longitude must be numeric"}), 400

    if not (-90.0 <= lat <= 90.0):
        return jsonify({"error": "latitude must be between -90 and 90"}), 400

    if not (-180.0 <= lon <= 180.0):
        return jsonify({"error": "longitude must be between -180 and 180"}), 400

    # Insert the detection without blocking for the geocoder
    result = insert_or_merge_detection(
        vehicle_id=data["vehicle_id"],
        lat=lat,
        lon=lon,
        severity=data["severity"],
        est_depth_cm=float(data.get("est_depth_cm", 0)),
        confidence=float(data.get("confidence", 0)),
        image_ref=data.get("image_ref"),
        address=None,  # Will be populated in the background
    )

    # Trigger background thread for geocoding if we have a valid ID and address is not yet populated
    detection_data = result.get("detection") or {}
    detection_id = detection_data.get("id")
    if detection_id and not detection_data.get("address"):
        threading.Thread(
            target=background_geocode, 
            args=(detection_id, lat, lon), 
            daemon=True
        ).start()

    status_code = 200 if result.get("merged") else 201
    return jsonify(result), status_code


@app.route("/api/detections", methods=["GET"])
def list_detections():
    """
    Query params:
      severity=minor|moderate|severe   filter by severity
      status=reported|acknowledged...  filter by status
      limit=100                        max rows (default 200)
      offset=0                         pagination offset
    """
    severity = request.args.get("severity")
    status = request.args.get("status")

    # Safely parse limit and offset to avoid 500 errors
    try:
        limit = min(int(request.args.get("limit", 200)), 1000)
    except ValueError:
        return jsonify({"error": "limit must be a valid integer"}), 400

    try:
        offset = int(request.args.get("offset", 0))
    except ValueError:
        return jsonify({"error": "offset must be a valid integer"}), 400

    query = "SELECT * FROM detections WHERE 1=1"
    params = []
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    if status:
        query += " AND status = ?"
        params.append(status)
        
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()

    return jsonify([dict(r) for r in rows])


@app.route("/api/detections/<detection_id>", methods=["GET"])
def get_detection(detection_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (detection_id,)).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(dict(row))


@app.route("/api/detections/<detection_id>", methods=["PATCH"])
def update_detection(detection_id):
    """Body: {"status": "acknowledged"} — used by municipal officials via the dashboard."""
    data = request.get_json(force=True, silent=True) or {}
    new_status = data.get("status")
    
    if new_status not in ALLOWED_STATUSES:
        return jsonify({"error": f"status must be one of {sorted(ALLOWED_STATUSES)}"}), 400

    with get_db() as conn:
        cur = conn.execute(
            "UPDATE detections SET status = ? WHERE id = ?", 
            (new_status, detection_id)
        )
        conn.commit()  # Explicitly commit the transaction
        
        if cur.rowcount == 0:
            return jsonify({"error": "not found"}), 404
            
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (detection_id,)).fetchone()

    return jsonify(dict(row))


@app.route("/api/stats", methods=["GET"])
def stats():
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM detections").fetchone()["c"]
        by_severity = conn.execute(
            "SELECT severity, COUNT(*) c FROM detections WHERE status != 'repaired' GROUP BY severity"
        ).fetchall()
        by_status = conn.execute(
            "SELECT status, COUNT(*) c FROM detections GROUP BY status"
        ).fetchall()
        vehicles = conn.execute(
            "SELECT COUNT(DISTINCT vehicle_id) c FROM detections"
        ).fetchone()["c"]

    return jsonify({
        "total_reports": total,
        "active_vehicles": vehicles,
        "open_by_severity": {r["severity"]: r["c"] for r in by_severity},
        "by_status": {r["status"]: r["c"] for r in by_status},
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Offline Local Image Detection (YOLOv8 CPU Inference)
# ---------------------------------------------------------------------------
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB limit
SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "detection"))

_yolo_model = None
_model_lock = threading.Lock()


def get_yolo_model():
    """Lazy-loads and caches the YOLOv8 pothole model instance in memory."""
    global _yolo_model
    if _yolo_model is None:
        with _model_lock:
            if _yolo_model is None:
                candidates = [
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pothole_best.pt")),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pothole_best.pt")),
                ]
                model_path = next((p for p in candidates if os.path.exists(p)), None)
                if not model_path:
                    raise FileNotFoundError(
                        f"pothole_best.pt not found in candidate paths: {candidates}"
                    )
                from ultralytics import YOLO
                _yolo_model = YOLO(model_path)
    return _yolo_model


def classify_2d_severity(area_px: int, frame_area_px: int) -> str:
    """Provisional heuristic based on 2D area ratio (no stereo depth)."""
    ratio = area_px / max(frame_area_px, 1)
    if ratio >= 0.04:
        return "severe"
    elif ratio >= 0.015:
        return "moderate"
    return "minor"


@app.route("/api/detect-image", methods=["POST"])
def detect_image():
    """
    Offline local inference endpoint.
    Accepts multipart/form-data:
      - image: image file (JPEG, PNG, WEBP, max 10MB)
      - conf: optional float confidence threshold (default 0.40, range 0.10 to 0.90)
    Returns:
      JSON with detection count, bounding boxes, confidence, provisional severity,
      annotated image in base64, and explicit depth/calibration disclaimers.
      Does NOT insert into or modify the database.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Please attach an image in the 'image' field."}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"error": "Empty filename or no file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        return jsonify({
            "error": f"Unsupported file extension '{ext}'. Allowed formats: JPEG, PNG, WEBP."
        }), 400

    image_bytes = file.read()
    if len(image_bytes) == 0:
        return jsonify({"error": "Uploaded image file is empty (0 bytes)."}), 400

    if len(image_bytes) > MAX_IMAGE_SIZE:
        return jsonify({
            "error": f"Image file size ({len(image_bytes)} bytes) exceeds maximum limit of 10MB."
        }), 400

    try:
        conf_val = float(request.form.get("conf", 0.40))
        conf_val = max(0.10, min(0.90, conf_val))
    except (TypeError, ValueError):
        conf_val = 0.40

    import base64
    import cv2
    import numpy as np

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode uploaded image. Invalid or corrupted image."}), 400

    img_h, img_w = img.shape[:2]
    frame_area = img_h * img_w

    try:
        model = get_yolo_model()
        results = model.predict(img, conf=conf_val, device="cpu")[0]
    except Exception as e:
        return jsonify({"error": f"YOLO inference error: {str(e)}"}), 500

    detections = []
    for box in results.boxes:
        xyxy = [int(v) for v in box.xyxy[0].tolist()]
        box_w = max(0, xyxy[2] - xyxy[0])
        box_h = max(0, xyxy[3] - xyxy[1])
        area_px = box_w * box_h
        area_ratio = round(area_px / max(frame_area, 1), 4)
        confidence = round(float(box.conf[0].item()), 4)
        cls_id = int(box.cls[0].item())
        class_name = model.names.get(cls_id, "pothole") if hasattr(model, "names") else "pothole"
        severity = classify_2d_severity(area_px, frame_area)

        detections.append({
            "class_name": class_name,
            "confidence": confidence,
            "bbox": xyxy,
            "severity": severity,
            "area_ratio": area_ratio,
        })

    annotated_bgr = results.plot()
    _, buffer = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64_str = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "count": len(detections),
        "detections": detections,
        "annotated_image": f"data:image/jpeg;base64,{b64_str}",
        "depth_status": "Depth: unavailable — calibration required",
        "severity_disclaimer": "Severity is a provisional estimate based on 2D image area; depth measurement requires calibrated stereo cameras."
    })


@app.route("/api/sample-images/<filename>", methods=["GET"])
def get_sample_image(filename):
    """Safely serves existing sample road images for testing."""
    if filename in {"left.webp", "right.webp"}:
        return send_from_directory(SAMPLE_DIR, filename)
    return jsonify({"error": "Sample image not found"}), 404



if __name__ == "__main__":
    init_db()
    import os
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)