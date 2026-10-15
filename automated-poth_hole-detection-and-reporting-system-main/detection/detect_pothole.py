"""
Pothole Detection + Stereo Depth Estimation (YOLO-only)
=========================================================

Given a pair of stereo road-surface images (left/right camera), this module:
  1. Detects potholes using a YOLOv8 model fine-tuned on a pothole dataset.
  2. Computes a stereo disparity map to estimate real-world depth.
  3. Classifies each detected pothole's severity (minor / moderate / severe)
     using its estimated depth and surface area.

This requires actual trained weights (a .pt file fine-tuned on pothole
images) — it will NOT detect potholes out of the box with a stock COCO
checkpoint, since "pothole" isn't a COCO class. See ../training/ to train
your own weights from a public pothole dataset.

Run standalone:
    python detect_pothole.py --left left.jpg --right right.jpg --weights best.pt
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PotholeDetection:
    bbox: List[int]            # [x, y, w, h] in pixels
    confidence: float          # 0-1 detection confidence
    area_px: int                # pixel area of the detected region
    est_depth_cm: float         # provisional uncalibrated depth from stereo disparity
    severity: str                # provisional severity: "minor" | "moderate" | "severe"
    frame_timestamp: float
    depth_provisional: bool = True  # uncalibrated heuristic; requires cv2.stereoCalibrate

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

def classify_severity(depth_cm: float, area_px: int, frame_area_px: int) -> str:
    """
    Classify pothole severity using depth (primary signal, per the stereo
    disparity estimate) and relative surface area (secondary signal).

    Thresholds are placeholders — calibrate against ground-truth measurements
    (e.g. IRC/municipal pothole severity standards) before real deployment.
    """
    area_ratio = area_px / max(frame_area_px, 1)

    if depth_cm >= 5.0 or area_ratio >= 0.04:
        return "severe"
    elif depth_cm >= 2.5 or area_ratio >= 0.015:
        return "moderate"
    else:
        return "minor"


# ---------------------------------------------------------------------------
# Stereo depth estimation
# ---------------------------------------------------------------------------

class StereoDepthEstimator:
    """
    Wraps OpenCV's semi-global block matching (SGBM) to turn a rectified
    stereo pair into a disparity map, then converts disparity -> depth
    using the standard pinhole stereo relation:

        depth = (focal_length_px * baseline_m) / disparity_px

    `focal_length_px` and `baseline_m` must be calibrated for the specific
    camera rig (checkerboard calibration via cv2.calibrateCamera /
    cv2.stereoCalibrate). Defaults below are placeholders for a typical
    dashboard stereo rig (~6cm baseline, ~700px focal length at 720p) and
    MUST be replaced with your rig's real calibration before trusting
    absolute depth numbers.
    """

    def __init__(self, focal_length_px: float = 700.0, baseline_m: float = 0.06):
        self.focal_length_px = focal_length_px
        self.baseline_m = baseline_m
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 6,   # must be divisible by 16
            blockSize=7,
            P1=8 * 3 * 7 ** 2,
            P2=32 * 3 * 7 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )

    def compute_disparity(self, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        disparity = self.matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        return disparity

    def depth_at_region(self, disparity: np.ndarray, bbox: List[int]) -> float:
        """Estimate depth (in cm) of a bbox region using median disparity inside it."""
        x, y, w, h = bbox
        region = disparity[y:y + h, x:x + w]
        valid = region[region > 0]
        if valid.size == 0:
            return 0.0
        # A pothole is a *depression* relative to the surrounding road plane,
        # not raw depth-from-camera. In production, fit a plane to the road
        # surface around the region and take the deviation of the pothole
        # region from that plane. Here we approximate depth variation across
        # the region as a stand-in for depression depth.
        depth_variation_m = float(np.std(valid)) * self.baseline_m
        return round(depth_variation_m * 100, 2)  # -> cm


# ---------------------------------------------------------------------------
# YOLO detector
# ---------------------------------------------------------------------------

class YoloPotholeDetector:
    """
    Requires a checkpoint fine-tuned on a pothole dataset — see
    ../training/train_yolo.py and ../training/dataset_setup.md to produce one.
    A stock yolov8n.pt/yolov8s.pt COCO checkpoint will NOT detect potholes;
    "pothole" is not one of COCO's 80 classes.
    """

    def __init__(self, weights_path: str, conf_threshold: float = 0.4):
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame_bgr: np.ndarray) -> List[dict]:
        results = self.model.predict(frame_bgr, conf=self.conf_threshold, verbose=False)[0]
        candidates = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            candidates.append({
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "confidence": float(box.conf[0]),
                "area_px": int((x2 - x1) * (y2 - y1)),
            })
        return candidates


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_stereo_pair(
    left_path: str,
    right_path: str,
    weights_path: str,
    conf_threshold: float = 0.4,
) -> List[PotholeDetection]:
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    if left is None or right is None:
        raise FileNotFoundError("Could not read one or both stereo images.")

    detector = YoloPotholeDetector(weights_path, conf_threshold=conf_threshold)
    raw_detections = detector.detect(left)

    depth_estimator = StereoDepthEstimator()
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    disparity = depth_estimator.compute_disparity(left_gray, right_gray)

    frame_area = left.shape[0] * left.shape[1]
    now = time.time()
    results = []
    for det in raw_detections:
        depth_cm = depth_estimator.depth_at_region(disparity, det["bbox"])
        severity = classify_severity(depth_cm, det["area_px"], frame_area)
        results.append(PotholeDetection(
            bbox=det["bbox"],
            confidence=round(det["confidence"], 2),
            area_px=det["area_px"],
            est_depth_cm=depth_cm,
            severity=severity,
            frame_timestamp=now,
        ))
    return results


def draw_detections(frame_bgr: np.ndarray, detections: List[PotholeDetection]) -> np.ndarray:
    """Overlay bounding boxes + severity labels for visual debugging."""
    color_map = {"minor": (0, 200, 0), "moderate": (0, 165, 255), "severe": (0, 0, 255)}
    out = frame_bgr.copy()
    for d in detections:
        x, y, w, h = d.bbox
        color = color_map.get(d.severity, (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{d.severity} [prov. {d.est_depth_cm}cm] ({int(d.confidence * 100)}%)"
        cv2.putText(out, label, (x, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    return out


def post_detections_to_backend(detections: List[PotholeDetection], api_url: str,
                                vehicle_id: str, latitude: float, longitude: float):
    """
    Sends each detection to the backend's ingestion endpoint, exactly like a
    real onboard unit would after each detection — this is the piece that
    actually makes them show up on the dashboard. Requires the `requests`
    package (pip install requests) and the backend running at api_url.
    """
    import requests
    print(f"  Note: Using simulated/demo coordinates ({latitude}, {longitude}) - no hardware GPS connected.")
    for d in detections:
        payload = {
            "vehicle_id": vehicle_id,
            "latitude": latitude,
            "longitude": longitude,
            "severity": d.severity,
            "est_depth_cm": d.est_depth_cm,
            "confidence": d.confidence,
        }
        try:
            resp = requests.post(api_url, json=payload, timeout=5)
            resp.raise_for_status()
            result = resp.json()
            tag = "MERGED into existing report" if result.get("merged") else "NEW report created"
            print(f"  -> Posted to backend: {tag} (id={result['detection']['id']})")
        except Exception as e:
            print(f"  -> FAILED to post to backend: {e}")
            print(f"     Is the backend running at {api_url}? (python backend/app.py)")


if __name__ == "__main__":
    default_weights = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pothole_best.pt"))
    parser = argparse.ArgumentParser(description="Detect potholes from a stereo image pair using YOLO.")
    parser.add_argument("--left", required=True, help="Path to left camera frame")
    parser.add_argument("--right", required=True, help="Path to right camera frame")
    parser.add_argument("--weights", default=default_weights, help=f"Path to pothole-trained YOLO weights (.pt) (default: {default_weights})")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--out", default=None, help="Optional path to save annotated image")
    parser.add_argument("--post-to-backend", action="store_true",
                         help="Automatically POST any detections to the backend so they appear on the dashboard")
    parser.add_argument("--api-url", default="http://localhost:5000/api/detections",
                         help="Backend ingestion endpoint (used with --post-to-backend)")
    parser.add_argument("--vehicle-id", default="demo-vehicle",
                         help="Vehicle identifier (default: demo-vehicle)")
    parser.add_argument("--lat", type=float, default=8.5241,
                         help="Sample/demo latitude (default: 8.5241, Thiruvananthapuram; no hardware GPS attached)")
    parser.add_argument("--lon", type=float, default=76.9366,
                         help="Sample/demo longitude (default: 76.9366, Thiruvananthapuram; no hardware GPS attached)")
    args = parser.parse_args()

    dets = process_stereo_pair(args.left, args.right, weights_path=args.weights, conf_threshold=args.conf)
    print(json.dumps([d.to_dict() for d in dets], indent=2))

    if args.out:
        left_img = cv2.imread(args.left)
        annotated = draw_detections(left_img, dets)
        cv2.imwrite(args.out, annotated)
        print(f"Annotated image saved to {args.out}")

    if args.post_to_backend:
        if not dets:
            print("No potholes detected — nothing to post.")
        else:
            print(f"\nPosting {len(dets)} detection(s) to {args.api_url} ...")
            post_detections_to_backend(dets, args.api_url, args.vehicle_id, args.lat, args.lon)
