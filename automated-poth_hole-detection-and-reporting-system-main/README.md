# Roadstate — Live AI Pothole Detection & Reporting System

> Real-time pothole detection from moving vehicles, using stereo vision for depth/severity classification, automatic GPS geo-tagging, duplicate-report merging, and a live map dashboard for municipal officials.

Manual pothole reporting — citizen complaints and periodic road surveys — is slow and reactive. Roadstate automates the whole loop: a vehicle-mounted stereo camera detects a pothole, estimates how severe it is, tags its exact location, and gets it onto a live municipal dashboard, automatically merging duplicate reports from different vehicles along the way.

Built end-to-end for a hackathon — every component below is real, runnable code, tested together (detection → backend → dashboard verified live with a simulated vehicle fleet).

## What it does

- 🎥 **Detects potholes in real time** from a vehicle-mounted stereo camera, using a YOLOv8 model
- 📏 **Estimates depth and severity** (minor / moderate / severe) via stereo disparity mapping
- 📍 **Geo-tags every detection** with GPS coordinates and free reverse-geocoding (no paid API)
- 🔁 **Automatically merges duplicate reports** — multiple vehicles hitting the same pothole become one alert, not ten
- 🖥️ **Live municipal dashboard** — real-time map + feed with a reported → acknowledged → scheduled → repaired workflow
- 🚌 **Deployable on public transit and opt-in private vehicles** — a crowdsourced road-health sensor network cities don't have to build themselves

## What's out of scope

This is a software prototype — it does not include the physical stereo camera rig or onboard computer (Raspberry Pi / Jetson Nano). Everything else — the detection algorithm, backend, geolocation, and dashboard — is real, runnable code, not a mockup.

## Quick start

Jump to the [setup instructions](#1-quick-start-full-demo-no-camera-needed) below, or straight to `training/dataset_setup.md` if you want to train your own detector first.

---

```
pothole-system/
├── detection/
│   ├── detect_pothole.py     # Pothole detection (YOLO-only) + stereo depth estimation
│   └── requirements.txt
├── training/
│   ├── train_yolo.py           # Fine-tune YOLOv8 on a pothole dataset
│   ├── data.yaml.example        # Template dataset config
│   ├── dataset_setup.md         # Where to get a public pothole dataset
│   └── requirements.txt
├── backend/
│   ├── app.py                 # Flask REST API
│   ├── models.py               # SQLite schema + duplicate-report merging
│   ├── geolocation.py           # Free reverse-geocoding (OSM Nominatim)
│   └── requirements.txt
├── simulator/
│   └── simulate_reports.py     # Fakes a vehicle fleet to demo the pipeline
└── dashboard/
    └── dashboard.html           # Live municipal dashboard (map + feed)
```

---

## 1. Quick start (full demo, no camera needed)

```bash
# Terminal 1 — backend
cd backend
pip install -r requirements.txt
python app.py                    # -> http://localhost:5000

# Terminal 2 — simulate a fleet of vehicles reporting potholes
cd simulator
pip install requests
python simulate_reports.py --count 60 --interval 0.2

# Then open dashboard/dashboard.html in a browser
```

The dashboard polls the backend every 4 seconds, so watch reports and stats
update live as the simulator runs. If you open the dashboard *without* the
backend running, it automatically falls back to a generated demo dataset so
the UI is still fully explorable (you'll see a yellow "Demo data" indicator
in the header instead of the green "Live" one).

## 2. Training a pothole-detection model (required before detection works)

The detector is **YOLO-only** — it needs weights fine-tuned on a pothole
dataset (a stock COCO checkpoint won't detect potholes; "pothole" isn't a
COCO class). See `training/dataset_setup.md` for where to get a public
pothole dataset, then:

```bash
cd training
pip install -r requirements.txt
python train_yolo.py --data pothole_dataset/data.yaml --epochs 80
# -> weights land at runs/detect/train/weights/best.pt
```

## 3. Running detection on stereo images

```bash
cd detection
pip install -r requirements.txt
python detect_pothole.py --left left.jpg --right right.jpg \
  --weights ../training/runs/detect/train/weights/best.pt --out annotated.jpg
```

This prints detected potholes as JSON (bbox, confidence, estimated depth,
severity) and saves an annotated debug image. To feed a real detection into
the backend:

```bash
curl -X POST http://localhost:5000/api/detections \
  -H "Content-Type: application/json" \
  -d '{"vehicle_id":"bus-042","latitude":8.5241,"longitude":76.9366,
       "severity":"moderate","est_depth_cm":3.2,"confidence":0.87}'
```

In a real onboard unit, this POST call is what the vehicle unit fires every
time `detect_pothole.py`'s pipeline flags a pothole, using the vehicle's live
GPS coordinates instead of hardcoded ones.

---

## 3. What's real vs. what's a placeholder

Being upfront about where this prototype is production-ready and where it's
a stand-in for something that needs real-world calibration or training data:

| Component | Status |
|---|---|
| Backend API, dedup logic, dashboard, geolocation | **Fully working.** Tested end-to-end in this build. |
| YOLO detector + training pipeline | **Working code.** Ships with no pre-trained weights — you train them yourself on a public pothole dataset (see `training/dataset_setup.md`). This is a real, necessary ML step, not something to fake with placeholder weights. |
| Stereo depth estimation | **Working OpenCV SGBM pipeline**, but `focal_length_px` / `baseline_m` are placeholder camera-calibration values. Real depth accuracy requires calibrating your actual stereo rig with a checkerboard pattern (`cv2.stereoCalibrate`). |
| Severity thresholds | Placeholder cm/area cutoffs — calibrate against your municipality's actual road-damage severity standard. |
| Daylight-only operation | Not yet implemented in code — straightforward to add (check sunrise/sunset via a free API, or an ambient light sensor reading, before running the detection loop). |

## 4. Suggested next steps

1. **Fine-tune YOLOv8 on more of your own region's road photos** once you
   have a working baseline — the public datasets are a starting point, not
   a substitute for photos of your actual deployment area.
2. **Calibrate your actual stereo camera rig** and update `focal_length_px`
   / `baseline_m` in `StereoDepthEstimator`.
3. **Deploy the backend** behind a production WSGI server (gunicorn/uWSGI)
   with PostgreSQL instead of SQLite once you're past prototype scale.
4. **Add authentication** to the `PATCH /api/detections/<id>` endpoint before
   any real municipal staff use it — right now it's open, which is fine for
   a demo but not for production.
