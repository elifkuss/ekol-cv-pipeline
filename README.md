# 📦 Ekol Logistics Real-Time Smart Packaging & Anomaly Detection Pipeline

An end-to-end, production-ready Computer Vision pipeline designed for monitoring warehouse packaging bands, ensuring worker privacy (KVKK compliance), validating orders against a Warehouse Management System (WMS), and reporting real-time performance metrics.

## 🏗️ Project Architecture & Design Philosophy

This project follows the **Separation of Concerns (SoC)** principle. Instead of a single "spaghetti" script, the system is divided into modular, highly-cohesive components. This makes it easy to maintain, scale, and test.

```text
ekol-cv-pipeline/
├── config/
│   └── bytetrack.yaml         # ByteTrack tracking hyper-parameters
├── data/
│   └── packaging_depo.mp4     # Raw input video from the warehouse camera
├── src/
│   ├── __init__.py            # Module identifier
│   ├── preprocessor.py        # Worker face blurring (DataScrubber)
│   ├── tracker.py             # YOLOv8 and ByteTrack tracking manager
│   ├── state_machine.py       # Warehouse business logic and WMS integration
│   └── pipeline.py            # Main pipeline coordinator & Performance Tracker
├── tests/
│   ├── __init__.py            # Test module identifier
│   └── test_pipeline.py       # Automated unit tests (JUnit/Unittest style)
├── requirements.txt           # Production dependencies
├── .gitignore                 # Files excluded from git tracking
├── run_pipeline.py            # Local command-line runner for video streams
├── api.py                     # Non-blocking FastAPI REST server
└── README.md                  # System documentation & interview cheat sheet
```

## ✨ Core Features & Component Breakdown

### 1. Dynamic Data Scrubbing & KVKK Privacy (`src/preprocessor.py`)
* **The Problem:** Recording workers in industrial facilities 24/7 raises privacy and legal concerns (KVKK/GDPR).
* **Our Solution:** The `DataScrubber` automatically detects `person` classes (COCO class 0) using YOLO. It isolates the top 25% of the bounding box (estimated face area) and applies a heavy, real-time **Gaussian Blur** filter. The surrounding packaging area and products remain clear for analysis.

### 2. Multi-Object Tracking with YOLOv8 & ByteTrack (`src/tracker.py`)
* **The Problem:** Hand occlusions can cause object detectors to lose track of items, leading to duplicate counts.
* **Our Solution:** We integrated YOLOv8 with **ByteTrack**. ByteTrack utilizes Kalman Filters to maintain object identity (Track IDs) even during short-term occlusions, ensuring reliable, continuous tracking.

### 3. Business Logic State Machine (`src/state_machine.py`)
* **The Problem:** Hardcoding business rules inside deep learning loops is bad software engineering.
* **Our Solution:** A separate `WarehouseStateMachine` manages the lifecycle of items.
  * **In Transit:** Items moving on the left side of the belt.
  * **Success:** When an item enters the packaging zone (right 35% of the frame) and matches the `active_order` pulled from WMS.
  * **Anomaly Mismatch:** When an incorrect item is packed. It generates warnings and flags immediate anomalies.
  * **Already Processed:** A lock-and-key system using `track_id` sets to prevent duplicate counts.

### 4. Observability & Performance Tracking (`src/pipeline.py`)
* Every processed frame is timed in milliseconds.
* The `PerformanceTracker` dynamically calculates average latency, successful packagings, detected anomalies, and estimated FPS, outputting a professional diagnostic report at completion.

### 5. Multi-Threaded FastAPI Servicing (`api.py`)
* Provides a high-throughput endpoint `/process_video` that allows WMS or other external web applications to upload raw MP4 videos.
* **Engineering Detail:** Uses synchronous `def` instead of `async def` for video processing to prevent main server loop from freezing.

---

## 🎓 Academic Connections & Strengths (Elif Kuş Profile)

The architectural choices in this project are grounded in rigorous academic and professional experiences:

* **Data Scrubbing & Bias Mitigation (TÜBİTAK Project):**
  * *My Experience:* Analyzed massive medical datasets from the UK Biobank and resolved statistical biases.
  * *Application here:* Applied the same discipline to sanitize visual data—cleaning noisy worker backgrounds and anonymizing faces.
* **Handling Rare Anomalies (SMOTE Projects):**
  * *My Experience:* Used SMOTE to handle heavily imbalanced datasets, achieving a 93% success rate.
  * *Application here:* Industrial packaging errors are rare. The `WarehouseStateMachine` acts as an analytical filter designed to capture these rare anomalies (`ANOMALY_MISMATCH`).

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites & Environment Setup
Create and activate a clean Python virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Running the Pipeline (Local Video Processor)
Place your warehouse test video inside the `data/` folder and name it `packaging_depo.mp4`. Run the pipeline:

```bash
python run_pipeline.py
```

### 4. Running the Web API (FastAPI Server)
Launch the server:
```bash
python api.py
```
* The API will start running at `http://127.0.0.1:8000`.
* You can open your browser to `http://127.0.0.1:8000/docs` to use the auto-generated Swagger UI for easy testing.

#### 💡 API Usage Examples

**Option A: Using cURL (Terminal)**
Send a video to the API and save the processed response directly:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/process_video' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@data/packaging_depo.mp4;type=video/mp4' \
  --output processed_result.mp4
```

**Option B: Using Python (`requests` library)**
How a WMS or external service would interact with this API programmatically:
```python
import requests

url = "http://127.0.0.1:8000/process_video"
video_path = "data/packaging_depo.mp4"

with open(video_path, "rb") as video_file:
    files = {"file": video_file}
    print("Uploading video for processing...")
    response = requests.post(url, files=files)

if response.status_code == 200:
    with open("processed_result.mp4", "wb") as f:
        f.write(response.content)
    print("✅ Success: Processed video downloaded!")
else:
    print(f"❌ Error: {response.status_code}")
```

### 5. Running Automated Unit Tests
```bash
python -m pytest tests/test_pipeline.py -v
```

---

## 📊 Technical Performance and Evaluation Report (Example Output)

```text
=== TECHNICAL PERFORMANCE AND EVALUATION REPORT ===
-> Total Frames Processed: 350
-> Average Latency Ms: 12.45 ms
-> Estimated Fps: 80.32 FPS
-> Successful Packagings: 12
-> Detected Anomalies: 1
===================================================
```