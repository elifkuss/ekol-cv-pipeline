# 📦 Ekol Logistics Real-Time Product Tracking System

This project is a Computer Vision pipeline designed to track products at warehouse stations in real-time. The system monitors the workspace from an overhead camera, detects when a worker is present, identifies the products (Boxed, Shirt, or Pants), and logs the results into a CSV file every 100ms.

## 🏗️ Project Architecture & Design

This system is built using an **Object-Oriented Programming (OOP)** approach to keep the code clean and organized. Instead of a single complex script, the project is divided into modular components. This makes it easy to maintain, test, and run locally on standard hardware (like an NVidia 2080 GPU).

The pipeline consists of three main parts:
*   **Person Tracking Module:** Checks if the worker is at the table to start the process.
*   **Product Recognition Module:** Identifies the category of the product the worker is handling.
*   **Product Logging Module:** Saves the start time, end time, category, and bounding box location into a CSV report.

### 🤖 AI Models & Tracking Strategy (Why I Chose What I Chose)

I didn't want to weigh the system down with one massive, slow AI model. Instead, I split the workload into specific, highly optimized pieces:

*   **YOLOv8 Nano (Person Detection):** I only needed to find where the workers are so the system can blur their faces. There was no need for a heavy model here. I went with the `Nano` version because it is incredibly lightweight and fast. It handles the KVKK privacy filter smoothly without dragging down the overall FPS.
*   **YOLO-World (Product Recognition):** Instead of spending weeks collecting thousands of warehouse images and training a custom model from scratch, I decided to use YOLO-World. Thanks to its "zero-shot" magic, I just give it text prompts like "shirt", "pants", or "cardboard box", and it finds them instantly. This makes the system super flexible—if a new product type arrives at the warehouse tomorrow, we can just type its name and track it with zero retraining.
*   **ByteTrack (Multi-Object Tracking):** Finding the items is one thing, but we also need to make sure we don't count the same shirt twice. I used ByteTrack to assign a unique ID to each item. Even if a worker's hand covers the product for a second, ByteTrack remembers where it was, keeping the same ID and completely preventing duplicate counts on the band.

```text
ekol-cv-pipeline/
├── config/
│   └── bytetrack.yaml           # ByteTrack tracking hyper-parameters
├── data/
│   └── packaging_depo.mp4       # Raw input test video from the warehouse camera
├── output/
│   └── product_logs.csv         # Automatically generated CSV log for processed products
├── src/
│   ├── __init__.py              # Module identifier
│   ├── pipeline.py              # Main coordination: Person tracking, product recognition, and logging
│   ├── preprocessor.py          # DataScrubber for worker face blurring (KVKK privacy)
│   ├── state_machine.py         # Warehouse business logic and active order tracking
│   └── tracker.py               # YOLO and ByteTrack tracking manager
├── tests/
│   ├── __init__.py              # Test module identifier
│   └── test_pipeline.py         # Automated unit tests (JUnit/Unittest style)
├── venv/                        # Python virtual environment (Not tracked in git)
├── .gitignore                   # Files excluded from git tracking
├── api.py                       # FastAPI REST server for web-based video processing
├── README.md                    # System documentation
├── requirements.txt             # Production dependencies
└── run_pipeline.py              # Local command-line runner for video streams
```
⚠️ Note on Model Weights:
Due to GitHub's file size limits, the pre-trained model weights (yolov8n.pt for person detection and yolov8s-world.pt for zero-shot product recognition) are not included in this repository. Ensure they are downloaded or placed in the root directory before running the pipeline. (If missing, the ultralytics library will attempt to download them automatically upon first execution).

## ✨ Core Features & How It Works

### 1. Worker Privacy and KVKK Compliance (`preprocessor.py`)
* *(A quick personal note: During our first interview, Batı Bey mentioned the company’s strong emphasis on KVKK regulations. Keeping his feedback in mind, I applied a Gaussian blur to hide the employees' faces to ensure privacy.)*

* Continuous 24/7 camera recording in industrial environments can raise legal and privacy concerns. Our system detects workers in the frame and applies a real-time Gaussian Blur exclusively to their facial areas for anonymization. The products on the table remain crystal clear for analysis.

### 2. Advanced Product Recognition (`pipeline.py`)
* The system utilizes the YOLO-World model to recognize clothing (shirts, pants, etc.) and cardboard boxes. When it detects a new product, it registers the item, tracks it until it leaves the camera's view (using ByteTrack logic), and calculates the total time it remained in the system.

### 3. Smart Business Logic & Logging (`state_machine.py` & `pipeline.py`)
* Products are not just simply counted. The system tracks exactly when a product arrives at the table, how long it is processed, and when it leaves, second by second. Once the processing is complete, the item's data is automatically logged into the `output/product_logs.csv` file.

### 4. Multi-Threaded FastAPI Servicing (`api.py`)
* Provides a high-throughput endpoint `/process_video` that allows WMS or other external web applications to upload raw MP4 videos. Uses synchronous `def` instead of `async def` for video processing to prevent main server loop from freezing.

---

## 🎓 Behind the Architecture (My Background)

The architectural choices in this project are grounded in rigorous academic and professional experiences:

*   **Data Scrubbing & Bias Mitigation (TÜBİTAK Research):** 
    *   *My Experience:* Analyzed massive medical datasets from the UK Biobank and resolved statistical biases.
    *   *Application Here:* Applied the same discipline to sanitize visual data—cleaning noisy worker backgrounds and anonymizing faces via the `DataScrubber`.

*   **Handling Rare Anomalies (SMOTE Projects):** 
    *   *My Experience:* Used SMOTE to handle heavily imbalanced network security datasets, achieving a 93% success rate in threat detection.
    *   *Application Here:* Industrial packaging errors are rare. Instead of forcing an AI to guess, the `WarehouseStateMachine` acts as a deterministic analytical filter designed to capture these rare anomalies (`ANOMALY_MISMATCH`).
    
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
=== EKOL LOGISTICS REAL-TIME PIPELINE ===
2026-08-21 15:55:24 - INFO - Loading YOLO-World Model: yolov8s-world.pt
2026-08-21 15:55:32 - INFO - CSV Logger initialized at: output/product_logs.csv
2026-08-21 15:55:32 - INFO - Processing video: data/packaging_depo.mp4
2026-08-21 15:55:36 - INFO - DETECTED: New Shirt entered the scene! ID: 1
2026-08-21 15:55:40 - INFO - LOG RECORDED -> Shirt (3.2s - 7.5s)
2026-08-21 15:56:22 - INFO - Pipeline completed! Output: output/product_logs.csv
```
