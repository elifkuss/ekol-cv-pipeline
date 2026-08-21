import cv2
import numpy as np
import logging
import time
import os
import csv
from typing import Dict, Tuple, Optional
from ultralytics import YOLO

logger = logging.getLogger("EkolCasePipeline")

class PersonTrackerModule:

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def is_worker_active(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:

        results = self.model.track(source=frame, persist=True, classes=[0], verbose=False)
        if not results or len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return False, None
        
        box = results[0].boxes[0]
        xyxy = box.xyxy.cpu().numpy().astype(int)[0]
        return True, (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
    

class ProductRecognitionModule:

    def __init__(self, model_path: str = "yolov8s-world.pt"):
        logger.info(f"YOLO-World Model yükleniyor: {model_path}")
        self.model = YOLO(model_path)

        self.model.set_classes(["shirt", "pants", "cardboard box"])

    def detect_products(self, frame: np.ndarray):
        results = self.model.track(source=frame, persist=True, verbose=False)
        return results[0] if results else None
    

class ProductLoggingModule:

    def __init__(self, output_csv_path: str = "output/product_logs.csv"):
        self.output_csv_path = output_csv_path
        self.active_tracks: Dict[int, dict] = {}  
        os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
        self._initialize_csv()

    def _initialize_csv(self):

        with open(self.output_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["start_time", "end_time", "product_category", "last_bbox"])
        logger.info(f"CSV Logger initialized at: {self.output_csv_path}")

    def _map_to_case_category(self, label: str) -> str:
        label = label.lower()
        if "shirt" in label:
            return "Gömlek"
        if "pants" in label:
            return "Pantolon"
        return "Kutulanmiş"

    def update_tracks(self, results, current_time_sec: float):

        if results is None or results.boxes is None:
            return

        boxes = results.boxes
        if boxes.id is None:
            return

        track_ids = boxes.id.int().cpu().tolist()
        xyxys = boxes.xyxy.cpu().numpy().astype(int)
        class_ids = boxes.cls.int().cpu().tolist()

        for track_id, xyxy, class_id in zip(track_ids, xyxys, class_ids):
            label = results.names[class_id]
            category = self._map_to_case_category(label)

            if track_id not in self.active_tracks:
                # Ürün sistemde İLK kez görüldü (Başlangıç Zamanı Tetiklendi)
                self.active_tracks[track_id] = {
                    "start_time": round(current_time_sec, 2),
                    "end_time": round(current_time_sec, 2),
                    "category": category,
                    "last_bbox": [int(x) for x in xyxy]
                }
                logger.info(f"DETECTED: New {category} entered the scene! ID: {track_id}")
            else:
                # Ürün hala bant üzerinde (Bitiş zamanı ve son koordinatlar güncelleniyor)
                self.active_tracks[track_id]["end_time"] = round(current_time_sec, 2)
                self.active_tracks[track_id]["last_bbox"] = [int(x) for x in xyxy]

    def finalize_lost_tracks(self, current_time_sec: float, force: bool = False, max_age_sec: float = 1.5):

        lost_ids = []
        for track_id, data in self.active_tracks.items():
            # Eğer video bittiyse (force=True) veya nesne ekrandan çıkalı max_age_sec geçmişse
            if force or (current_time_sec - data["end_time"]) > max_age_sec:
                self._write_to_csv(data)
                lost_ids.append(track_id)
        
        for track_id in lost_ids:
            del self.active_tracks[track_id]

    def _write_to_csv(self, track_data: dict):
        """Veriyi CSV dosyasina ekler (append mode)."""
        with open(self.output_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                f"{track_data['start_time']}s",
                f"{track_data['end_time']}s",
                track_data["category"],
                str(track_data["last_bbox"])
            ])
        logger.info(f"LOG RECORDED -> {track_data['category']} ({track_data['start_time']}s - {track_data['end_time']}s)")


