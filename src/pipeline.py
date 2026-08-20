import time
import numpy as np
import cv2
from typing import Tuple
from src.preprocessor import DataScrubber
from src.tracker import EkolYOLOTracker
from src.state_machine import WarehouseStateMachine

class PerformanceTracker:
    """
    This class measures the speed of the code (latency and FPS).
    """
    def __init__(self):
        self.frame_times = []
        self.success_count = 0
        self.anomaly_count = 0

    def record_frame(self, latency_ms: float):
        self.frame_times.append(latency_ms)

    def record_event(self, event_type: str):
        if event_type == "SUCCESS":
            self.success_count += 1
        elif event_type == "ANOMALY_MISMATCH":
            self.anomaly_count += 1

    def generate_report(self) -> dict:
        avg_latency = np.mean(self.frame_times) if self.frame_times else 0
        fps = 1000 / avg_latency if avg_latency > 0 else 0
        return {
            "total_frames_processed": len(self.frame_times),
            "average_latency_ms": round(avg_latency, 2),
            "estimated_fps": round(fps, 2),
            "successful_packagings": self.success_count,
            "detected_anomalies": self.anomaly_count
        }

class EkolCVPipeline:
    """
    This is the main pipeline class that connects the scrubber, tracker, and state machine.
    """
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.scrubber = DataScrubber()
        self.tracker = EkolYOLOTracker(model_path=model_path)
        self.state_machine = WarehouseStateMachine()
        self.perf_tracker = PerformanceTracker()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        start_time = time.time()
        
        # 1. Track objects using YOLO and ByteTrack
        results = self.tracker.track_objects(frame)
        
        # 2. Blur worker faces for privacy (Data Scrubbing)
        frame = self.scrubber.anonymize_workers(frame, results)
        
        # 3. Check packaging rules and state machine
        boxes = results.boxes
        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            xyxys = boxes.xyxy.cpu().numpy().astype(int)
            class_ids = boxes.cls.int().cpu().tolist()
            
            for track_id, xyxy, class_id in zip(track_ids, xyxys, class_ids):
                label = self.tracker.model.names[class_id]
                x1, y1, x2, y2 = xyxy
                
                status = self.state_machine.check_packaging_zone(track_id, label, xyxy, frame.shape)
                self.perf_tracker.record_event(status)
                
                # Draw boxes: Red for anomaly, Green for normal
                color = (0, 255, 0) if status != "ANOMALY_MISMATCH" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                text = f"ID {track_id}: {label.upper()}"
                if status == "ANOMALY_MISMATCH":
                    text += " [ERROR]"
                elif status == "SUCCESS":
                    text += " [OK]"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw active target order on screen
        cv2.putText(
            frame, f"WMS TARGET: {self.state_machine.active_order}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2
        )
        
        # Calculate speed metrics
        latency_ms = (time.time() - start_time) * 1000
        self.perf_tracker.record_frame(latency_ms)
        
        return frame
