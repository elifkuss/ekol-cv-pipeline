import numpy as np
from typing import Tuple
from ultralytics import YOLO

class EkolYOLOTracker:
    """
    This class loads the YOLOv8 model and manages the ByteTrack system.
    """
    def __init__(self, model_path: str = "yolov8n.pt", tracker_config: str = "config/bytetrack.yaml"):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config

    def track_objects(self, frame: np.ndarray):
        """
        Runs YOLO and ByteTrack on the frame and returns the results.
        """
        # 'persist=True' keeps track IDs active across frames
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            verbose=False
        )
        return results[0]