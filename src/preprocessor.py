import cv2
import numpy as np
from typing import Tuple

class DataScrubber:
    """
    This class blurs the faces of workers to protect their privacy (KVKK rules).
    """
    def __init__(self, blur_kernel: Tuple[int, int] = (51, 51)):
        # We use a 51x51 kernel size to get a strong blur effect
        self.blur_kernel = blur_kernel

    def anonymize_workers(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Finds 'person' objects and blurs their face area.
        """
        if not hasattr(results, 'boxes') or results.boxes is None:
            return frame

        boxes = results.boxes
        for box in boxes:
            cls_id = int(box.cls.item())
            # In the COCO dataset, 'person' class ID is 0
            if cls_id == 0:
                xyxy = box.xyxy.cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                
                # We assume the face is in the top 25% of the person's body box
                face_height = int((y2 - y1) * 0.25)
                face_y2 = y1 + face_height
                
                # Keep coordinates inside the frame boundaries
                h, w, _ = frame.shape
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w, x2), min(h, face_y2)
                
                if (x2_c - x1_c) > 0 and (y2_c - y1_c) > 0:
                    face_roi = frame[y1_c:y2_c, x1_c:x2_c]
                    # Apply Gaussian Blur to anonymize the face
                    blurred_roi = cv2.GaussianBlur(face_roi, self.blur_kernel, 0)
                    frame[y1_c:y2_c, x1_c:x2_c] = blurred_roi
                    
        return frame