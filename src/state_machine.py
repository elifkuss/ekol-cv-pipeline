import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger("EkolYOLOTrackerPipeline")

class WarehouseStateMachine:
    """
    This class checks if the worker packs the correct items.
    It compares the detected items with the active order.
    """
    def __init__(self):
        self.active_order: Optional[str] = None
        self.processed_ids = set() # Saved IDs to prevent counting the same item twice

    def set_active_order(self, order_item: str):
        # Update the active order from the WMS (Warehouse Management System)
        self.active_order = order_item.upper()
        logger.info(f"SYSTEM: Active WMS Order Updated -> Expected Item: {self.active_order}")

    def check_packaging_zone(self, track_id: int, label: str, xyxy: np.ndarray, frame_shape: Tuple[int, int]) -> str:
        # If we already processed this item, skip it
        if track_id in self.processed_ids:
            return "ALREADY_PROCESSED"

        h, w = frame_shape[:2]
        x1, y1, x2, y2 = xyxy
        
        # Packaging Line Simulation: The right 35% of the screen is the end of the belt
        packaging_line_x = int(w * 0.65)
        
        if x2 > packaging_line_x:
            self.processed_ids.add(track_id)
            normalized_label = self._map_to_warehouse_category(label)
            
            # Check if the item matches the expected order
            if normalized_label == self.active_order:
                logger.info(f"SUCCESS: Correct item packed! Track ID: {track_id} | Item: {normalized_label}")
                return "SUCCESS"
            else:
                logger.warning(
                    f"ANOMALY: Wrong item packed! Track ID: {track_id} | "
                    f"Expected: {self.active_order} | Found: {normalized_label}"
                )
                return "ANOMALY_MISMATCH"
                
        return "IN_TRANSIT"

    def _map_to_warehouse_category(self, label: str) -> str:
        # Maps standard COCO classes to warehouse item names
        label = label.upper()
        mapping = {
            "TIE": "GOMLEK",
            "HANDBAG": "GOMLEK",
            "PERSON": "CALISAN",
            "BOTTLE": "KUTU",
            "BACKPACK": "PAKET"
        }
        return mapping.get(label, label)