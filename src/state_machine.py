import logging
import numpy as np
from typing import Tuple, Dict, Optional

logger = logging.getLogger("EkolCasePipeline")

class WarehouseStateMachine:
    """
    This class checks if the worker packs the correct items.
    It compares the detected items with the active order.
    """
    def __init__(self, fps: int=30):
        self.fps = fps

        self.frame_interval = max(1, int(fps * 0.1))

        self.worker_present = False

        self.active_products = Dict[int,dict] = {}

    def update_worker_status(self, is_present: bool):
        self.worker_present = is_present

        if not is_present:
            logger.warning("Calisan masadan ayrildi!")
    
    def process_frame(self, detected_objects: list, current_time: float) -> list:

        completed_logs = []
        
        if not self.worker_present:
            self.active_products.clear()
            return completed_logs
            
        current_frame_track_ids = set()
        
        for obj in detected_objects:
            track_id, class_name, bbox = obj
            current_frame_track_ids.add(track_id)
            
            if track_id not in self.active_products:
                # 1. DURUM: Ürün sisteme yeni girdi (Başlangıç zamanı kaydedilir)
                self.active_products[track_id] = {
                    "class_name": class_name,
                    "start_time": current_time,
                    "last_seen_time": current_time,
                    "last_bbox": bbox,
                    "frame_counter": 0
                }
                logger.info(f"Yeni ürün tespit edildi: ID {track_id} ({class_name})")
            else:
                # 2. DURUM: Ürün zaten takipte. 100ms kontrolü (frame_counter ile)
                prod = self.active_products[track_id]
                prod["frame_counter"] += 1
                
                if prod["frame_counter"] >= self.frame_interval:
                    prod["last_seen_time"] = current_time
                    prod["last_bbox"] = bbox
                    prod["class_name"] = class_name  # Sınıf bilgisini güncel tut
                    prod["frame_counter"] = 0
                    logger.debug(f"Ürün ID {track_id} takip ediliyor (100ms güncellemesi).")

    
        missing_ids = [tid for tid in self.active_products if tid not in current_frame_track_ids]
        
        for tid in missing_ids:
            finished_product = self.active_products.pop(tid)
            completed_logs.append({
                "category": finished_product["class_name"],
                "start_time": finished_product["start_time"],
                "end_time": finished_product["last_seen_time"],
                "last_bbox": finished_product["last_bbox"]
            })
            logger.info(f"Ürün istasyondan ayrildi ve tamamlandi: ID {tid}")
            
        return completed_logs