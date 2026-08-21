import os
import time
import logging
import cv2
import sys
from pathlib import Path

# Add project root directory
sys.path.append(str(Path(__file__).parent))

from src.pipeline import PersonTrackerModule, ProductRecognitionModule, ProductLoggingModule

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EkolCVPipelineRunner")

class EkolCasePipeline:

    def __init__(self, video_path: str, output_csv: str = "output/product_logs.csv"):
        self.video_path = video_path
        self.output_csv = output_csv
        
        # Initialize modules
        self.person_tracker = PersonTrackerModule()
        self.product_recognizer = ProductRecognitionModule()
        self.product_logger = ProductLoggingModule(output_csv)
        
        self.cap = None
        self.fps = 30
        
    def run(self):

        logger.info(f"Processing video: {self.video_path}")
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logger.error("Could not open video!")
            return
            
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            current_time = frame_count / self.fps
            
            # Process every 100ms
            if frame_count % max(1, int(self.fps * 0.1)) == 0:
                # 1. Worker tracking
                worker_active, worker_bbox = self.person_tracker.is_worker_active(frame)
                
                if worker_active:
                    # 2. Product detection
                    results = self.product_recognizer.detect_products(frame)
                    
                    # 3. Logging
                    self.product_logger.update_tracks(results, current_time)
                    
                    # Finalize lost tracks
                    self.product_logger.finalize_lost_tracks(current_time)
            
            # Show status every 30 frames
            if frame_count % 30 == 0:
                logger.info(f"Frame {frame_count}, Active tracking: {len(self.product_logger.active_tracks)}")
        
        # Video finished - finalize all tracks
        self.product_logger.finalize_lost_tracks(current_time, force=True)
        self.cap.release()
        
        logger.info(f"Pipeline completed! Output: {self.output_csv}")
        logger.info(f"Total {len(self.product_logger.active_tracks)} products processed.")


def main():
    print("\n=== EKOL LOGISTICS REAL-TIME PIPELINE ===\n")
    
    # Create sample video for testing (if it does not exist)
    video_path = "data/packaging_depo.mp4"
    output_csv = "output/product_logs.csv"
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    if not os.path.exists(video_path):
        logger.warning(f"Video not found: {video_path}")
        logger.info("Creating sample video...")
        create_sample_video(video_path)
    
    try:
        pipeline = EkolCasePipeline(video_path=video_path, output_csv=output_csv)
        pipeline.run()
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

def create_sample_video(output_path: str):

    cap = cv2.VideoCapture(0)  # Webcam
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
    
    for _ in range(100):  # 100 frames
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    logger.info(f"Sample video created: {output_path}")

if __name__ == "__main__":
    main()