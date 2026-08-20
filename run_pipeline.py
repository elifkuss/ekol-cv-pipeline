import cv2
import numpy as np
import time
from src.pipeline import EkolCVPipeline

def main():
    print("\n=== EKOL LOGISTICS REAL-TIME YOLO & BYTETRACK PIPELINE STARTING ===\n")
    
    # Initialize the pipeline
    pipeline = EkolCVPipeline(model_path="yolov8n.pt")
    
    # Simulate an active order from WMS ('tie' class represents shirt)
    pipeline.state_machine.set_active_order("TIE")
    
    print("\n--- Running validation on synthetic frames ---")
    mock_frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
    cv2.putText(mock_frame, "Ekol Onsite Sim", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    for i in range(10):
        _ = pipeline.process_frame(mock_frame)
        time.sleep(0.01)
        
    # Print the Performance Report
    report = pipeline.perf_tracker.generate_report()
    print("\n=== TECHNICAL PERFORMANCE AND EVALUATION REPORT ===")
    for k, v in report.items():
        print(f"-> {k.replace('_', ' ').title()}: {v}")
    print("===================================================\n")

if __name__ == "__main__":
    main()
