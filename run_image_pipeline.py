import cv2
import os
import glob
import time
import numpy as np
from src.pipeline import EkolCVPipeline

def main():
    print("\n=== EKOL LOGISTICS REAL-TIME IMAGE BATCH RUNNER ===\n")
    
    # 1. Initialize the modular pipeline (YOLOv8 & ByteTrack)
    # The model is loaded once and used for all images
    pipeline = EkolCVPipeline(model_path="yolov8n.pt")
    
    # 2. Set the expected active order from WMS
    pipeline.state_machine.set_active_order("TIE")
    
    # Define input and output folders
    input_dir = "data/images"
    output_dir = "data/output_images"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images in the input folder
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        
    # If the input directory is empty, create a synthetic test image to make it runnable out of the box
    if not image_paths:
        print(f"Warning: No images found in '{input_dir}' directory.")
        print("Creating a synthetic test image to demonstrate the pipeline...")
        
        os.makedirs(input_dir, exist_ok=True)
        # Create a simple grey image with text
        mock_img = np.ones((480, 640, 3), dtype=np.uint8) * 120
        cv2.putText(mock_img, "Ekol Test Frame", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        test_path = os.path.join(input_dir, "test_image.jpg")
        cv2.imwrite(test_path, mock_img)
        image_paths.append(test_path)
        
    print(f"-> Found {len(image_paths)} image(s) to process.")
    
    # 3. Process each image sequentially
    for idx, img_path in enumerate(sorted(image_paths)):
        filename = os.path.basename(img_path)
        print(f"[{idx+1}/{len(image_paths)}] Processing: {filename}...")
        
        # Read the image frame
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"   Error: Could not read {filename}. Skipping.")
            continue
            
        # Run our unified computer vision and data scrubbing pipeline
        processed_frame = pipeline.process_frame(frame)
        
        # Save the processed image to output directory
        output_path = os.path.join(output_dir, f"processed_{filename}")
        cv2.imwrite(output_path, processed_frame)
        
    # 4. Generate and print the Final Performance Report
    report = pipeline.perf_tracker.generate_report()
    print("\n=== FINAL PERFORMANCE REPORT ===")
    for k, v in report.items():
        print(f"-> {k.replace('_', ' ').title()}: {v}")
    print("================================\n")
    print(f"Success! Processed images are saved in: '{output_dir}'")

if __name__ == "__main__":
    main()
