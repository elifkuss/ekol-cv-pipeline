import cv2  
import time
from src.pipeline import EkolCVPipeline

def main():
    print("\n=== EKOL LOGISTICS REAL-TIME RUNNER ===\n")
    
    # 1. Pipeline'ı başlatın (İnternet yoksa lokaldeki yolov8n.pt dosyasını kullanır)
    pipeline = EkolCVPipeline(model_path="yolov8n.pt")
    
    # 2. WMS'den gelen hedef siparişi tanımlayın
    pipeline.state_machine.set_active_order("TIE") # Örn: Gömlek siparişi
    
    # 3. Gerçek videoyu okuyun
    video_path = "data/packaging_depo.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    print(f"-> Processing video: {video_path}...")
    
    # Video çıktı ayarları (Eğer işlenmiş videoyu kaydetmek isterseniz)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fps = int(cap.get(cv2.CAP_PROP_FPS))             
    out = cv2.VideoWriter('data/output_processed.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Her bir kareyi bizim hazırladığımız modüler boru hattından geçirin
        processed_frame = pipeline.process_frame(frame)
        
        # İşlenmiş kareyi yeni videoya yazın
        out.write(processed_frame)
        
        # (Opsiyonel) Ekranı olan bir bilgisayardaysanız anlık görüntüyü izleyin:
        # cv2.imshow("Ekol Live Packaging Monitor", processed_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Kaynakları serbest bırakın
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # 4. Performans ve Anomali Raporunu yazdırın
    report = pipeline.perf_tracker.generate_report()
    print("\n=== FINAL PERFORMANCE REPORT ===")
    for k, v in report.items():
        print(f"-> {k.replace('_', ' ').title()}: {v}")
    print("================================\n")

if __name__ == "__main__":
    main()