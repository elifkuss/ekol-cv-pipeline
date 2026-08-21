import cv2
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from src.pipeline import EkolCVPipeline

app = FastAPI(title="Ekol CV Pipeline")

# Pipeline'ı başlat (Model bir kez belleğe yüklenir)
pipeline = EkolCVPipeline(model_path="yolov8n.pt")
pipeline.state_machine.set_active_order("TIE")

# Çıktı klasörünü oluştur
os.makedirs("output", exist_ok=True)

@app.get("/")
def home():
    return {"message": "Ekol CV API is running!"}


@app.post("/process_video")
def process_video(file: UploadFile = File(...)):
    """
    Video yükler, işler ve işlenmiş videoyu döndürür.
    """
    # 1. Videoyu kaydet (await yerine senkron okuma kullanıldı)
    input_path = f"output/input_{file.filename}"
    with open(input_path, "wb") as buffer:
        buffer.write(file.file.read())
    
    # 2. Videoyu işle
    output_path = f"output/processed_{file.filename}"
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # mp4v formatında videoyu kaydetme ayarı
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Senin yazdığın yapay zeka boru hattından (pipeline) geçir
        processed = pipeline.process_frame(frame)
        out.write(processed)
    
    # Kaynakları serbest bırak
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # 3. Raporu al ve terminal ekranına yazdır
    report = pipeline.perf_tracker.generate_report()
    print("\n=== PERFORMANS RAPORU ===")
    for k, v in report.items():
        print(f"{k}: {v}")
    print("=========================\n")
    
    # 4. İşlenmiş videoyu doğrudan kullanıcıya (tarayıcıya/postman'e) döndür
    return FileResponse(output_path, media_type="video/mp4", filename=f"processed_{file.filename}")

if __name__ == "__main__":
    import uvicorn
    # Çalıştırmak için: python api.py
    uvicorn.run(app, host="0.0.0.0", port=8000)