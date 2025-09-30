from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import uvicorn
import time

# Set descriptive process title for monitoring tools
try:
    from setproctitle import setproctitle
    setproctitle("visiongate-mobilenetv3")
except ImportError:
    pass  # setproctitle not available, continue without it

# Load the trained MobileNetV3 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_classes = 2
model = models.mobilenet_v3_large(weights=None)  # No pretrained weights
model.classifier = nn.Sequential(
    nn.Linear(960, 1280),
    nn.Hardswish(),
    nn.Dropout(0.2),
    nn.Linear(1280, num_classes)
)

# Try to load the best model first, fallback to regular model
try:
    model.load_state_dict(torch.load("models/gate_mobilenetv3_best.pth", map_location=device))
    print("Loaded best MobileNetV3 model")
except FileNotFoundError:
    try:
        model.load_state_dict(torch.load("models/gate_mobilenetv3.pth", map_location=device))
        print("Loaded MobileNetV3 model")
    except FileNotFoundError:
        raise RuntimeError("No trained MobileNetV3 model found! Please train the model first.")

model = model.to(device)
model.eval()

# Classes must match training order
class_names = ["closed", "open"]

crop_box = (1280, 300, 1365, 800)

# Preprocessing transforms (must match training - 64x64 for MobileNetV3)
# Since training data is already 64x64, we only need resize + normalization
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Match training: square resize to 64x64
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

app = FastAPI(title="VisionGate API", description="MobileNetV3-based gate status classifier")

def preprocess(image: Image.Image):
    """Preprocess image for prediction"""
    img = image.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """Classify gate image as open or closed"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        start_time = time.time()
        
        # Load and preprocess image
        img = Image.open(file.file)
        input_tensor = preprocess(img)
        
        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, predicted = torch.max(outputs, 1)
            
        inference_time = time.time() - start_time
        
        result = {
            "status": class_names[predicted.item()],
            "confidence": float(probabilities[predicted.item()]),
            "probabilities": {
                "closed": float(probabilities[0]),
                "open": float(probabilities[1])
            },
            "inference_time_ms": round(inference_time * 1000, 2),
            "model": "ResNet101",
            "image_size": list(img.size)
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Gate Classification API - ResNet101", 
        "status": "ready",
        "model": "ResNet101",
        "input_size": "384x384",
        "classes": class_names,
        "device": str(device)
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)