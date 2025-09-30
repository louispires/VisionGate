from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import onnxruntime as ort
import time
import os
from typing import Dict

# Set descriptive process title for monitoring tools
try:
    from setproctitle import setproctitle
    setproctitle("visiongate-mobilenetv3-onnx")
except ImportError:
    pass  # setproctitle not available, continue without it

# Configuration
MODEL_PATH = os.getenv("ONNX_MODEL", "models/gate_mobilenetv3.onnx")
INPUT_SIZE = (64, 64)
CLASS_NAMES = ["closed", "open"]
CROP_BOX = (1280, 300, 1365, 800)

# Session options
so = ort.SessionOptions()
so.enable_mem_pattern = True
so.enable_cpu_mem_arena = True
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Attempt to load with OpenVINO or DML / CPU fallback
preferred_eps = [
    ("OpenVINOExecutionProvider", {"device_type": "GPU_FP32"}),
    ("DmlExecutionProvider", {}),  # Windows DirectML fallback
    ("CUDAExecutionProvider", {}),
    ("CPUExecutionProvider", {})
]

available_eps = ort.get_available_providers()
selected_providers = []
for ep in preferred_eps:
    name = ep[0]
    if name in available_eps:
        if len(ep) > 1 and isinstance(ep[1], dict) and ep[1]:
            selected_providers.append((name, ep[1]))
        else:
            selected_providers.append(name)

if not selected_providers:
    raise RuntimeError(f"No suitable ONNX Runtime providers found. Available: {available_eps}")

print("Available providers:", available_eps)
print("Using providers (in priority order):", selected_providers)

# Create inference session
session = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=selected_providers)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

app = FastAPI(title="VisionGate API - ONNXRuntime (Intel GPU preferred)")

def preprocess(image: Image.Image) -> np.ndarray:
    """Preprocess image exactly matching training pipeline"""
    # Convert to RGB and resize to match training (no crop/rotation needed)
    img = image.convert("RGB")
    img = img.resize(INPUT_SIZE)  # Resize to 64x64 to match training
    
    # Convert to numpy and normalize
    arr = np.array(img).astype(np.float32) / 255.0
    
    # Apply ImageNet normalization (same as training)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    
    # Convert to CHW format and add batch dimension
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    arr = np.expand_dims(arr, 0)  # NCHW
    return arr

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        start = time.time()
        img = Image.open(file.file)
        inp = preprocess(img)
        outputs = session.run([output_name], {input_name: inp})
        logits = outputs[0]
        probs = softmax(logits[0])
        pred_idx = int(np.argmax(probs))
        latency = (time.time() - start) * 1000
        return {
            "status": CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
            "latency_ms": round(latency, 2),
            "providers": session.get_providers(),
            "model": os.path.basename(MODEL_PATH)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Gate Classification ONNXRuntime Service",
        "model": os.path.basename(MODEL_PATH),
        "providers": session.get_providers(),
        "input_size": INPUT_SIZE,
        "classes": CLASS_NAMES
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "providers": session.get_providers()}

def softmax(x):
    x = np.asarray(x)
    e = np.exp(x - np.max(x))
    return e / e.sum()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
