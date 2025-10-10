# VisionGate - Docker Deployment

A MobileNetV3-based gate classification service containerized with Docker and GPU support.

## 🚀 Quick Start

### Prerequisites
- Docker installed
- NVIDIA Docker runtime (for GPU support)
- Trained model file (`gate_mobilenetv3.pth`)

### Build and Run

**Option 1: Using Docker Compose (Recommended)**
```bash
docker-compose up --build
```

**Option 2: Using Build Scripts**
```bash
# Linux/Mac
chmod +x deploy.sh
./deploy.sh

# Windows
deploy.bat
```

**Option 3: Manual Docker Commands**
```bash
# Build
docker build -t gate-classifier .

# Run
docker run -d --name gate-classifier-app --gpus all -p 8000:8000 gate-classifier
```

## 🌐 API Endpoints

- **Health Check**: `GET /health`
- **Classification**: `POST /classify` (upload image)
- **API Info**: `GET /`
- **Interactive Docs**: `http://localhost:8000/docs`

## 📋 Usage Example

```python
import requests

# Test the API
url = "http://localhost:8000/classify"
with open("gate_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    print(response.json())
```

## 🔧 Configuration

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: GPU device selection (default: 0)

### Model Files Required
- `gate_mobilenetv3.pth` (required)
- `gate_mobilenetv3_best.pth` (optional, preferred if available)

## 📊 Container Specs
- **Base Image**: nvidia/cuda:12.1-runtime-ubuntu20.04
- **Python**: 3.8+
- **PyTorch**: CUDA 12.1 compatible
- **Input Size**: 384x384 pixels
- **Port**: 8000

## 🛠 Development Commands

```bash
# View logs
docker logs -f gate-classifier-app

# Shell access
docker exec -it gate-classifier-app /bin/bash

# Stop container
docker stop gate-classifier-app

# Remove container
docker rm gate-classifier-app
```

## 🎯 Performance
- **GPU Inference**: ~50-100ms per image
- **CPU Inference**: ~1-3s per image
- **Model Size**: ~170MB
- **Memory Usage**: ~2-4GB GPU memory