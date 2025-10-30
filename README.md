# 🚪 Gate Classification Model

# VisionGate

A deep learning project for classifying train gate states (open/closed) using computer vision. This project implements a MobileNetV3-based classifier optimized for real-time inference on Intel hardware.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Training](#-training)
- [Inference Servers](#-inference-servers)
- [Docker Deployment](#-docker-deployment)
- [Testing](#-testing)
- [Model Performance](#-model-performance)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## 🎯 Project Overview

This system classifies gate images as either **"open"** or **"closed"** using a fine-tuned MobileNetV3-Large model. The project supports multiple deployment scenarios:

- **PyTorch inference** (CPU/CUDA/Intel XPU)
- **ONNX Runtime** with Intel OpenVINO optimization
- **Docker containers** for production deployment
- **REST API** for easy integration

### Key Specifications:
- **Input**: 64x64 RGB images
- **Architecture**: MobileNetV3-Large (4.2M parameters)
- **Accuracy**: 98.88% training, 89.60% validation
- **Memory**: ~0.64 GB GPU usage
- **Classes**: `["closed", "open"]`

## ✨ Features

- 🚀 **High Performance**: MobileNetV3 optimized for edge deployment
- 🔧 **Multiple Backends**: PyTorch, ONNX Runtime, Intel OpenVINO
- 🐳 **Docker Ready**: Production-ready containers
- 📊 **Comprehensive Testing**: Debug tools and API tests
- 🖥️ **Multi-Platform**: CUDA, Intel ARC GPU, CPU support
- 📈 **Monitoring**: GPU usage tracking with `intel_gpu_top`
- 🎛️ **Process Management**: Named processes with `setproctitle`

## 📁 Project Structure

```
VisionGate/
├── 📄 README.md                 # This file
├── 📄 requirements.txt          # Python dependencies
├── 🧠 train_gate.py            # Main training script
│
├── 📁 dataset/                  # Training data
│   ├── train/
│   │   ├── closed/             # Closed gate images
│   │   └── open/               # Open gate images
│   └── val/
│       ├── closed/
│       └── open/
│
├── 📁 models/                   # Trained models
│   ├── gate_mobilenetv3.pth           # Latest model
│   ├── gate_mobilenetv3_best.pth      # Best checkpoint
│   ├── gate_mobilenetv3.onnx          # ONNX export
│   └── gate_mobilenetv3_best.onnx     # Best ONNX model
│
├── 📁 servers/                  # Inference servers
│   ├── server_mobilenetv3.py          # Standard PyTorch server
│   ├── server_mobilenetv3_intel.py    # Intel XPU optimized
│   └── server_mobilenetv3_onnx.py     # ONNX Runtime server
│
├── 📁 docker/                   # Docker deployment
│   ├── Dockerfile.onnx.intel          # Intel GPU container
│   ├── .dockerignore                  # Docker ignore rules
│   └── DOCKER_README.md               # Docker documentation
│
├── 📁 scripts/                  # Utility scripts
│   ├── export_onnx.py                 # ONNX model export
│   └── verify_onnx.py                 # ONNX verification
│
├── 📁 tests/                    # Testing & debugging
│   ├── test_api.ps1                   # PowerShell API tests
│   ├── test_curl.ps1                  # cURL tests
│   ├── quick_test.ps1                 # Quick health checks
│   ├── debug_model.py                 # Model debugging
│   ├── test_real_images.py            # Training data tests
│   └── test_server_preprocessing.py   # Preprocessing tests
│
├── 📁 docs/                     # Documentation
└── 📁 OLDModels/               # Legacy model archives
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone and setup
cd TrainGateModel
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### 2. Train Model (if needed)
```bash
python train_gate.py
```

### 3. Start Server
```bash
# Standard PyTorch server
python servers/server_mobilenetv3.py

# Intel optimized server  
python servers/server_mobilenetv3_intel.py

# ONNX server
python servers/server_mobilenetv3_onnx.py
```

### 4. Test API
```powershell
# PowerShell test
./tests/test_api.ps1

# Quick health check
./tests/quick_test.ps1
```

## 📦 Installation

### Prerequisites
- **Python 3.13+** (fully supported)
- **AMD Ryzen 9 9950X3D** (or any modern CPU)
- **Docker** (for containerized deployment)

### Core Dependencies
```bash
# PyTorch with CPU support (optimized for AMD Ryzen)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Server and inference dependencies
pip install fastapi uvicorn pillow numpy python-multipart setproctitle

# ONNX Runtime with OpenVINO for inference optimization
pip install onnxruntime-openvino openvino onnx onnxscript
```

### Full Installation
```bash
# Create virtual environment with Python 3.13
py -3.13 -m venv .venv
.venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

## 🎓 Training

### Dataset Structure
Organize your images in the following structure:
```
dataset/
├── train/
│   ├── closed/    # Closed gate images
│   └── open/      # Open gate images
└── val/
    ├── closed/    # Validation closed images
    └── open/      # Validation open images
```

### Training Command
```bash
python train_gate.py
```

### Training Configuration
```python
# Key parameters in train_gate.py
batch_size = 16        # Batch size for training
num_epochs = 20        # Training epochs
learning_rate = 0.001  # Learning rate
input_size = (64, 64)  # Input image resolution
```

### Training Output
- `models/gate_mobilenetv3.pth` - Final model
- `models/gate_mobilenetv3_best.pth` - Best checkpoint
- `models/gate_mobilenetv3.onnx` - ONNX export

## 🖥️ Inference Servers

### 1. Standard PyTorch Server
```bash
python servers/server_mobilenetv3.py
```
- **Port**: 8000
- **Backend**: PyTorch (CUDA/CPU)
- **Memory**: ~0.64 GB GPU

### 2. Intel Optimized Server
```bash
python servers/server_mobilenetv3_intel.py
```
- **Port**: 8000
- **Backend**: Intel Extension for PyTorch
- **GPU**: Intel ARC optimized
- **Process**: `gate-classifier-mobilenetv3-intel`

### 3. ONNX Runtime Server
```bash
python servers/server_mobilenetv3_onnx.py
```
- **Port**: 8000
- **Backend**: ONNX Runtime with OpenVINO EP
- **Optimization**: Intel GPU acceleration
- **Process**: `gate-classifier-mobilenetv3-onnx`

### API Endpoints

#### Health Check
```http
GET /health
```
Response:
```json
{
  "status": "healthy",
  "model": "MobileNetV3",
  "providers": ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
}
```

#### Classification
```http
POST /classify
Content-Type: multipart/form-data
```
Request:
```
file: [image file]
```
Response:
```json
{
  "prediction": "open",
  "confidence": 0.9995,
  "probabilities": {
    "closed": 0.0005,
    "open": 0.9995
  },
  "processing_time": 0.023
}
```

## 🐳 Docker Deployment

### Build Container
```bash
# Intel GPU optimized container
docker build -f docker/Dockerfile.onnx.intel -t valiente/gate-classifier-onnx-intel:latest .
```

### Run Container
```bash
# Standard run
docker run -d --name gate-classifier-onnx -p 8000:8000 valiente/gate-classifier-onnx-intel:latest

# With GPU access (Intel)
docker run -d --device=/dev/dri --name gate-classifier-onnx -p 8000:8000 valiente/gate-classifier-onnx-intel:latest
```

### Deploy to Docker Hub
```bash
# Build and push
docker build -f docker/Dockerfile.onnx.intel -t valiente/gate-classifier-onnx-intel:latest .
docker push valiente/gate-classifier-onnx-intel:latest
```

## 🧪 Testing

### API Testing
```powershell
# Comprehensive API test
./tests/test_api.ps1

# Quick health check
./tests/quick_test.ps1

# cURL tests
./tests/test_curl.ps1
```

### Model Debugging
```bash
# Debug model outputs
python tests/debug_model.py

# Test with training images
python tests/test_real_images.py

# Verify preprocessing
python tests/test_server_preprocessing.py
```

### ONNX Verification
```bash
# Verify ONNX model
python scripts/verify_onnx.py
```

## 📊 Model Performance

### Training Results
- **Training Accuracy**: 98.88%
- **Validation Accuracy**: 89.60%
- **Training Loss**: 0.0197
- **Validation Loss**: 0.2704
- **Parameters**: 4,204,594 (4.2M)
- **GPU Memory**: 0.64 GB

### Inference Performance
- **Preprocessing**: ~2ms
- **Inference**: ~20ms (GPU), ~50ms (CPU)
- **Total Response**: ~25ms (including JSON serialization)

### Model Comparison
| Model | Parameters | Memory (GB) | Accuracy | Speed |
|-------|------------|-------------|----------|-------|
| MobileNetV3 | 4.2M | 0.64 | 98.88% | ⭐⭐⭐⭐⭐ |
| ResNet101 | 42M | ~8.0 | 100%* | ⭐⭐⭐ |

*ResNet101 showed overfitting issues with narrow images

## 🔧 Troubleshooting

### Common Issues

#### 1. Server Always Returns "Closed"
**Problem**: Preprocessing mismatch between training and inference
**Solution**: Ensure server preprocessing matches training (64x64 square resize only)

#### 2. CUDA Out of Memory
**Problem**: Batch size too large for GPU
**Solution**: Reduce `batch_size` in `train_gate.py`

#### 3. Intel GPU Not Detected
**Problem**: Intel Extension not properly installed
**Solution**: 
```bash
pip install intel-extension-for-pytorch==2.8.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

#### 4. Docker Build Fails
**Problem**: Missing model files
**Solution**: Ensure ONNX model exists before building:
```bash
python scripts/export_onnx.py
```

#### 5. Port Already in Use
**Problem**: Another service using port 8000
**Solution**: 
```bash
# Check what's using the port
netstat -ano | findstr :8000

# Or use different port in server code
uvicorn.run(app, host='0.0.0.0', port=8001)
```

### Debug Commands
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check Intel XPU
python -c "import intel_extension_for_pytorch as ipex; print(f'XPU: {torch.xpu.is_available()}')"

# Test model loading
python tests/debug_model.py

# Verify ONNX export
python scripts/verify_onnx.py
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests: `python tests/test_real_images.py`
5. Submit pull request

### Adding New Features
- **New architectures**: Update `train_gate.py` model section
- **New servers**: Add to `servers/` directory
- **New tests**: Add to `tests/` directory
- **Docker variants**: Add to `docker/` directory

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent framework
- **Intel** for OpenVINO and Intel Extension for PyTorch
- **FastAPI** for the web framework
- **MobileNetV3** architecture creators

---

**Built with ❤️ for reliable gate monitoring**

For questions or support, please open an issue in the repository.