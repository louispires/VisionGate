# VisionGate — Agent Guide

Binary image classifier for train gate states (open/closed) using MobileNetV3-Large. See [README.md](README.md) for full project overview and [docs/](docs/) for API and model documentation.

## Architecture

```
Input Image → Resize 64×64 → ImageNet Normalize → MobileNetV3-Large → [closed, open]
```

- **Input:** 64×64 RGB — non-negotiable; all pipelines must use this size
- **Normalization:** mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- **Classes:** `["closed", "open"]` — alphabetical (ImageFolder order); index 0 = closed, 1 = open
- **Custom head:** `Linear(960→1280) → Hardswish → Dropout(0.3) → Linear(1280→2)`

## Key Commands

```powershell
# Setup
python -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt

# Train
python train_gate.py

# Serve (preferred: ONNX + OpenVINO/DirectML)
python servers/server_mobilenetv3_onnx.py

# Serve (PyTorch — better for debugging)
python servers/server_mobilenetv3.py

# Export model to ONNX
python scripts/export_onnx.py

# Verify ONNX model
python scripts/verify_onnx.py

# Quick API health check
.\tests\quick_test.ps1

# Full API test suite
.\tests\test_api.ps1

# Docker build & push
.\deploy.ps1
```

## Codebase Layout

| Path | Purpose |
|------|---------|
| `train_gate.py` | Training script (MobileNetV3, DirectML/CUDA/CPU) |
| `servers/server_mobilenetv3_onnx.py` | **Production server** — ONNX + OpenVINO/DirectML |
| `servers/server_mobilenetv3.py` | Dev/debug PyTorch server |
| `servers/server_pytorch.py` | Generic PyTorch server variant |
| `scripts/export_onnx.py` | Export `.pth` → `.onnx` |
| `scripts/verify_onnx.py` | Validate ONNX model output |
| `scripts/analyze_dataset.py` | Dataset class balance check |
| `scripts/split_dataset.py` | Split raw images into train/val/test |
| `models/` | Saved checkpoints (`.pth`) and exports (`.onnx`) |
| `dataset/` | `train/`, `val/`, `test/` — each with `closed/` and `open/` subdirs |
| `config.ini` | Reference config (most servers hardcode values; use env vars to override) |
| `docker/Dockerfile.onnx.intel` | Production container (OpenVINO 2025.3.0, Intel GPU) |
| `docs/API.md` | REST API reference |
| `docs/MODEL.md` | Model architecture details |

## Inference Server

The ONNX server (`servers/server_mobilenetv3_onnx.py`) auto-selects the best provider:
1. `OpenVINOExecutionProvider` — Intel GPU with `device_type: GPU` (fastest)
2. `DmlExecutionProvider` — Windows AMD/Intel GPU (DirectML)
3. `CUDAExecutionProvider` — NVIDIA GPU
4. `CPUExecutionProvider` — fallback

Override model path: `ONNX_MODEL=models/gate_mobilenetv3_best.onnx python servers/server_mobilenetv3_onnx.py`

**API:** `POST /classify` (multipart `file`), `GET /health`, `GET /`

## Training Conventions

- Device priority: DirectML GPU → CUDA → CPU
- `num_workers=0` always — required on Windows (pickling errors otherwise)
- Best checkpoint auto-saved to `models/gate_mobilenetv3_best.pth`; final to `models/gate_mobilenetv3.pth`
- ONNX auto-exported at end of training
- `cpu_threads = 16` hardcoded for AMD Ryzen 9 9950X3D — adjust for other hardware

## Known Pitfalls

- **numpy version:** Must be `<2.3.0`. Do NOT run `pip install --upgrade numpy`; it breaks OpenCV/OpenVINO
- **config.ini is mostly reference:** Servers hardcode most values. To change behavior, edit server files directly or use env vars
- **`crop_box` and `rotation_angle`** in config.ini are legacy/deprecated — not used in current 64×64 pipeline
- **Docker registry:** `valiente/gate-classifier-onnx-intel` — see `deploy.ps1` and `config.ini [docker]`

## Dataset Structure

```
dataset/
├── train/closed/   ← Training images of closed gates
├── train/open/     ← Training images of open gates
├── val/closed/
├── val/open/
├── test/closed/
└── test/open/
```

Use `scripts/analyze_dataset.py` to check class balance before training.
