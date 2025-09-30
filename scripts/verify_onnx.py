"""Verification utility for ONNX model & runtime providers.
Run: python verify_onnx.py
"""
import os
import onnx
import onnxruntime as ort
import numpy as np

MODEL = os.getenv("ONNX_MODEL", "gate_resnet101.onnx")
EXPECTED_INPUT_SHAPE = (1, 3, 384, 384)
CLASS_COUNT = 2

print(f"== ONNX Model Verification ==")
print(f"Model path: {MODEL}")

# 1. Structural validation
model_proto = onnx.load(MODEL)
onxx_check = None
try:
    onnx.checker.check_model(model_proto)
    print("[OK] Model structure is valid ONNX")
except Exception as e:
    print("[FAIL] ONNX structure invalid:", e)
    raise SystemExit(1)

# 2. List model inputs/outputs
inputs = [(i.name, i.type.tensor_type.shape) for i in model_proto.graph.input]
outputs = [(o.name, o.type.tensor_type.shape) for o in model_proto.graph.output]
print("Inputs:")
for n, s in inputs:
    dims = [d.dim_param or d.dim_value for d in s.dim]
    print(f"  - {n}: {dims}")
print("Outputs:")
for n, s in outputs:
    dims = [d.dim_param or d.dim_value for d in s.dim]
    print(f"  - {n}: {dims}")

# 3. Runtime provider availability
available = ort.get_available_providers()
print("Available providers:", available)

preferred = [
    ("OpenVINOExecutionProvider", {"device_type": "GPU_FP32"}),
    ("OpenVINOExecutionProvider", {"device_type": "GPU"}),
    ("OpenVINOExecutionProvider", {"device_type": "CPU_FP32"}),
    ("DmlExecutionProvider", {}),
    ("CUDAExecutionProvider", {}),
    ("CPUExecutionProvider", {})
]

selected = []
for spec in preferred:
    name = spec[0]
    if name in available:
        if len(spec) > 1 and isinstance(spec[1], dict) and spec[1]:
            selected.append((name, spec[1]))
        else:
            selected.append(name)

if not selected:
    print("[WARN] No preferred EPs available, defaulting to CPU")
    selected = ["CPUExecutionProvider"]

print("Using providers:", selected)

# 4. Create session & run a dummy inference
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(MODEL, providers=selected, sess_options=so)

inp = session.get_inputs()[0]
out = session.get_outputs()[0]
print(f"Session input name: {inp.name}, shape: {inp.shape}")
print(f"Session output name: {out.name}, shape: {out.shape}")

# Dummy data (handle dynamic batch)
shape = [d if isinstance(d, int) else 1 for d in inp.shape]
if len(shape) != 4:
    print("[WARN] Unexpected input rank, expected 4")

# Ensure correct size for 384 variant
if shape[2] in (224, '224') or shape[3] in (224, '224'):
    print('[INFO] Model appears 224-sized, adjusting test tensor to 224x224')
    test_h, test_w = 224, 224
else:
    test_h, test_w = 384, 384

x = np.random.rand(1, 3, test_h, test_w).astype(np.float32)
res = session.run([out.name], {inp.name: x})
logits = res[0]
print("Inference OK. Output shape:", logits.shape)

if logits.shape[1] != CLASS_COUNT:
    print(f"[WARN] Output second dimension {logits.shape[1]} != expected {CLASS_COUNT}")
else:
    print("[OK] Output class count matches")

print("== Verification complete ==")
