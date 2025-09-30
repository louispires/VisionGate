import torch
import torch.nn as nn
from torchvision import models

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2

# Recreate the MobileNetV3 model architecture
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
    model_name = "models/gate_mobilenetv3_best"
    print("Loaded best MobileNetV3 model")
except FileNotFoundError:
    try:
        model.load_state_dict(torch.load("models/gate_mobilenetv3.pth", map_location=device))
        model_name = "models/gate_mobilenetv3"
        print("Loaded MobileNetV3 model")
    except FileNotFoundError:
        raise RuntimeError("No trained MobileNetV3 model found! Please train the model first.")

model = model.to(device)
model.eval()

# Export to ONNX with correct input size for MobileNetV3 (64x64)
dummy_input = torch.randn(1, 3, 64, 64, device=device)
output_path = f"{model_name}.onnx"

torch.onnx.export(model, dummy_input,
                  output_path,
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"},
                                "output": {0: "batch_size"}})
print(f"Model exported to ONNX format: {output_path}")