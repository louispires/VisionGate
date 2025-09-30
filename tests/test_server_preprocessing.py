#!/usr/bin/env python3
"""Quick test of the fixed server without running it"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

def test_server_preprocessing():
    """Test that server preprocessing matches training"""
    
    # Load model (same as server)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v3_large(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(960, 1280),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(1280, 2)
    )
    model.load_state_dict(torch.load("gate_mobilenetv3_best.pth", map_location=device))
    model = model.to(device)
    model.eval()
    
    # Server preprocessing (fixed version)
    server_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Match training: square resize to 64x64
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    class_names = ["closed", "open"]
    
    # Test with training images
    test_dirs = [
        ("dataset/train/closed", "closed"),
        ("dataset/train/open", "open")
    ]
    
    print("🔧 Testing FIXED server preprocessing:\n")
    
    for test_dir, expected_class in test_dirs:
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Test first image
            img_file = images[0]
            img_path = os.path.join(test_dir, img_file)
            
            # Load image
            img = Image.open(img_path).convert("RGB")
            print(f"Testing {expected_class.upper()} image: {img_file}")
            print(f"Original size: {img.size}")
            
            # Apply server preprocessing
            img_tensor = server_transform(img).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_idx = torch.argmax(output, dim=1).item()
                predicted_class = class_names[predicted_idx]
            
            is_correct = predicted_class == expected_class
            print(f"Expected: {expected_class}, Predicted: {predicted_class} {'✅' if is_correct else '❌'}")
            print(f"Probabilities: closed={probabilities[0][0]:.3f}, open={probabilities[0][1]:.3f}")
            print(f"Raw output: {output[0].cpu().numpy()}")
            print()

if __name__ == "__main__":
    test_server_preprocessing()