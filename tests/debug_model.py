#!/usr/bin/env python3
"""Debug script to test the trained MobileNetV3 model"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

def load_model():
    """Load the trained MobileNetV3 model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model architecture
    model = models.mobilenet_v3_large(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(960, 1280),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(1280, 2)  # 2 classes
    )
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load("gate_mobilenetv3_best.pth", map_location=device))
        print("✅ Loaded best MobileNetV3 model")
        model_name = "best"
    except FileNotFoundError:
        try:
            model.load_state_dict(torch.load("gate_mobilenetv3.pth", map_location=device))
            print("✅ Loaded MobileNetV3 model")
            model_name = "regular"
        except FileNotFoundError:
            print("❌ No trained model found!")
            return None, None, None
    
    model = model.to(device)
    model.eval()
    
    return model, device, model_name

def test_model_outputs(model, device):
    """Test what the model outputs for various inputs"""
    print("\n🔍 Testing model outputs:")
    
    # Test with different inputs
    test_cases = [
        "Random noise",
        "All zeros", 
        "All ones"
    ]
    
    inputs = [
        torch.randn(1, 3, 64, 64),  # Random
        torch.zeros(1, 3, 64, 64),  # Zeros
        torch.ones(1, 3, 64, 64)    # Ones
    ]
    
    class_names = ["closed", "open"]
    
    with torch.no_grad():
        for i, (name, input_tensor) in enumerate(zip(test_cases, inputs)):
            input_tensor = input_tensor.to(device)
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            
            print(f"{i+1}. {name}:")
            print(f"   Raw output: {output[0].cpu().numpy()}")
            print(f"   Probabilities: {probabilities[0].cpu().numpy()}")
            print(f"   Predicted: {class_names[predicted_class]} (index: {predicted_class})")
            print()

def test_image_preprocessing():
    """Test the preprocessing pipeline"""
    print("\n🖼️ Testing preprocessing pipeline:")
    
    # Check if we have test images
    test_dirs = ["dataset/train/closed", "dataset/train/open", "TEST"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                test_image = os.path.join(test_dir, images[0])
                print(f"Found test image: {test_image}")
                
                # Load and check image
                img = Image.open(test_image)
                print(f"Original image size: {img.size}")
                print(f"Original image mode: {img.mode}")
                
                # Apply preprocessing
                transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                img_tensor = transform(img.convert("RGB")).unsqueeze(0)
                print(f"Processed tensor shape: {img_tensor.shape}")
                print(f"Tensor range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
                print(f"Tensor mean: {img_tensor.mean():.3f}")
                print()
                break

if __name__ == "__main__":
    print("🐛 MobileNetV3 Model Debug Script")
    print("=" * 50)
    
    # Load model
    model, device, model_name = load_model()
    if model is None:
        exit(1)
    
    print(f"Model type: {model_name}")
    
    # Test model outputs
    test_model_outputs(model, device)
    
    # Test preprocessing
    test_image_preprocessing()
    
    print("✅ Debug complete!")