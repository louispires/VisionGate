#!/usr/bin/env python3
"""Test model with actual training images"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

def test_with_training_images():
    """Test the model with actual training images"""
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v3_large(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(960, 1280),
        nn.Hardswish(),
        nn.Dropout(0.3),  # Updated to match training config
        nn.Linear(1280, 2)
    )
    
    # Load best model
    model.load_state_dict(torch.load("models/gate_mobilenetv3_best.pth", map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    class_names = ["closed", "open"]
    
    # Test with actual images from training set
    test_dirs = [
        ("dataset/train/closed", "closed"),
        ("dataset/train/open", "open")
    ]
    
    print("🧪 Testing with actual training images:\n")
    
    for test_dir, expected_class in test_dirs:
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            correct = 0
            total = 0
            
            for i, img_file in enumerate(images[:50]):  # Test first 5 images
                img_path = os.path.join(test_dir, img_file)
                
                # Load and preprocess
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_idx = torch.argmax(output, dim=1).item()
                    predicted_class = class_names[predicted_idx]
                
                is_correct = predicted_class == expected_class
                if is_correct:
                    correct += 1
                total += 1
                
                print(f"{expected_class.upper()} Image {i+1}: {img_file}")
                print(f"  Expected: {expected_class}, Predicted: {predicted_class} {'✅' if is_correct else '❌'}")
                print(f"  Probabilities: closed={probabilities[0][0]:.3f}, open={probabilities[0][1]:.3f}")
                print(f"  Raw output: {output[0].cpu().numpy()}")
                print()
            
            print(f"{expected_class.upper()} Images: {correct}/{total} correct ({correct/total*100:.1f}%)\n")

if __name__ == "__main__":
    test_with_training_images()