import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

def test_final_model(model_path="../models/gate_mobilenetv3_best.pth", dataset_path="../dataset"):
    """
    Test the final trained model on the test set
    This should only be run ONCE after training is complete
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧪 Testing final model on: {device}")
    
    # Check if test set exists
    test_path = os.path.join(dataset_path, "test")
    if not os.path.exists(test_path):
        print("❌ Test set not found! Make sure you've split your dataset.")
        print("Run: python scripts/split_dataset.py")
        return
    
    # Load model architecture (same as training)
    model = models.mobilenet_v3_large()
    model.classifier = nn.Sequential(
        nn.Linear(960, 1280),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(1280, 2)
    )
    
    # Load trained weights
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Train your model first: python train_gate.py")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✅ Loaded model from: {model_path}")
    
    # Test transforms (same as validation - NO augmentation)
    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=16, 
        shuffle=False,  # Don't shuffle for consistent results
        num_workers=0
    )
    
    print(f"📊 Test set size: {len(test_dataset)} images")
    print(f"Classes: {test_dataset.classes}")
    
    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []
    correct = 0
    total = 0
    
    print("\n🔍 Running inference on test set...")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Progress indicator
            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx * len(inputs)}/{len(test_dataset)} images...")
    
    # Calculate final accuracy
    accuracy = 100 * correct / total
    
    print(f"\n🎯 FINAL TEST RESULTS:")
    print("=" * 50)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{total}")
    
    # Per-class accuracy
    class_names = test_dataset.classes
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    for i in range(len(all_labels)):
        label = all_labels[i]
        class_total[label] += 1
        if all_preds[i] == label:
            class_correct[label] += 1
    
    print(f"\n📊 Per-Class Results:")
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"  {class_name}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    # Detailed metrics
    print(f"\n📈 Detailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n🔍 Confusion Matrix:")
    print(f"{'':>12} {'Predicted':>20}")
    print(f"{'':>12} {class_names[0]:>10} {class_names[1]:>10}")
    print(f"Actual {class_names[0]:>6}: {cm[0][0]:>8} {cm[0][1]:>8}")
    print(f"       {class_names[1]:>6}: {cm[1][0]:>8} {cm[1][1]:>8}")
    
    # Confidence analysis
    all_probs = np.array(all_probs)
    confidence_scores = np.max(all_probs, axis=1)
    avg_confidence = np.mean(confidence_scores)
    
    print(f"\n🎲 Confidence Analysis:")
    print(f"Average confidence: {avg_confidence:.4f}")
    print(f"High confidence (>0.9): {np.sum(confidence_scores > 0.9)}/{len(confidence_scores)} ({100*np.sum(confidence_scores > 0.9)/len(confidence_scores):.1f}%)")
    print(f"Low confidence (<0.6): {np.sum(confidence_scores < 0.6)}/{len(confidence_scores)} ({100*np.sum(confidence_scores < 0.6)/len(confidence_scores):.1f}%)")
    
    # Save results
    results_file = "test_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Final Test Results - {model_path}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"Total Images: {total}\n")
        f.write(f"Correct: {correct}\n\n")
        f.write("Per-Class Results:\n")
        for i, class_name in enumerate(class_names):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                f.write(f"  {class_name}: {class_acc:.2f}%\n")
        f.write(f"\nAverage Confidence: {avg_confidence:.4f}\n")
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return {
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'per_class_accuracy': {class_names[i]: 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(len(class_names))},
        'avg_confidence': avg_confidence,
        'confusion_matrix': cm.tolist()
    }

def compare_models(model_paths=None):
    """
    Compare multiple models on the test set
    """
    if model_paths is None:
        model_paths = [
            "models/gate_mobilenetv3_best.pth",
            "models/gate_resnet101_best.pth"
        ]
    
    print("🏆 Model Comparison on Test Set:")
    print("=" * 60)
    
    results = {}
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"\nTesting: {model_path}")
            print("-" * 40)
            result = test_final_model(model_path)
            if result:
                results[model_path] = result
    
    # Summary comparison
    if len(results) > 1:
        print(f"\n🏆 COMPARISON SUMMARY:")
        print("-" * 40)
        for model_path, result in results.items():
            model_name = os.path.basename(model_path)
            print(f"{model_name}: {result['accuracy']:.2f}% accuracy")

if __name__ == "__main__":
    print("🚀 Final Model Testing")
    print("=" * 50)
    print("⚠️  This should only be run ONCE after training is complete!")
    print("The test set provides an unbiased estimate of real-world performance.\n")
    
    # Test the best model
    test_final_model()