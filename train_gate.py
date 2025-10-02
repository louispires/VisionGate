import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, models, transforms

def main():
    # === Configuration ===
    data_dir = "dataset"   # root folder with train/ and val/
    batch_size = 32  # Good for 7,000 images
    num_epochs = 50  # Sufficient for larger dataset
    learning_rate = 0.0001  # Perfect for MobileNet
    weight_decay = 1e-4    # Good regularization
    num_classes = 2  # open, closed
    
    # Add early stopping
    patience = 5
    
    # Enhanced CUDA setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA not available. Using CPU (training will be much slower)")
        batch_size = 4  # Smaller batch size for MobileNet

    # === Data transforms ===
    # Using 64x64 resolution to match server preprocessing (square resize!)
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),     # Increase rotation for more variety
            transforms.ColorJitter(
                brightness=0.4,    # Increase for more lighting variations
                contrast=0.4,      # Increase for weather conditions
                saturation=0.3,    # Increase for seasonal changes
                hue=0.15          # Increase color shifts
            ),
            transforms.RandomGrayscale(p=0.15),       # Increase poor lighting simulation
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),  # More motion blur
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3),  # Add translation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # === Datasets & loaders ===
    image_datasets = {
        x: datasets.ImageFolder(root=f"{data_dir}/{x}",
                                transform=data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x],
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=0)  # Set to 0 for Windows compatibility
        for x in ["train", "val"]
    }
    class_names = image_datasets["train"].classes
    print("Classes:", class_names)

    # === Model ===
    # Using MobileNetV3 Large for efficient inference and better aspect ratio handling
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    # Replace final classifier for 2-class classification
    model.classifier = nn.Sequential(
        nn.Linear(960, 1280),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes)
    )
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # === Training loop ===
    best_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss, running_corrects = 0.0, 0
            
            # Clear GPU cache at start of each phase
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Print progress every 10 batches
                if batch_idx % 10 == 0:
                    current = batch_idx * len(inputs)
                    total = len(image_datasets[phase])
                    print(f"{phase} [{current:>5}/{total:>5}] Loss: {loss.item():.4f}")

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # Early stopping logic
            if phase == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), "models/gate_mobilenetv3_best.pth")
                    print(f"✅ New best validation accuracy: {best_acc:.4f}")
                else:
                    epochs_without_improvement += 1
                    print(f"⏳ No improvement for {epochs_without_improvement}/{patience} epochs")
                    
                # Check if we should stop early
                if epochs_without_improvement >= patience:
                    print(f"🛑 Early stopping triggered after {patience} epochs without improvement")
                    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch + 1}")
                    break
        
        # Break outer loop if early stopping triggered
        if epochs_without_improvement >= patience:
            break
        
        # GPU memory info
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Max GPU memory used: {memory_used:.2f} GB")

    # === Final results ===
    print(f"\n🎯 Training completed!")
    print(f"Best validation accuracy: {best_acc:.4f} achieved at epoch {best_epoch + 1}")
    print(f"Total epochs trained: {epoch + 1}")

    # === Save model ===
    torch.save(model.state_dict(), "models/gate_mobilenetv3.pth")
    print("Model saved as models/gate_mobilenetv3.pth")

    # === Export to ONNX ===
    model.eval()  # Set to evaluation mode for export
    dummy_input = torch.randn(1, 3, 64, 64, device=device)  # Updated for 64x64 input
    try:
        torch.onnx.export(model, dummy_input,
                          "models/gate_mobilenetv3.onnx",
                          input_names=["input"],
                          output_names=["output"],
                          dynamic_axes={"input": {0: "batch_size"},
                                        "output": {0: "batch_size"}})
        print("Model exported to ONNX: models/gate_mobilenetv3.onnx")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("Model still saved as PyTorch file")

if __name__ == '__main__':
    main()