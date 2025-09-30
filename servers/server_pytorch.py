from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import uvicorn

# Load the trained PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("gate_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

# Classes must match training order
class_names = ["closed", "open"]

# Preprocessing transforms (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

app = FastAPI()

def preprocess(image: Image.Image):
    """Preprocess image for prediction"""
    img = image.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """Classify gate image as open or closed"""
    img = Image.open(file.file)
    input_tensor = preprocess(img)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    result = {
        "status": class_names[predicted.item()],
        "confidence": float(confidence[predicted.item()]),
        "probabilities": {
            "closed": float(confidence[0]),
            "open": float(confidence[1])
        }
    }
    
    return result

@app.get("/")
async def root():
    return {"message": "Gate Classification API", "status": "ready"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)