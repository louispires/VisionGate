from fastapi import FastAPI, UploadFile, File
from openvino.runtime import Core
from PIL import Image
import numpy as np
import uvicorn

# === Load OpenVINO model ===
ie = Core()
model = ie.read_model("gate_resnet18.xml")   # from OpenVINO optimizer
compiled_model = ie.compile_model(model, "GPU")
output_layer = compiled_model.outputs[0]

# Classes must match training order (ImageFolder sorts alphabetically)
class_names = ["closed", "open"]

app = FastAPI()

def preprocess(image: Image.Image):
    img = image.resize((224, 224)).convert("RGB")
    arr = np.array(img).transpose(2, 0, 1)[None, :].astype(np.float32) / 255.0
    return arr

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    img = Image.open(file.file)
    input_tensor = preprocess(img)
    result = compiled_model(input_tensor)[output_layer]
    pred = int(np.argmax(result))
    return {"status": class_names[pred]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
