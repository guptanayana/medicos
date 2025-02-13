from fastapi import FastAPI, HTTPException
from transformers import MobileViTForImageClassification, AutoImageProcessor
from PIL import Image
import torch
from pydantic import BaseModel
import numpy as np

checkpoint_path = "/home/yashwardhan/Downloads/oral_disease_mobilevit/checkpoint-5830"
model = MobileViTForImageClassification.from_pretrained(checkpoint_path)
image_processor = AutoImageProcessor.from_pretrained(checkpoint_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

LABELS = ['Calculus', 'Data caries', 'Gingivitis', 'Mouth Ulcer', 'Tooth Discoloration', 'hypodontia']

# FastAPI app
app = FastAPI(
    title="Oral Disease Detection API",
    description="API for detecting oral diseases using MobileViT model",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    image_path: str

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Load and process image
        image = Image.open(request.image_path)
        inputs = image_processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()

        all_probs = {LABELS[i]: float(prob) for i, prob in enumerate(probabilities[0])}

        return PredictionResponse(
            predicted_class=LABELS[predicted_class_idx],
            confidence=confidence,
            all_probabilities=all_probs
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Oral Disease Detection API",
        "available_labels": LABELS
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
