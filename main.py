import os
from fastapi import FastAPI, HTTPException
from transformers import MobileViTForImageClassification, AutoImageProcessor
from PIL import Image
import torch
from pydantic import BaseModel
import numpy as np
import requests
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

# Get port from environment variable (Railway sets this)
PORT = int(os.environ.get("PORT", 8000))

# Model loading from Hugging Face Hub
MODEL_ID = os.environ.get("MODEL_ID", "YOUR-USERNAME/YOUR-MODEL-NAME")
# Set use_auth_token if your model is private
USE_AUTH_TOKEN = os.environ.get("HF_TOKEN", None)

# Load model and processor from Hugging Face
model = MobileViTForImageClassification.from_pretrained(
    MODEL_ID, 
    use_auth_token=USE_AUTH_TOKEN
)
image_processor = AutoImageProcessor.from_pretrained(
    MODEL_ID,
    use_auth_token=USE_AUTH_TOKEN
)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        # Download image from URL
        response = requests.get(request.image_path)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Create image from the downloaded bytes
        image = Image.open(BytesIO(response.content))
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
        "available_labels": LABELS,
        "status": "online"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    
    # Railway requires binding to 0.0.0.0 and using the PORT environment variable
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)