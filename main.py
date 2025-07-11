from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model.predictor import predict_disease
from typing import Dict

app = FastAPI()

# Frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # หรือใช้ ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_image(image: UploadFile = File(...)) -> Dict:
    image_bytes = await image.read()
    result = predict_disease(image_bytes)
    return result
