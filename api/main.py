"""
FastAPI application for TruthGuard-AI
Provides RESTful API for fact-checking predictions
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import config

# Initialize FastAPI app
app = FastAPI(
    title="TruthGuard-AI API",
    description="Transformer-based Multi-Class Fact Verification System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None


class FactCheckRequest(BaseModel):
    """Request model for fact-checking"""
    statement: str = Field(..., description="The statement to verify", min_length=1)
    return_probabilities: bool = Field(False, description="Whether to return class probabilities")


class FactCheckResponse(BaseModel):
    """Response model for fact-checking"""
    statement: str
    prediction: str
    confidence: float
    label_id: int
    probabilities: Optional[dict] = None
    timestamp: str


class BatchFactCheckRequest(BaseModel):
    """Request model for batch fact-checking"""
    statements: List[str] = Field(..., description="List of statements to verify")
    return_probabilities: bool = Field(False, description="Whether to return class probabilities")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    timestamp: str


@app.on_event("startup")
async def load_model():
    """Load model and tokenizer on startup"""
    global model, tokenizer, device
    
    try:
        print("Loading TruthGuard-AI model...")
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load model
        model_path = config.MODEL_PATH
        print(f"Model path: {model_path}")
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model.to(device)
        model.eval()
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("API will start but predictions will fail until model is loaded.")


def predict_statement(statement: str, return_probs: bool = False):
    """
    Make prediction for a single statement
    
    Args:
        statement: The statement to verify
        return_probs: Whether to return probabilities
        
    Returns:
        Dictionary with prediction results
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Tokenize input
    inputs = tokenizer(
        statement,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=config.MAX_LENGTH
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][prediction].item()
    
    result = {
        'statement': statement,
        'prediction': config.LABEL_MAP[prediction],
        'confidence': float(confidence),
        'label_id': int(prediction),
        'timestamp': datetime.now().isoformat()
    }
    
    if return_probs:
        result['probabilities'] = {
            config.LABEL_MAP[i]: float(probs[0][i].item())
            for i in range(config.NUM_LABELS)
        }
    
    return result


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "TruthGuard-AI API",
        "version": "1.0.0",
        "description": "Transformer-based Multi-Class Fact Verification System",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "labels": "/labels",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        device=str(device) if device is not None else "unknown",
        timestamp=datetime.now().isoformat()
    )


@app.get("/labels", response_model=dict)
async def get_labels():
    """Get available labels"""
    return {
        "labels": config.LABEL_MAP,
        "num_labels": config.NUM_LABELS
    }


@app.post("/predict", response_model=FactCheckResponse)
async def predict(request: FactCheckRequest):
    """
    Predict the veracity of a single statement
    
    Args:
        request: FactCheckRequest containing the statement
        
    Returns:
        FactCheckResponse with prediction results
    """
    try:
        result = predict_statement(request.statement, request.return_probabilities)
        return FactCheckResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict", response_model=List[FactCheckResponse])
async def batch_predict(request: BatchFactCheckRequest):
    """
    Predict the veracity of multiple statements
    
    Args:
        request: BatchFactCheckRequest containing list of statements
        
    Returns:
        List of FactCheckResponse with prediction results
    """
    try:
        results = []
        for statement in request.statements:
            result = predict_statement(statement, request.return_probabilities)
            results.append(FactCheckResponse(**result))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
