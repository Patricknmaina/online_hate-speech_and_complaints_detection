"""
This is the main file for the FastAPI application.
It contains the code for the API endpoints and the model loading.
"""

# Import necessary libraries
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import sys
from typing import Dict, Any, Optional, List
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep.feature_engineering import FeatureEngineering

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

app = FastAPI(
    title="Safaricom Tweet Classification API",
    description="API for classifying tweets directed towards Safaricom",
    version="1.0.0"
)

# Add CORS middleware for security and allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin
    allow_credentials=True, # Allows cookies/auth headers in requests
    allow_methods=["*"], # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers in requests (like Authorization, Content-Type)
)

# Pydantic models for request/response
class TweetRequest(BaseModel):
    text: str
    user_id: Optional[str] = None

class TweetResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    user_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    model_info: Dict[str, Any]

# Global variables for model and vectorizer
model = None
vectorizer = None
feature_engineering = None

# Load the model and vectorizer
def load_model():
    """Load the trained model and vectorizer"""
    global model, vectorizer, feature_engineering
    
    try:
        # Load the trained model
        model_path = "../models/best_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            # If no saved model exists, we'll need to train one
            print("No saved model found. Please train a model first.")
            return False
        
        # Load the vectorizer
        vectorizer_path = "../models/vectorizer.pkl"
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
        else:
            print("No saved vectorizer found. Please train a model first.")
            return False
        
        # Initialize feature engineering
        feature_engineering = FeatureEngineering(pd.DataFrame())
        
        print("Model and vectorizer loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Preprocess the text
def preprocess_text(text: str) -> str:
    """Preprocess the input text"""
    if feature_engineering is None:
        raise ValueError("Feature engineering not initialized")
    
    # Create a temporary dataframe with the text
    temp_df = pd.DataFrame({'Content': [text]})
    temp_fe = FeatureEngineering(temp_df)
    
    # Apply the same preprocessing pipeline
    temp_fe.clean_text('Content')
    temp_fe.tokenize_text('Content')
    temp_fe.lemmatize_text()
    temp_fe.create_processed_text()
    
    return temp_fe.data['processed_text'].iloc[0]

# Predict the class of a tweet
def predict_tweet(text: str) -> Dict[str, Any]:
    """Predict the class of a tweet"""
    if model is None or vectorizer is None:
        raise ValueError("Model or vectorizer not loaded")
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Vectorize the text
    text_vectorized = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    
    # Get prediction probabilities
    probabilities = model.predict_proba(text_vectorized)[0]
    confidence = max(probabilities)
    
    # Get class names
    class_names = model.classes_
    prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "probabilities": prob_dict
    }

# Initialize the model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    print("Starting up the API...")
    if not load_model():
        print("Warning: Model could not be loaded. API may not function properly.")

# Health check endpoint
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    model_loaded = model is not None and vectorizer is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        message="Safaricom Tweet Classification API is running" if model_loaded else "Model not loaded",
        model_info={
            "model_loaded": model_loaded,
            "model_type": type(model).__name__ if model else None,
            "vectorizer_type": type(vectorizer).__name__ if vectorizer else None
        }
    )

# Predict the class of a tweet endpoint
@app.post("/predict", response_model=TweetResponse)
async def predict_tweet_endpoint(request: TweetRequest):
    """Predict the class of a tweet"""
    try:
        if model is None or vectorizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Make prediction
        result = predict_tweet(request.text)
        
        return TweetResponse(
            text=request.text,
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            user_id=request.user_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Predict the class of multiple tweets endpoint
@app.post("/predict/batch")
async def predict_batch_tweets(tweets: List[TweetRequest]):
    """Predict classes for multiple tweets"""
    try:
        if model is None or vectorizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        for tweet_request in tweets:
            result = predict_tweet(tweet_request.text)
            results.append(TweetResponse(
                text=tweet_request.text,
                prediction=result["prediction"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                user_id=tweet_request.user_id
            ))
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Get information about the loaded model
@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "vectorizer_type": type(vectorizer).__name__,
        "classes": model.classes_.tolist() if hasattr(model, 'classes_') else None,
        "feature_count": len(vectorizer.vocabulary_) if vectorizer else None
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 