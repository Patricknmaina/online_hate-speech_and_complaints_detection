"""
FastAPI app for Safaricom Tweet Classification
Supports both Scikit-learn (local) and XLM-RoBERTa (Hugging Face Hub) models
"""

# -------------------------- Imports --------------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
import joblib
import os
import sys
import zipfile

# For NLP preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Hugging Face
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn import functional as F

# For chatbot integration
import requests
import asyncio
from datetime import datetime

# -------------------------- Project Setup --------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prep.feature_engineering import FeatureEngineering

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# -------------------------- FastAPI App --------------------------
app = FastAPI(
    title="Safaricom Tweet Classification API",
    description="API for classifying tweets directed towards Safaricom",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------- Pydantic Models --------------------------
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

class ChatRequest(BaseModel):
    message: str
    sender_id: Optional[str] = "web_user"

class ChatResponse(BaseModel):
    responses: List[Dict[str, Any]]
    sender_id: str
    timestamp: str

# -------------------------- Globals --------------------------
model = None
vectorizer = None
feature_engineering = None

transformer_model = None
transformer_tokenizer = None
transformer_classes = None

# -------------------------- Load Scikit-learn Model --------------------------
def load_model():
    global model, vectorizer, feature_engineering
    try:
        model_path = "../models/best_model.pkl"
        vectorizer_path = "../models/vectorizer.pkl"

        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            print("No saved model found.")
            return False

        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
        else:
            print("No saved vectorizer found.")
            return False

        feature_engineering = FeatureEngineering(pd.DataFrame())
        print("✅ Scikit-learn model and vectorizer loaded.")
        return True
    except Exception as e:
        print(f"❌ Error loading sklearn model: {e}")
        return False

# -------------------------- Load Transformer Model from Hugging Face --------------------------
def load_transformer_model(
    repo_id: str = "patrickmaina/safaricom-hatespeech-detector",
    use_auth_token: Optional[str] = None
):
    global transformer_model, transformer_tokenizer, transformer_classes
    try:
        # Load tokenizer from Hugging Face Hub
        transformer_tokenizer = AutoTokenizer.from_pretrained(repo_id, token=use_auth_token)

        # Define label mapping (must match training!)
        label2id = {
            "Customer care complaint": 0,
            "Data protection and privacy concern": 1,
            "Hate Speech": 2,
            "Internet or airtime bundle complaint": 3,
            "MPESA complaint": 4,
            "Network reliability problem": 5,
            "Neutral": 6
        }
        id2label = {v: k for k, v in label2id.items()}

        # Load model from Hugging Face Hub
        transformer_model = AutoModelForSequenceClassification.from_pretrained(
            repo_id,
            id2label=id2label,
            label2id=label2id,
            token=use_auth_token
        )
        transformer_model.eval()

        transformer_classes = id2label

        print(f"✅ Transformer model loaded from Hugging Face Hub: {repo_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to load transformer model from Hugging Face: {e}")
        return False

# -------------------------- Preprocessing (Sklearn) --------------------------
def preprocess_text(text: str) -> str:
    if feature_engineering is None:
        raise ValueError("Feature engineering not initialized")
    temp_df = pd.DataFrame({'Content': [text]})
    temp_fe = FeatureEngineering(temp_df)
    temp_fe.clean_text('Content')
    temp_fe.tokenize_text('Content')
    temp_fe.lemmatize_text()
    temp_fe.create_processed_text()
    return temp_fe.data['processed_text'].iloc[0]

# -------------------------- Predict with Sklearn --------------------------
def predict_tweet(text: str) -> Dict[str, Any]:
    if model is None or vectorizer is None:
        raise ValueError("Model or vectorizer not loaded")
    processed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    confidence = max(probabilities)
    class_names = model.classes_
    prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "probabilities": prob_dict
    }

# -------------------------- Predict with Transformer --------------------------
def predict_with_transformer(text: str) -> Dict[str, Any]:
    if transformer_model is None or transformer_tokenizer is None:
        raise ValueError("Transformer model or tokenizer not loaded")
    inputs = transformer_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = transformer_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze().tolist()
    pred_idx = int(torch.argmax(logits, dim=1).item())
    prediction = transformer_classes.get(pred_idx, f"class_{pred_idx}")
    prob_dict = {transformer_classes[i]: float(p) for i, p in enumerate(probs)}
    return {
        "prediction": prediction,
        "confidence": float(max(probs)),
        "probabilities": prob_dict
    }

# -------------------------- FastAPI Startup --------------------------
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting API...")

    # Load scikit-learn model locally
    load_model()

    # Load transformer model from Hugging Face Hub
    hf_token = os.getenv("HF_TOKEN", None)  # Use env var if repo is private
    load_transformer_model(
        repo_id="patrickmaina/safaricom-hatespeech-detector",
        use_auth_token=hf_token
    )

# -------------------------- Endpoints --------------------------
@app.get("/", response_model=HealthResponse)
async def root():
    model_loaded = model is not None and vectorizer is not None
    transformer_loaded = transformer_model is not None and transformer_tokenizer is not None
    return HealthResponse(
        status="healthy" if model_loaded or transformer_loaded else "unhealthy",
        message="API is up and running",
        model_info={
            "sklearn_loaded": model_loaded,
            "transformer_loaded": transformer_loaded,
            "sklearn_model_type": type(model).__name__ if model else None,
            "transformer_model_type": type(transformer_model).__name__ if transformer_model else None
        }
    )

@app.post("/predict", response_model=TweetResponse)
async def predict_tweet_endpoint(request: TweetRequest):
    try:
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

@app.post("/predict/transformer", response_model=TweetResponse)
async def predict_transformer_endpoint(request: TweetRequest):
    try:
        result = predict_with_transformer(request.text)
        return TweetResponse(
            text=request.text,
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            user_id=request.user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transformer prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch_tweets(tweets: List[TweetRequest]):
    try:
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

@app.post("/predict/transformer/batch")
async def predict_batch_transformer(tweets: List[TweetRequest]):
    try:
        texts = [t.text for t in tweets]
        inputs = transformer_tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = transformer_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)

        results = []
        for i, tweet in enumerate(tweets):
            pred_idx = int(torch.argmax(probs[i]).item())
            prediction = transformer_classes.get(pred_idx, f"class_{pred_idx}")
            prob_dict = {transformer_classes[j]: float(p) for j, p in enumerate(probs[i])}
            results.append(TweetResponse(
                text=tweet.text,
                prediction=prediction,
                confidence=float(torch.max(probs[i])),
                probabilities=prob_dict,
                user_id=tweet.user_id
            ))
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transformer batch prediction error: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    return {
        "sklearn_model_type": type(model).__name__ if model else None,
        "transformer_model_type": type(transformer_model).__name__ if transformer_model else None,
        "transformer_classes": transformer_classes if transformer_classes else None
    }

# -------------------------- Chatbot Integration --------------------------
RASA_URL = "http://localhost:5005"

async def check_rasa_status():
    """Check if Rasa server is available"""
    try:
        response = requests.get(f"{RASA_URL}/status", timeout=5)
        return response.status_code == 200
    except:
        return False

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """
    Send a message to the Rasa chatbot and get the response
    """
    try:
        # Check if Rasa is available
        if not await check_rasa_status():
            # Fallback to tweet prediction if Rasa is not available
            try:
                # Use the existing tweet prediction function (synchronous call)
                tweet_result = predict_tweet(request.message)
                
                # Extract prediction and confidence
                prediction = tweet_result["prediction"]
                confidence = tweet_result["confidence"]
                
                fallback_responses = {
                    'MPESA complaint': f'I understand you\'re having an MPESA issue (confidence: {confidence:.1%}). Let me help you with that. Our MPESA team is available 24/7 to assist you. Could you provide more details about your specific MPESA problem?',
                    'Customer care complaint': f'Thank you for reaching out. I can see you need customer care assistance (confidence: {confidence:.1%}). How can I help you today? Please describe your concern in detail.',
                    'Network reliability problem': f'I notice you\'re experiencing network issues (confidence: {confidence:.1%}). Our technical team is working to resolve network problems in your area. Are you experiencing slow internet, call drops, or no signal?',
                    'Data protection and privacy concern': f'Your privacy concerns are important to us (confidence: {confidence:.1%}). Safaricom takes data protection seriously. Can you tell me more about your specific privacy concern?',
                    'Internet or airtime bundle complaint': f'I see you have concerns about our bundles (confidence: {confidence:.1%}). Let me help you find the best solution for your data needs. What specific issue are you experiencing with your bundles?',
                    'Neutral': 'Thank you for contacting Safaricom! How can I assist you today? Feel free to ask me about our services, report any issues, or get help with your account.',
                    'Hate Speech': f'I understand you\'re frustrated with our services (confidence: {confidence:.1%}). Let me help address your concerns and improve your experience. What specific issue can I help you resolve today?'
                }
                
                response_text = fallback_responses.get(prediction, f'I received your message "{request.message}". How can I help you with Safaricom services today?')
                
                return ChatResponse(
                    responses=[{"text": response_text}],
                    sender_id=request.sender_id,
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                print(f"Fallback prediction error: {str(e)}")
                # More specific fallback based on message content
                message_lower = request.message.lower()
                if any(word in message_lower for word in ['mpesa', 'money', 'transaction', 'payment']):
                    response_text = "I can help you with MPESA issues. What specific problem are you experiencing with your transaction?"
                elif any(word in message_lower for word in ['network', 'slow', 'connection', 'internet', 'data']):
                    response_text = "I can help you with network issues. Are you experiencing slow internet, call problems, or connectivity issues?"
                elif any(word in message_lower for word in ['customer', 'care', 'support', 'help', 'complaint']):
                    response_text = "I'm here to help with your customer service needs. What can I assist you with today?"
                elif any(word in message_lower for word in ['bundle', 'airtime', 'credit']):
                    response_text = "I can help you with airtime and data bundles. What would you like to know about our packages?"
                else:
                    response_text = f"Hello! I'm your Safaricom AI assistant. I see you said: '{request.message}'. How can I help you with Safaricom services today?"
                
                return ChatResponse(
                    responses=[{"text": response_text}],
                    sender_id=request.sender_id,
                    timestamp=datetime.now().isoformat()
                )
        
        # Send message to Rasa
        payload = {
            "sender": request.sender_id,
            "message": request.message
        }
        
        response = requests.post(
            f"{RASA_URL}/webhooks/rest/webhook",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            rasa_responses = response.json()
            if not rasa_responses:
                # If Rasa returns empty response, provide a fallback
                rasa_responses = [{"text": "I'm not sure how to respond to that. Can you please rephrase your question?"}]
            
            return ChatResponse(
                responses=rasa_responses,
                sender_id=request.sender_id,
                timestamp=datetime.now().isoformat()
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to communicate with chatbot")
            
    except Exception as e:
        # Fallback response in case of any error
        return ChatResponse(
            responses=[{"text": "I'm sorry, I'm having trouble processing your message right now. Please try again later."}],
            sender_id=request.sender_id,
            timestamp=datetime.now().isoformat()
        )

@app.get("/chat/status")
async def get_chat_status():
    """
    Check the status of the chatbot service
    """
    rasa_available = await check_rasa_status()
    return {
        "rasa_available": rasa_available,
        "rasa_url": RASA_URL,
        "fallback_mode": not rasa_available,
        "status": "operational"
    }

# -------------------------- Main --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
