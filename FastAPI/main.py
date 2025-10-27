"""
FastAPI app for Safaricom Tweet Classification
Optimized with lazy loading and environment-based model selection
Supports both Scikit-learn (local) and transformer models (e.g., mBERT)
Falls back to Hugging Face Inference API if transformer model cannot be loaded due to memory limits
"""

# -------------------------- Imports --------------------------
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import gc
from functools import lru_cache
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import joblib
import os
import sys
import zipfile
import requests
import psutil
from datetime import datetime

# For NLP preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Hugging Face
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn import functional as F

# -------------------------- Configuration --------------------------
class Config:
    """Configuration class for environment-based settings"""
    
    # Environment variables with defaults
    USE_LIGHTWEIGHT_MODEL = os.getenv("USE_LIGHTWEIGHT_MODEL", "false").lower() == "true"
    MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "2048"))
    HF_TOKEN = os.getenv("HF_TOKEN")
    MODEL_CACHE_SIZE = int(os.getenv("MODEL_CACHE_SIZE", "1"))
    ENABLE_MODEL_QUANTIZATION = os.getenv("ENABLE_MODEL_QUANTIZATION", "true").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Model paths
    SKLEARN_MODEL_PATH = os.getenv("SKLEARN_MODEL_PATH", "models/best_model.pkl")
    VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "models/vectorizer.pkl")
    
    # Hugging Face settings
    # Default to mBERT; override with your fine-tuned mBERT repo via env
    HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "patrickmaina/safaricom-hatespeech-detector")
    HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_REPO}"
    
    @classmethod
    def should_use_hf_inference(cls) -> bool:
        """Determine if we should use HF Inference API based on memory constraints"""
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        return (
            cls.USE_LIGHTWEIGHT_MODEL or 
            cls.MAX_MEMORY_MB < 1024 or 
            available_memory < 800
        )
    
    @classmethod
    def get_torch_device(cls) -> str:
        """Get the appropriate torch device"""
        if torch.cuda.is_available() and cls.MAX_MEMORY_MB > 2048:
            return "cuda"
        else:
            return "cpu"

config = Config()

# -------------------------- Logging Setup --------------------------
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------- Project Setup --------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data_prep.feature_engineering import FeatureEngineering
except ImportError:
    logger.warning("FeatureEngineering not found. Creating placeholder.")
    class FeatureEngineering:
        def __init__(self, data):
            self.data = data

# -------------------------- NLTK Setup --------------------------
def setup_nltk():
    """Download NLTK data with error handling"""
    try:
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        logger.info("✅ NLTK data downloaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ NLTK download error: {e}")

# Download NLTK data
setup_nltk()

# -------------------------- Model Manager --------------------------
class ModelManager:
    """Centralized model management with lazy loading and caching"""
    
    def __init__(self):
        self._sklearn_model = None
        self._sklearn_vectorizer = None
        self._transformer_model = None
        self._transformer_tokenizer = None
        self._transformer_classes = None
        self._feature_engineering = None
        self._model_loaded_sklearn = False
        self._model_loaded_transformer = False
        self._use_hf_inference = config.should_use_hf_inference()
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"🔧 ModelManager initialized")
        logger.info(f"📊 Max Memory: {config.MAX_MEMORY_MB}MB")
        logger.info(f"🤖 Use HF Inference: {self._use_hf_inference}")
        logger.info(f"💾 Available Memory: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    
    @lru_cache(maxsize=config.MODEL_CACHE_SIZE)
    def get_sklearn_model(self):
        """Lazy load scikit-learn model with caching"""
        if self._sklearn_model is None:
            try:
                logger.info("🔄 Loading scikit-learn model...")
                if os.path.exists(config.SKLEARN_MODEL_PATH):
                    self._sklearn_model = joblib.load(config.SKLEARN_MODEL_PATH)
                    self._model_loaded_sklearn = True
                    logger.info("✅ Scikit-learn model loaded successfully")
                else:
                    logger.error(f"❌ Model file not found: {config.SKLEARN_MODEL_PATH}")
                    raise FileNotFoundError(f"Model file not found: {config.SKLEARN_MODEL_PATH}")
            except Exception as e:
                logger.error(f"❌ Error loading sklearn model: {e}")
                raise
        return self._sklearn_model
    
    @lru_cache(maxsize=config.MODEL_CACHE_SIZE)
    def get_sklearn_vectorizer(self):
        """Lazy load scikit-learn vectorizer with caching"""
        if self._sklearn_vectorizer is None:
            try:
                logger.info("🔄 Loading vectorizer...")
                if os.path.exists(config.VECTORIZER_PATH):
                    self._sklearn_vectorizer = joblib.load(config.VECTORIZER_PATH)
                    logger.info("✅ Vectorizer loaded successfully")
                else:
                    logger.error(f"❌ Vectorizer file not found: {config.VECTORIZER_PATH}")
                    raise FileNotFoundError(f"Vectorizer file not found: {config.VECTORIZER_PATH}")
            except Exception as e:
                logger.error(f"❌ Error loading vectorizer: {e}")
                raise
        return self._sklearn_vectorizer
    
    def get_feature_engineering(self):
        """Lazy load feature engineering"""
        if self._feature_engineering is None:
            self._feature_engineering = FeatureEngineering(pd.DataFrame())
        return self._feature_engineering
    
    @lru_cache(maxsize=config.MODEL_CACHE_SIZE)
    def get_transformer_tokenizer(self):
        """Lazy load transformer tokenizer with caching"""
        if self._transformer_tokenizer is None and not self._use_hf_inference:
            try:
                logger.info("🔄 Loading transformer tokenizer...")
                self._transformer_tokenizer = AutoTokenizer.from_pretrained(
                    config.HF_MODEL_REPO, 
                    token=config.HF_TOKEN
                )
                logger.info("✅ Transformer tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"❌ Error loading transformer tokenizer: {e}")
                self._use_hf_inference = True
                logger.info("🔄 Falling back to HF Inference API")
        return self._transformer_tokenizer
    
    @lru_cache(maxsize=config.MODEL_CACHE_SIZE)
    def get_transformer_model(self):
        """Lazy load transformer model with memory optimization"""
        if self._transformer_model is None and not self._use_hf_inference:
            try:
                logger.info("🔄 Loading transformer model...")
                
                # Define label mappings
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
                self._transformer_classes = id2label
                
                # Load model with optimizations
                model_kwargs = {
                    "id2label": id2label,
                    "label2id": label2id,
                    "token": config.HF_TOKEN
                }
                
                # Add memory optimizations if enabled
                if config.ENABLE_MODEL_QUANTIZATION:
                    model_kwargs["torch_dtype"] = torch.float16
                    if torch.cuda.is_available():
                        model_kwargs["device_map"] = "auto"
                
                self._transformer_model = AutoModelForSequenceClassification.from_pretrained(
                    config.HF_MODEL_REPO,
                    **model_kwargs
                )
                
                # Set to evaluation mode
                self._transformer_model.eval()
                
                # Move to appropriate device
                device = config.get_torch_device()
                if device == "cuda" and torch.cuda.is_available():
                    self._transformer_model = self._transformer_model.to(device)
                
                self._model_loaded_transformer = True
                logger.info(f"✅ Transformer model loaded successfully on {device}")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("⚠️ Out of memory: falling back to HF Inference API")
                    self._use_hf_inference = True
                    self.clear_transformer_cache()
                else:
                    logger.error(f"❌ Failed to load transformer model: {e}")
                    raise
            except Exception as e:
                logger.error(f"❌ Error loading transformer model: {e}")
                self._use_hf_inference = True
        
        return self._transformer_model
    
    def get_transformer_classes(self):
        """Get transformer model classes"""
        if self._transformer_classes is None and not self._use_hf_inference:
            # Try to load the model to get classes
            self.get_transformer_model()
        return self._transformer_classes
    
    def is_using_hf_inference(self) -> bool:
        """Check if using HF Inference API"""
        return self._use_hf_inference
    
    def clear_sklearn_cache(self):
        """Clear scikit-learn model cache and free memory"""
        logger.info("🧹 Clearing sklearn model cache...")
        self._sklearn_model = None
        self._sklearn_vectorizer = None
        self._model_loaded_sklearn = False
        self.get_sklearn_model.cache_clear()
        self.get_sklearn_vectorizer.cache_clear()
        gc.collect()
    
    def clear_transformer_cache(self):
        """Clear transformer model cache and free memory"""
        logger.info("🧹 Clearing transformer model cache...")
        if self._transformer_model is not None:
            del self._transformer_model
        if self._transformer_tokenizer is not None:
            del self._transformer_tokenizer
        
        self._transformer_model = None
        self._transformer_tokenizer = None
        self._model_loaded_transformer = False
        
        self.get_transformer_model.cache_clear()
        self.get_transformer_tokenizer.cache_clear()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
    
    def clear_all_cache(self):
        """Clear all model caches"""
        logger.info("🧹 Clearing all model caches...")
        self.clear_sklearn_cache()
        self.clear_transformer_cache()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model status information"""
        return {
            "sklearn_loaded": self._model_loaded_sklearn,
            "transformer_loaded": self._model_loaded_transformer,
            "use_hf_inference": self._use_hf_inference,
            "sklearn_model_type": type(self._sklearn_model).__name__ if self._sklearn_model else None,
            "transformer_model_type": "HF_Inference_API" if self._use_hf_inference else type(self._transformer_model).__name__ if self._transformer_model else None,
            "memory_usage": f"{psutil.virtual_memory().percent}%",
            "available_memory_mb": int(psutil.virtual_memory().available / (1024 * 1024))
        }

# Initialize model manager
model_manager = ModelManager()

# -------------------------- Async Context Manager --------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting up API...")
    
    # Optionally warm up models on startup
    if not config.USE_LIGHTWEIGHT_MODEL:
        try:
            # Warm up in background
            asyncio.create_task(warm_up_models())
        except Exception as e:
            logger.warning(f"⚠️ Model warmup failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down API...")
    model_manager.clear_all_cache()
    logger.info("✅ Cleanup completed")

async def warm_up_models():
    """Warm up models in background"""
    try:
        logger.info("🔥 Warming up models...")
        
        # Warm up sklearn model first
        try:
            await asyncio.get_event_loop().run_in_executor(
                model_manager._executor,
                predict_tweet,
                "Test warmup tweet for sklearn model"
            )
            logger.info("✅ Sklearn model warmed up successfully")
        except Exception as e:
            logger.warning(f"⚠️ Sklearn model warmup failed: {e}")
        
        # Warm up transformer model with a simple prediction
        if not model_manager.is_using_hf_inference():
            await asyncio.get_event_loop().run_in_executor(
                model_manager._executor,
                predict_with_transformer,
                "Test warmup tweet"
            )
        
        logger.info("✅ Models warmed up successfully")
    except Exception as e:
        logger.warning(f"⚠️ Model warmup failed: {e}")

# -------------------------- FastAPI App --------------------------
app = FastAPI(
    title="Safaricom Tweet Classification API",
    description="Optimized API for classifying tweets directed towards Safaricom with lazy loading and environment-based model selection",
    version="3.0.0",
    lifespan=lifespan
)

origins = [
    "https://safarimeter-v2.netlify.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:5176",
    "http://localhost:5177",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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

class ChatRequest(BaseModel):
    message: str
    sender_id: Optional[str] = "default"

class ChatMessage(BaseModel):
    text: str
    image: Optional[str] = None
    buttons: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    responses: List[ChatMessage]
    sender_id: str
    timestamp: str

class ChatStatus(BaseModel):
    rasa_available: bool
    rasa_url: str
    fallback_mode: bool
    status: str

class HealthResponse(BaseModel):
    status: str
    message: str
    model_info: Dict[str, Any]
    system_info: Dict[str, Any]

class ModelInfoResponse(BaseModel):
    model_info: Dict[str, Any]
    config: Dict[str, Any]

# -------------------------- Preprocessing (Sklearn) --------------------------
def preprocess_text(text: str) -> str:
    """Preprocess text for sklearn model"""
    try:
        feature_engineering = model_manager.get_feature_engineering()
        temp_df = pd.DataFrame({'Content': [text]})
        temp_fe = FeatureEngineering(temp_df)
        temp_fe.clean_text('Content')
        temp_fe.tokenize_text('Content')
        temp_fe.lemmatize_text()
        temp_fe.create_processed_text()
        return temp_fe.data['processed_text'].iloc[0]
    except Exception as e:
        logger.error(f"❌ Text preprocessing error: {e}")
        # Fallback to simple preprocessing
        return text.lower().strip()

# -------------------------- Predict with Sklearn --------------------------
def predict_tweet(text: str) -> Dict[str, Any]:
    """Predict using scikit-learn model"""
    try:
        model = model_manager.get_sklearn_model()
        vectorizer = model_manager.get_sklearn_vectorizer()
        
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
    except Exception as e:
        logger.error(f"❌ Sklearn prediction error: {e}")
        raise ValueError(f"Sklearn prediction failed: {str(e)}")

# -------------------------- Predict with Transformer --------------------------
def predict_with_transformer(text: str) -> Dict[str, Any]:
    """Predict using transformer model or HF Inference API"""
    try:
        # Use HF Inference API if configured
        if model_manager.is_using_hf_inference():
            return _predict_with_hf_api(text)
        
        # Local inference
        model = model_manager.get_transformer_model()
        tokenizer = model_manager.get_transformer_tokenizer()
        classes = model_manager.get_transformer_classes()
        
        if model is None or tokenizer is None:
            logger.warning("⚠️ Model/tokenizer not available, falling back to HF API")
            return _predict_with_hf_api(text)
        
        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze()
            
            # Handle single dimension case
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
        
        pred_idx = int(torch.argmax(logits, dim=1).item())
        prediction = classes.get(pred_idx, f"class_{pred_idx}")
        
        # Convert probabilities to dict
        prob_dict = {classes[i]: float(probs[i]) for i in range(len(probs))}
        
        return {
            "prediction": prediction,
            "confidence": float(torch.max(probs).item()),
            "probabilities": prob_dict
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("⚠️ GPU OOM, falling back to HF Inference API")
            model_manager._use_hf_inference = True
            model_manager.clear_transformer_cache()
            return _predict_with_hf_api(text)
        else:
            logger.error(f"❌ Transformer prediction error: {e}")
            raise ValueError(f"Transformer prediction failed: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Transformer prediction error: {e}")
        raise ValueError(f"Transformer prediction failed: {str(e)}")

def _predict_with_hf_api(text: str) -> Dict[str, Any]:
    """Predict using Hugging Face Inference API"""
    try:
        headers = {"Authorization": f"Bearer {config.HF_TOKEN}"} if config.HF_TOKEN else {}
        payload = {"inputs": text}
        
        response = requests.post(config.HF_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    result = result[0]  # Unwrap nested list
                
                prediction = max(result, key=lambda x: x["score"])["label"]
                confidence = max(result, key=lambda x: x["score"])["score"]
                prob_dict = {entry["label"]: entry["score"] for entry in result}
                
                return {
                    "prediction": prediction,
                    "confidence": confidence,
                    "probabilities": prob_dict
                }
            else:
                raise ValueError(f"Unexpected API response format: {result}")
        
        elif response.status_code == 503:
            raise ValueError("Model is currently loading, please try again in a few moments")
        else:
            raise ValueError(f"API error {response.status_code}: {response.text}")
            
    except requests.exceptions.Timeout:
        raise ValueError("Request timeout - the model may be starting up")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ HF API prediction error: {e}")
        raise ValueError(f"HF Inference API failed: {str(e)}")

# -------------------------- Batch Predictions --------------------------
def predict_batch_with_transformer(texts: List[str]) -> List[Dict[str, Any]]:
    """Batch predict with transformer model"""
    try:
        if model_manager.is_using_hf_inference():
            # HF API doesn't support true batch processing, so we process individually
            results = []
            for text in texts:
                result = predict_with_transformer(text)
                result["text"] = text
                results.append(result)
            return results
        
        # Local batch inference
        model = model_manager.get_transformer_model()
        tokenizer = model_manager.get_transformer_tokenizer()
        classes = model_manager.get_transformer_classes()
        
        if model is None or tokenizer is None:
            logger.warning("⚠️ Model/tokenizer not available for batch processing")
            return [predict_with_transformer(text) for text in texts]
        
        # Batch tokenization
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
        
        results = []
        for i, text in enumerate(texts):
            pred_idx = int(torch.argmax(probs[i]).item())
            prediction = classes.get(pred_idx, f"class_{pred_idx}")
            prob_dict = {classes[j]: float(probs[i][j]) for j in range(len(classes))}
            
            results.append({
                "prediction": prediction,
                "confidence": float(torch.max(probs[i]).item()),
                "probabilities": prob_dict,
                "text": text
            })
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {e}")
        # Fallback to individual predictions
        return [predict_with_transformer(text) for text in texts]

# -------------------------- Endpoints --------------------------
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint with detailed information"""
    model_info = model_manager.get_model_info()
    
    system_info = {
        "memory_usage_percent": psutil.virtual_memory().percent,
        "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    status = "healthy" if (model_info["sklearn_loaded"] or model_info["transformer_loaded"] or model_info["use_hf_inference"]) else "unhealthy"
    
    return HealthResponse(
        status=status,
        message="API is up and running with optimized model loading",
        model_info=model_info,
        system_info=system_info
    )

@app.get("/health")
async def health_check():
    """Simple health check endpoint for Railway platform"""
    try:
        model_info = model_manager.get_model_info()
        is_healthy = (model_info["sklearn_loaded"] or 
                     model_info["transformer_loaded"] or 
                     model_info["use_hf_inference"])
        
        if is_healthy:
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        else:
            return {"status": "unhealthy", "message": "No models loaded"}
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get detailed model and configuration information"""
    return ModelInfoResponse(
        model_info=model_manager.get_model_info(),
        config={
            "use_lightweight_model": config.USE_LIGHTWEIGHT_MODEL,
            "max_memory_mb": config.MAX_MEMORY_MB,
            "model_cache_size": config.MODEL_CACHE_SIZE,
            "enable_quantization": config.ENABLE_MODEL_QUANTIZATION,
            "hf_model_repo": config.HF_MODEL_REPO,
            "torch_device": config.get_torch_device()
        }
    )

@app.post("/predict", response_model=TweetResponse)
async def predict_tweet_endpoint(request: TweetRequest):
    """Predict using scikit-learn model"""
    try:
        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            model_manager._executor,
            predict_tweet,
            request.text
        )
        
        return TweetResponse(
            text=request.text,
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            user_id=request.user_id
        )
    except Exception as e:
        logger.error(f"❌ Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/transformer", response_model=TweetResponse)
async def predict_transformer_endpoint(request: TweetRequest):
    """Predict using transformer model"""
    try:
        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            model_manager._executor,
            predict_with_transformer,
            request.text
        )
        
        return TweetResponse(
            text=request.text,
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            user_id=request.user_id
        )
    except Exception as e:
        logger.error(f"❌ Transformer prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Transformer prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch_tweets(tweets: List[TweetRequest]):
    """Batch predict using scikit-learn model"""
    try:
        results = []
        
        # Process in batches to avoid overwhelming the system
        batch_size = 10
        for i in range(0, len(tweets), batch_size):
            batch = tweets[i:i + batch_size]
            
            # Run batch in thread pool
            loop = asyncio.get_event_loop()
            batch_futures = [
                loop.run_in_executor(
                    model_manager._executor,
                    predict_tweet,
                    tweet.text
                )
                for tweet in batch
            ]
            
            batch_results = await asyncio.gather(*batch_futures)
            
            for tweet_request, result in zip(batch, batch_results):
                results.append(TweetResponse(
                    text=tweet_request.text,
                    prediction=result["prediction"],
                    confidence=result["confidence"],
                    probabilities=result["probabilities"],
                    user_id=tweet_request.user_id
                ))
        
        return {"predictions": results}
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict/transformer/batch")
async def predict_batch_transformer(tweets: List[TweetRequest]):
    """Batch predict using transformer model"""
    try:
        texts = [t.text for t in tweets]
        
        # Run batch prediction in thread pool
        loop = asyncio.get_event_loop()
        batch_results = await loop.run_in_executor(
            model_manager._executor,
            predict_batch_with_transformer,
            texts
        )
        
        responses = []
        for tweet, result in zip(tweets, batch_results):
            responses.append(TweetResponse(
                text=tweet.text,
                prediction=result["prediction"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                user_id=tweet.user_id
            ))
        
        return {"predictions": responses}
    except Exception as e:
        logger.error(f"❌ Transformer batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Transformer batch prediction error: {str(e)}")

@app.post("/model/warm")
async def warm_up_endpoint(background_tasks: BackgroundTasks):
    """Warm up models endpoint"""
    try:
        background_tasks.add_task(warm_up_models)
        return {"status": "warming up", "message": "Models are being warmed up in the background"}
    except Exception as e:
        logger.error(f"❌ Warmup error: {e}")
        raise HTTPException(status_code=500, detail=f"Warmup error: {str(e)}")

@app.post("/model/clear-cache")
async def clear_cache_endpoint():
    """Clear model cache endpoint"""
    try:
        model_manager.clear_all_cache()
        return {"status": "success", "message": "Model cache cleared successfully"}
    except Exception as e:
        logger.error(f"❌ Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear error: {str(e)}")

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system metrics"""
    try:
        # Test transformer prediction if available
        test_prediction_success = False
        test_error = None
        
        try:
            if not config.USE_LIGHTWEIGHT_MODEL:
                test_result = predict_with_transformer("Health check test")
                test_prediction_success = test_result.get("prediction") is not None
        except Exception as e:
            test_error = str(e)
        
        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        model_info = model_manager.get_model_info()
        
        health_data = {
            "status": "healthy" if (model_info["sklearn_loaded"] or model_info["transformer_loaded"] or model_info["use_hf_inference"]) else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "models": model_info,
            "system": {
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "cpu_usage_percent": cpu_percent,
                "disk_usage_percent": psutil.disk_usage('/').percent
            },
            "test_prediction": {
                "success": test_prediction_success,
                "error": test_error
            },
            "configuration": {
                "lightweight_mode": config.USE_LIGHTWEIGHT_MODEL,
                "max_memory_mb": config.MAX_MEMORY_MB,
                "using_hf_inference": model_manager.is_using_hf_inference(),
                "torch_device": config.get_torch_device()
            }
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/metrics")
async def get_metrics():
    """Get application metrics for monitoring"""
    try:
        model_info = model_manager.get_model_info()
        memory = psutil.virtual_memory()
        
        metrics = {
            "memory_usage_bytes": memory.used,
            "memory_available_bytes": memory.available,
            "memory_percent": memory.percent,
            "cpu_percent": psutil.cpu_percent(),
            "models_loaded_sklearn": 1 if model_info["sklearn_loaded"] else 0,
            "models_loaded_transformer": 1 if model_info["transformer_loaded"] else 0,
            "using_hf_inference": 1 if model_info["use_hf_inference"] else 0,
            "timestamp": datetime.utcnow().timestamp()
        }
        
        if torch.cuda.is_available():
            metrics["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved()
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")

# -------------------------- Chat Endpoints --------------------------
@app.get("/chat/status", response_model=ChatStatus)
async def get_chat_status():
    """Get chatbot status - indicates if Rasa is available or using fallback"""
    rasa_url = os.getenv("RASA_URL", "http://localhost:5005")
    rasa_available = False
    
    # Try to check if Rasa is available
    try:
        response = requests.get(f"{rasa_url}/", timeout=2)
        rasa_available = response.status_code == 200
    except Exception as e:
        logger.debug(f"Rasa not available: {e}")
        rasa_available = False
    
    return ChatStatus(
        rasa_available=rasa_available,
        rasa_url=rasa_url,
        fallback_mode=not rasa_available,
        status="operational"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint with intelligent fallback
    - Attempts to use Rasa if available
    - Falls back to ML-based classification responses if Rasa is unavailable
    """
    try:
        rasa_url = os.getenv("RASA_URL", "http://localhost:5005")
        
        # Try Rasa first
        try:
            rasa_response = requests.post(
                f"{rasa_url}/webhooks/rest/webhook",
                json={
                    "sender": request.sender_id,
                    "message": request.message
                },
                timeout=5
            )
            
            if rasa_response.status_code == 200:
                rasa_messages = rasa_response.json()
                
                if rasa_messages:
                    chat_messages = []
                    for msg in rasa_messages:
                        chat_messages.append(ChatMessage(
                            text=msg.get("text", ""),
                            image=msg.get("image"),
                            buttons=msg.get("buttons")
                        ))
                    
                    return ChatResponse(
                        responses=chat_messages,
                        sender_id=request.sender_id,
                        timestamp=datetime.utcnow().isoformat()
                    )
        except Exception as e:
            logger.debug(f"Rasa unavailable, using fallback: {e}")
        
        # Fallback: Use ML classification to generate intelligent response
        # Try transformer first, fall back to sklearn if needed
        try:
            prediction_result = predict_with_transformer(request.message)
        except Exception as e:
            logger.debug(f"Transformer unavailable, using sklearn: {e}")
            prediction_result = predict_tweet(request.message)
        
        prediction = prediction_result["prediction"]
        confidence = prediction_result["confidence"]
        
        # Generate contextual response based on classification
        response_text = generate_fallback_response(prediction, confidence, request.message)
        
        return ChatResponse(
            responses=[ChatMessage(text=response_text)],
            sender_id=request.sender_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}")
        # Return a generic helpful message on error
        return ChatResponse(
            responses=[ChatMessage(
                text="I'm experiencing some technical difficulties. Please try again in a moment. For urgent issues, please contact Safaricom customer care directly."
            )],
            sender_id=request.sender_id,
            timestamp=datetime.utcnow().isoformat()
        )

def generate_fallback_response(prediction: str, confidence: float, message: str) -> str:
    """Generate intelligent fallback responses based on tweet classification"""
    
    responses = {
        "MPESA complaint": "I understand you're experiencing issues with MPESA. Our technical team is aware of transaction processing concerns. For immediate assistance, please call *234# or contact our MPESA support line. Your transaction is being reviewed.",
        
        "Customer care complaint": "Thank you for reaching out to Safaricom. I've noted your concern regarding customer service. A customer care representative will be with you shortly. For urgent matters, please call 100 from your Safaricom line or 0722000000.",
        
        "Network reliability problem": "I see you're experiencing network connectivity issues. Our technical team is working to improve service in affected areas. You can check network status updates at safaricom.co.ke/coverage or report specific issues via the mySafaricom app.",
        
        "Data protection and privacy concern": "We take data protection and privacy very seriously at Safaricom. Your concern has been noted and will be reviewed by our data protection team. For immediate privacy concerns, please email dpo@safaricom.co.ke.",
        
        "Internet or airtime bundle complaint": "I understand your concern about internet and airtime bundles. We're continuously working to improve our data offerings and coverage. You can check current bundle offers by dialing *544# or through the mySafaricom app.",
        
        "Neutral": "Thank you for your message! How can I assist you with Safaricom services today? I can help with MPESA, data bundles, network issues, or connect you with customer care.",
        
        "Hate Speech": "I'm here to help with Safaricom services. If you have concerns or complaints, I'd be happy to assist you in a constructive way. Please let me know how I can help improve your experience with us."
    }
    
    base_response = responses.get(prediction, responses["Neutral"])
    
    # Add confidence-based qualifier for uncertain predictions
    if confidence < 0.6:
        base_response = f"I'm here to help! {base_response}"
    
    return base_response

# -------------------------- Main --------------------------
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Configure logging for uvicorn
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": config.LOG_LEVEL,
            "handlers": ["default"],
        },
    }
    
    logger.info(f"🚀 Starting FastAPI server on {host}:{port}")
    logger.info(f"📊 Configuration: Lightweight={config.USE_LIGHTWEIGHT_MODEL}, MaxMem={config.MAX_MEMORY_MB}MB")
    
    uvicorn.run(
        "main:app", 
        host=host, 
        port=port,
        log_config=log_config,
        access_log=True
    )
