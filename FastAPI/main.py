"""
FastAPI app for Safaricom Tweet Classification
Lightweight version using OpenAI GPT-4o-mini for classification and chat
Falls back to sklearn model when OpenAI is unavailable
"""

# -------------------------- Imports --------------------------
import asyncio
import gc
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

import joblib
import nltk
import pandas as pd
import psutil
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables
load_dotenv()


# -------------------------- Configuration --------------------------
class Config:
    """Configuration class for environment-based settings"""

    # OpenAI settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Model selection
    USE_SKLEARN_ONLY = os.getenv("USE_SKLEARN_ONLY", "false").lower() == "true"

    # Model paths
    SKLEARN_MODEL_PATH = os.getenv("SKLEARN_MODEL_PATH", "models/best_model.pkl")
    VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "models/vectorizer.pkl")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Classification categories
    CATEGORIES = [
        "Customer care complaint",
        "Data protection and privacy concern",
        "Hate Speech",
        "Internet or airtime bundle complaint",
        "MPESA complaint",
        "Network reliability problem",
        "Neutral"
    ]

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
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.warning(f"NLTK download error: {e}")

setup_nltk()

# -------------------------- OpenAI Client --------------------------
openai_client = None
if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "your-openai-api-key-here":
    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    logger.info("OpenAI client initialized")
else:
    logger.warning("OpenAI API key not configured - will use sklearn fallback")

# -------------------------- Model Manager --------------------------
class ModelManager:
    """Centralized model management with lazy loading"""

    def __init__(self):
        self._sklearn_model = None
        self._sklearn_vectorizer = None
        self._feature_engineering = None
        self._model_loaded_sklearn = False
        self._executor = ThreadPoolExecutor(max_workers=10)

        logger.info("ModelManager initialized")
        logger.info(f"OpenAI available: {openai_client is not None}")
        logger.info(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.1f}GB")

    @lru_cache(maxsize=1)
    def get_sklearn_model(self):
        """Lazy load scikit-learn model with caching"""
        if self._sklearn_model is None:
            try:
                logger.info("Loading scikit-learn model...")
                if os.path.exists(config.SKLEARN_MODEL_PATH):
                    self._sklearn_model = joblib.load(config.SKLEARN_MODEL_PATH)
                    self._model_loaded_sklearn = True
                    logger.info("Scikit-learn model loaded successfully")
                else:
                    logger.error(f"Model file not found: {config.SKLEARN_MODEL_PATH}")
                    raise FileNotFoundError(f"Model file not found: {config.SKLEARN_MODEL_PATH}")
            except Exception as e:
                logger.error(f"Error loading sklearn model: {e}")
                raise
        return self._sklearn_model

    @lru_cache(maxsize=1)
    def get_sklearn_vectorizer(self):
        """Lazy load scikit-learn vectorizer with caching"""
        if self._sklearn_vectorizer is None:
            try:
                logger.info("Loading vectorizer...")
                if os.path.exists(config.VECTORIZER_PATH):
                    self._sklearn_vectorizer = joblib.load(config.VECTORIZER_PATH)
                    logger.info("Vectorizer loaded successfully")
                else:
                    logger.error(f"Vectorizer file not found: {config.VECTORIZER_PATH}")
                    raise FileNotFoundError(f"Vectorizer file not found: {config.VECTORIZER_PATH}")
            except Exception as e:
                logger.error(f"Error loading vectorizer: {e}")
                raise
        return self._sklearn_vectorizer

    def get_feature_engineering(self):
        """Lazy load feature engineering"""
        if self._feature_engineering is None:
            self._feature_engineering = FeatureEngineering(pd.DataFrame())
        return self._feature_engineering

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model status information"""
        return {
            "sklearn_loaded": self._model_loaded_sklearn,
            "openai_available": openai_client is not None,
            "openai_model": config.OPENAI_MODEL if openai_client else None,
            "sklearn_model_type": type(self._sklearn_model).__name__ if self._sklearn_model else None,
            "memory_usage": f"{psutil.virtual_memory().percent}%",
            "available_memory_mb": int(psutil.virtual_memory().available / (1024 * 1024))
        }

# Initialize model manager
model_manager = ModelManager()

# -------------------------- Async Context Manager --------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting up API...")
    yield
    logger.info("Shutting down API...")
    gc.collect()
    logger.info("Cleanup completed")

# -------------------------- FastAPI App --------------------------
app = FastAPI(
    title="Safaricom Tweet Classification API",
    description="Lightweight API using OpenAI GPT-4o-mini for tweet classification and chat",
    version="4.0.0",
    lifespan=lifespan
)

origins = [
    "https://safarimeter-v2.netlify.app",
    "https://safarimeter.netlify.app",
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
    model_used: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    sender_id: Optional[str] = "default"
    conversation_history: Optional[List[Dict[str, str]]] = None

class ChatMessage(BaseModel):
    text: str
    image: Optional[str] = None
    buttons: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    responses: List[ChatMessage]
    sender_id: str
    timestamp: str
    model_used: Optional[str] = None

class ChatStatus(BaseModel):
    openai_available: bool
    model: str
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
        temp_df = pd.DataFrame({'Content': [text]})
        temp_fe = FeatureEngineering(temp_df)
        temp_fe.clean_text('Content')
        temp_fe.tokenize_text('Content')
        temp_fe.lemmatize_text()
        temp_fe.create_processed_text()
        return temp_fe.data['processed_text'].iloc[0]
    except Exception as e:
        logger.error(f"Text preprocessing error: {e}")
        return text.lower().strip()

# -------------------------- Predict with Sklearn --------------------------
def predict_with_sklearn(text: str) -> Dict[str, Any]:
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
            "probabilities": prob_dict,
            "model_used": "sklearn"
        }
    except Exception as e:
        logger.error(f"Sklearn prediction error: {e}")
        raise ValueError(f"Sklearn prediction failed: {str(e)}")

# -------------------------- Predict with OpenAI --------------------------
def predict_with_openai(text: str) -> Dict[str, Any]:
    """Predict using OpenAI GPT-4o-mini"""
    if openai_client is None:
        raise ValueError("OpenAI client not configured")

    try:
        categories_str = "\n".join([f"- {cat}" for cat in config.CATEGORIES])

        system_prompt = f"""You are a tweet classifier for Safaricom (a Kenyan telecommunications company).
Classify the given tweet into exactly ONE of these categories:

{categories_str}

Respond in JSON format with:
- "prediction": the exact category name from the list above
- "confidence": a number between 0.0 and 1.0 indicating your confidence
- "reasoning": a brief explanation (1 sentence)

Important context:
- MPESA is Safaricom's mobile money service
- Tweets may be in English, Swahili, or Sheng (Kenyan slang)
- "Customer care complaint" is about service quality from staff
- "Network reliability problem" is about calls, SMS, or signal issues
- "Internet or airtime bundle complaint" is about data packages or airtime
- "Hate Speech" includes insults, threats, or discriminatory language
- "Neutral" is for general comments without complaints or issues"""

        response = openai_client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify this tweet:\n\n\"{text}\""}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=200
        )

        import json
        result = json.loads(response.choices[0].message.content)

        prediction = result.get("prediction", "Neutral")
        confidence = float(result.get("confidence", 0.8))

        # Validate prediction is in our categories
        if prediction not in config.CATEGORIES:
            # Find closest match
            prediction_lower = prediction.lower()
            for cat in config.CATEGORIES:
                if cat.lower() in prediction_lower or prediction_lower in cat.lower():
                    prediction = cat
                    break
            else:
                prediction = "Neutral"

        # Create probability distribution (OpenAI gives us confidence for the prediction)
        prob_dict = {cat: 0.0 for cat in config.CATEGORIES}
        prob_dict[prediction] = confidence
        remaining = 1.0 - confidence
        other_cats = [c for c in config.CATEGORIES if c != prediction]
        for cat in other_cats:
            prob_dict[cat] = remaining / len(other_cats)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "model_used": "openai"
        }

    except Exception as e:
        logger.error(f"OpenAI prediction error: {e}")
        raise ValueError(f"OpenAI prediction failed: {str(e)}")

# -------------------------- Combined Prediction --------------------------
def predict_tweet(text: str, prefer_openai: bool = True) -> Dict[str, Any]:
    """Predict with OpenAI first, fallback to sklearn"""
    if prefer_openai and openai_client is not None and not config.USE_SKLEARN_ONLY:
        try:
            return predict_with_openai(text)
        except Exception as e:
            logger.warning(f"OpenAI failed, falling back to sklearn: {e}")

    return predict_with_sklearn(text)

# -------------------------- Chat with OpenAI --------------------------
def chat_with_openai(message: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """Chat using OpenAI GPT-4o-mini with Safaricom customer service persona"""
    if openai_client is None:
        raise ValueError("OpenAI client not configured")

    system_prompt = """You are a helpful and friendly AI customer service assistant for Safaricom, Kenya's leading telecommunications company.

Your role:
- Help customers with questions about Safaricom services (MPESA, data bundles, network, etc.)
- Provide accurate information about Safaricom products and services
- Handle complaints professionally and empathetically
- Escalate complex issues by suggesting customers contact official support channels

Key information:
- MPESA support: Dial *234# or call the MPESA support line
- Customer care: Call 100 (from Safaricom) or 0722000000
- Data bundles: Check offers at *544# or mySafaricom app
- Network issues: Report via mySafaricom app or call customer care
- Privacy concerns: Email dpo@safaricom.co.ke

Guidelines:
- Be polite, professional, and empathetic
- Use simple language, avoid jargon
- If you don't know something, say so and suggest official channels
- Never share false information about prices or promotions
- Respond in the same language the customer uses (English, Swahili, or Sheng)"""

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history if provided
    if conversation_history:
        for msg in conversation_history[-10:]:  # Keep last 10 messages for context
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

    messages.append({"role": "user", "content": message})

    try:
        response = openai_client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"OpenAI chat error: {e}")
        raise ValueError(f"OpenAI chat failed: {str(e)}")

# -------------------------- Fallback Response Generator --------------------------
def generate_fallback_response(prediction: str, confidence: float, message: str) -> str:
    """Generate intelligent fallback responses based on tweet classification"""

    responses = {
        "MPESA complaint": "I understand you're experiencing issues with MPESA. For immediate assistance, please dial *234# or contact our MPESA support line. Your concern is important to us.",

        "Customer care complaint": "Thank you for reaching out. I've noted your concern regarding customer service. For urgent matters, please call 100 from your Safaricom line or 0722000000.",

        "Network reliability problem": "I see you're experiencing network issues. Our technical team is working to improve service. You can check network status at safaricom.co.ke/coverage or report via the mySafaricom app.",

        "Data protection and privacy concern": "We take data protection seriously at Safaricom. For privacy concerns, please email dpo@safaricom.co.ke.",

        "Internet or airtime bundle complaint": "I understand your concern about bundles. Check current offers by dialing *544# or through the mySafaricom app.",

        "Neutral": "Thank you for your message! How can I assist you with Safaricom services today?",

        "Hate Speech": "I'm here to help with Safaricom services. Please let me know how I can assist you constructively."
    }

    base_response = responses.get(prediction, responses["Neutral"])

    if confidence < 0.6:
        base_response = f"I'm here to help! {base_response}"

    return base_response

# -------------------------- Endpoints --------------------------
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint with detailed information"""
    model_info = model_manager.get_model_info()

    system_info = {
        "memory_usage_percent": psutil.virtual_memory().percent,
        "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
        "timestamp": datetime.utcnow().isoformat()
    }

    status = "healthy" if (model_info["sklearn_loaded"] or model_info["openai_available"]) else "unhealthy"

    return HealthResponse(
        status=status,
        message="API is running with OpenAI GPT-4o-mini and sklearn fallback",
        model_info=model_info,
        system_info=system_info
    )

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    try:
        model_info = model_manager.get_model_info()
        is_healthy = model_info["sklearn_loaded"] or model_info["openai_available"]

        if is_healthy:
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        else:
            return {"status": "unhealthy", "message": "No models available"}
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get detailed model and configuration information"""
    return ModelInfoResponse(
        model_info=model_manager.get_model_info(),
        config={
            "openai_model": config.OPENAI_MODEL,
            "use_sklearn_only": config.USE_SKLEARN_ONLY,
            "categories": config.CATEGORIES
        }
    )

@app.post("/predict", response_model=TweetResponse)
async def predict_endpoint(request: TweetRequest):
    """Predict using sklearn model"""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            model_manager._executor,
            predict_with_sklearn,
            request.text
        )

        return TweetResponse(
            text=request.text,
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            user_id=request.user_id,
            model_used="sklearn"
        )
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/openai", response_model=TweetResponse)
async def predict_openai_endpoint(request: TweetRequest):
    """Predict using OpenAI GPT-4o-mini with sklearn fallback"""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            model_manager._executor,
            predict_tweet,
            request.text,
            True  # prefer_openai
        )

        return TweetResponse(
            text=request.text,
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            user_id=request.user_id,
            model_used=result.get("model_used", "openai")
        )
    except Exception as e:
        logger.error(f"OpenAI prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch_endpoint(tweets: List[TweetRequest]):
    """Batch predict using sklearn model - concurrent processing"""
    try:
        loop = asyncio.get_event_loop()

        async def process_single_tweet(tweet: TweetRequest) -> TweetResponse:
            result = await loop.run_in_executor(
                model_manager._executor,
                predict_with_sklearn,
                tweet.text
            )
            return TweetResponse(
                text=tweet.text,
                prediction=result["prediction"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                user_id=tweet.user_id,
                model_used="sklearn"
            )

        # Process all tweets concurrently
        results = await asyncio.gather(*[process_single_tweet(t) for t in tweets])

        return {"predictions": list(results)}
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict/openai/batch")
async def predict_openai_batch_endpoint(tweets: List[TweetRequest]):
    """Batch predict using OpenAI with sklearn fallback - concurrent processing"""
    try:
        loop = asyncio.get_event_loop()

        async def process_single_tweet(tweet: TweetRequest) -> TweetResponse:
            result = await loop.run_in_executor(
                model_manager._executor,
                predict_tweet,
                tweet.text,
                True
            )
            return TweetResponse(
                text=tweet.text,
                prediction=result["prediction"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                user_id=tweet.user_id,
                model_used=result.get("model_used", "openai")
            )

        # Process all tweets concurrently
        results = await asyncio.gather(*[process_single_tweet(t) for t in tweets])

        return {"predictions": list(results)}
    except Exception as e:
        logger.error(f"OpenAI batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# -------------------------- Chat Endpoints --------------------------
@app.get("/chat/status", response_model=ChatStatus)
async def get_chat_status():
    """Get chatbot status"""
    return ChatStatus(
        openai_available=openai_client is not None,
        model=config.OPENAI_MODEL if openai_client else "sklearn-fallback",
        fallback_mode=openai_client is None,
        status="operational"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint using OpenAI with intelligent fallback"""
    try:
        # Try OpenAI first
        if openai_client is not None:
            try:
                response_text = await asyncio.get_event_loop().run_in_executor(
                    model_manager._executor,
                    chat_with_openai,
                    request.message,
                    request.conversation_history
                )

                return ChatResponse(
                    responses=[ChatMessage(text=response_text)],
                    sender_id=request.sender_id,
                    timestamp=datetime.utcnow().isoformat(),
                    model_used="openai"
                )
            except Exception as e:
                logger.warning(f"OpenAI chat failed, using fallback: {e}")

        # Fallback: Use ML classification to generate response
        try:
            prediction_result = predict_with_sklearn(request.message)
        except Exception as e:
            logger.error(f"Sklearn prediction failed: {e}")
            prediction_result = {"prediction": "Neutral", "confidence": 0.5}

        prediction = prediction_result["prediction"]
        confidence = prediction_result["confidence"]
        response_text = generate_fallback_response(prediction, confidence, request.message)

        return ChatResponse(
            responses=[ChatMessage(text=response_text)],
            sender_id=request.sender_id,
            timestamp=datetime.utcnow().isoformat(),
            model_used="sklearn-fallback"
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return ChatResponse(
            responses=[ChatMessage(
                text="I'm experiencing technical difficulties. Please try again or contact Safaricom customer care directly at 100."
            )],
            sender_id=request.sender_id,
            timestamp=datetime.utcnow().isoformat(),
            model_used="error-fallback"
        )

# -------------------------- Metrics Endpoint --------------------------
@app.get("/metrics")
async def get_metrics():
    """Get application metrics for monitoring"""
    try:
        model_info = model_manager.get_model_info()
        memory = psutil.virtual_memory()

        return {
            "memory_usage_bytes": memory.used,
            "memory_available_bytes": memory.available,
            "memory_percent": memory.percent,
            "cpu_percent": psutil.cpu_percent(),
            "sklearn_loaded": 1 if model_info["sklearn_loaded"] else 0,
            "openai_available": 1 if model_info["openai_available"] else 0,
            "timestamp": datetime.utcnow().timestamp()
        }
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")

# -------------------------- Main --------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting FastAPI server on {host}:{port}")
    logger.info(f"OpenAI Model: {config.OPENAI_MODEL}")
    logger.info(f"OpenAI Available: {openai_client is not None}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )
