# **Safarimeter: Online Hate Speech and Complaint Detection for Safaricom**

## **Project Overview**

Safarimeter is an AI-powered system designed to classify tweets mentioning Safaricom, Kenya's largest telecommunications company. The platform identifies customer complaints, categorizes issues (network, MPESA, billing, etc.), and detects hate speech or abusive content.

**Key Features:**
- **OpenAI GPT-4o-mini Integration** for intelligent classification and conversational AI
- **Automated CI/CD Pipelines** for frontend, backend, and Docker deployments
- **AWS EC2 Deployment** for scalable backend hosting
- **Real-time Tweet Classification** with confidence scores and probability distributions

**Live Demo:** [Safarimer App](https://safarimeter.netlify.app)

## **Architecture**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        SAFARIMETER ARCHITECTURE                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐           │
│   │  React + TS  │      │   FastAPI    │      │  OpenAI API  │           │
│   │   Frontend   │─────▶│   Backend    │─────▶│  GPT-4o-mini │          │
│   │  (Netlify)   │      │  (AWS EC2)   │      │              │           │
│   └──────────────┘      └──────┬───────┘      └──────────────┘           │
│                                │                                         │
│                                ▼                                         │
│                       ┌──────────────┐                                   │
│                       │ Sklearn Model│                                   │
│                       │  (Fallback)  │                                   │
│                       └──────────────┘                                   │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────┐       │
│   │                   GitHub Actions CI/CD                        │      │
│   │  • Frontend: Build, Lint, TypeCheck → Deploy to Netlify      │       │
│   │  • Backend: Lint, Test, Coverage → Deploy to AWS EC2         │       │
│   │  • Docker: Build & Push to GitHub Container Registry         │       │
│   └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### **Component Breakdown**

| Component | Technology | Deployment |
|-----------|------------|------------|
| Frontend | React 20, TypeScript, Tailwind CSS, Vite | Netlify |
| Backend | FastAPI, Python 3.12, OpenAI SDK | AWS EC2 |
| Primary AI | OpenAI GPT-4o-mini | OpenAI API |
| Fallback AI | Scikit-learn (Logistic Regression) | Local model |
| Chatbot | OpenAI-powered AI Assistant | Integrated |
| CI/CD | GitHub Actions | Automated |
| Container Registry | GitHub Container Registry (ghcr.io) | Docker |

## **Tech Stack**

### **Frontend**
- **React 20** with TypeScript
- **Tailwind CSS** for styling
- **Vite** for build tooling

### **Backend**
- **FastAPI** with async support for model endpoints
- **OpenAI SDK** for GPT-4o-mini integration
- **Scikit-learn** for fallback classification
- **NLTK** for text preprocessing
- **Pydantic** for data validation
- **Pytest** for running unit tests

### **DevOps & Infrastructure**
- **GitHub Actions** for CI/CD pipelines
- **Docker** with multi-stage builds
- **AWS EC2** for backend hosting (with Supervisor for process management)
- **Netlify** for frontend hosting
- **GitHub Container Registry** for Docker images

## **AI Models**

### **Primary: OpenAI GPT-4o-mini**

The system primarily uses OpenAI's GPT-4o-mini for:
- **Tweet Classification**: Accurate categorization with reasoning
- **Confidence Scoring**: 0.0 to 1.0 confidence values
- **AI Chat Assistant**: Context-aware Safaricom customer service responses

**Configuration:**
```
.env
OPENAI_MODEL=gpt-4o-mini
USE_SKLEARN_ONLY=false  # Set to true to disable OpenAI
```

### **Fallback: Scikit-learn**

When OpenAI is unavailable, the system automatically falls back to:
- **Logistic Regression** classifier
- **TF-IDF Vectorization**
- Pre-trained on labeled Safaricom tweets

### **Classification Categories**

| Category | Description |
|----------|-------------|
| Customer care complaint | Issues with Safaricom staff or service |
| MPESA complaint | Mobile money transaction problems |
| Network reliability problem | Signal, call, or SMS issues |
| Internet or airtime bundle complaint | Data/airtime package issues |
| Data protection and privacy concern | Privacy-related concerns |
| Neutral | General comments without complaints |
| Hate Speech | Abusive or discriminatory content |

## **CI/CD Pipeline**

The project uses **GitHub Actions** for automated testing, building, and deployment.

### **1. Backend CI/CD** (`.github/workflows/backend.yml`)

**Triggers:**
```
Push/PR to main on FastAPI/**, data_prep/**, requirements.txt, pyproject.toml
```

| Job | Steps |
|-----|-------|
| **Lint** | Setup Python 3.12, Install Ruff, Run linter on FastAPI/ and data_prep/ |
| **Test** | Install dependencies, Run pytest with coverage, Upload to Codecov |
| **Deploy** | SSH to AWS EC2, Pull latest code, Update dependencies, Restart Supervisor |

### **2. Frontend CI/CD** (`.github/workflows/frontend.yml`)

**Triggers:**
```
Push/PR to main on frontend/**
```

| Job | Steps |
|-----|-------|
| **Build** | Setup Node.js 20, Install deps, ESLint, TypeScript check, Build with Vite |
| **Deploy** | Download artifacts, Deploy to Netlify (production) |

### **3. Docker CI** (`.github/workflows/docker.yml`)

**Triggers:**
```
Push to main on Dockerfile, FastAPI/**, data_prep/**, requirements.txt
```

| Job | Steps |
|-----|-------|
| **Build & Push** | Login to GHCR, Extract metadata, Setup Buildx, Build & push with caching |

**Image Tags:**
- ghcr.io/patricknmaina/online_hate-speech_and_complaints_detection:latest
- ghcr.io/patricknmaina/online_hate-speech_and_complaints_detection:<sha>

### **Required GitHub Secrets**

| Secret | Purpose |
|--------|---------|
| `OPENAI_API_KEY` | OpenAI API authentication |
| `NETLIFY_AUTH_TOKEN` | Netlify deployment |
| `NETLIFY_SITE_ID` | Netlify site identifier |
| `AWS_EC2_HOST` | AWS EC2 instance IP/hostname |
| `AWS_EC2_SSH_KEY` | SSH private key for EC2 access |
| `VITE_API_BASE_URL` | Backend API URL for frontend |

## **Deployment**

### **Backend (AWS EC2)**

The FastAPI backend is deployed on AWS EC2 with Supervisor for process management:

```
# SSH into EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Service managed by Supervisor
sudo supervisorctl status safarimeter
sudo supervisorctl restart safarimeter
```

**EC2 Setup:**
- Ubuntu instance with Python 3.12
  - Instance type: `t3.micro`
- Supervisor for process management
- Virtual environment for dependencies
- Auto-deployment via GitHub Actions

### **Frontend (Netlify)**

The React frontend is auto-deployed to Netlify on push to `main`:

- **Production URL:** [https://safarimeter.netlify.app](https://safarimeter.netlify.app)
- **Build Command:** `npm run build`
- **Publish Directory:** `frontend/dist`

### **Docker Deployment**

```
# Pull from GitHub Container Registry
docker pull ghcr.io/patricknmaina/online_hate-speech_and_complaints_detection:latest

# Run container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e USE_SKLEARN_ONLY=false \
  ghcr.io/patricknmaina/online_hate-speech_and_complaints_detection:latest

# Docker compose
docker-compose up --build
```
## **Dataset**

* **Source:** 6,146 tweets scraped using **N8N** and **TwitterAPI.io**
* **Features collected:** Tweet ID, content, likes, retweets, replies, quotes, views, timestamp
* **Labeling strategy:** Weak supervision via **OpenAI GPT-4** and **Zero-Shot Classification** using `XLM-RoBERTa` transfomer model
* **Classes defined:**
  * Customer Care Complaint
  * MPESA Complaint
  * Network Reliability Problem
  * Internet & Airtime Bundle Complaint
  * Data Protection & Privacy Concern
  * Neutral
  * Hate Speech

## **API Endpoints**

### **Health & Status**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check with system info |
| GET | `/health` | Simple health check |
| GET | `/model/info` | Model configuration details |
| GET | `/chat/status` | Chatbot availability status |

### **Prediction**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Sklearn model prediction |
| POST | `/predict/openai` | OpenAI prediction with fallback |
| POST | `/predict/batch` | Batch sklearn predictions |
| POST | `/predict/openai/batch` | Batch OpenAI predictions |

### **Chat**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | AI assistant chat endpoint |

## **Local Development**

### **Prerequisites**
- Python 3.12+
- Node.js 20+
- Docker (optional)

### **1. Clone Repository**

```
git clone https://github.com/Patricknmaina online_hate-speech_and_complaints_detection.git
cd online_hate-speech_and_complaints_detection
```

### **2. Environment Setup**
For this project, we setup the virtual environment using `uv`, an extremely fast python package manager. Installation instructions for Windows, Mac and Linux can be found [here](https://docs.astral.sh/uv/getting-started/installation/)

```
# =====Create the virtual environment=====
uv sync # this will sync the uv setup in the repository, create the virtual environment and download all the required packages

# =====Activate the virtual environment=====

# windows
source .venv/Scripts/activate

# Linux/Mac
source .venv/bin/activate

# =====Configure environment variables=====

# navigate to the FastAPI directory
cd FastAPI/

# create the .env file from .env.example
cp .env.example .env

# Edit .env with your OPENAI_API_KEY
OPENAI_API_KEY = <your-openai-key>

# =====Run the server=====
python main.py
# or
uvicorn main:app --reload --port 8000
```

**API runs at:** `http://localhost:8000`

### **3. Frontend Setup**
```
cd frontend

# Install dependencies
npm install

# Configure environment
echo "VITE_API_BASE_URL=http://localhost:8000" > .env

# Run development server
npm run dev
```
**Frontend runs at:** `http://localhost:5173`

### **4. Run Tests**
```
# Backend tests
pytest tests/ -v --cov=FastAPI

# Frontend linting
cd frontend && npm run lint
```
## **Example API Usage**

### ***Single Prediction (OpenAI)***

**Request:**
```
curl -X POST "http://localhost:8000/predict/openai" \
  -H "Content-Type: application/json" \
  -d '{"text": "Safaricom network is very slow today!"}'
```

**Response:**
```
{
  "text": "Safaricom network is very slow today!",
  "prediction": "Network reliability problem",
  "confidence": 0.92,
  "probabilities": {
    "Customer care complaint": 0.01,
    "MPESA complaint": 0.01,
    "Network reliability problem": 0.92,
    "Internet or airtime bundle complaint": 0.03,
    "Data protection and privacy concern": 0.01,
    "Neutral": 0.01,
    "Hate Speech": 0.01
  },
  "model_used": "openai"
}
```
### ***Chat Request***

**Request:**
```
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "My MPESA transaction is stuck",
    "sender_id": "user123"
  }'
```

**Response:**
```
{
  "responses": [
    {
      "text": "I'm sorry to hear about your stuck MPESA transaction. Let me help you resolve this. First, please check if you received an SMS confirmation. If the transaction is still pending after 24 hours, you can dial *234# and select 'Reverse Transaction' or contact our support line at 100."
    }
  ],
  "sender_id": "user123",
  "timestamp": "2026-01-11T10:30:00Z",
  "model_used": "openai"
}
```
## **Environment Variables**

### **Backend (.env)**
```
# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# OpenAI Configuration (Primary)
OPENAI_API_KEY=sk-your-api-key
OPENAI_MODEL=gpt-4o-mini

# Model Selection
USE_SKLEARN_ONLY=false

# Sklearn Fallback Models
SKLEARN_MODEL_PATH=models/best_model.pkl
VECTORIZER_PATH=models/vectorizer.pkl
```

### **Frontend (.env)**
```
VITE_API_BASE_URL=http://localhost:8000
```
## **Model Evaluation**

### **Classical ML Models (Baseline)**

| Model               | Accuracy | Precision | Recall   | F1-score |
|---------------------|----------|-----------|----------|----------|
|Logistic Regression  | 0.6959   | 0.7151    | 0.6959   | 0.7027   |
| Naive Bayes         | 0.6846   | 0.6770    | 0.6846   | 0.6752   |
| Random Forest       | 0.6886   | 0.6606    | 0.6886   | 0.6537   |

### **Transformer Models**

| Model           | Accuracy   | Precision  | Recall     | F1-score   |
|-----------------|------------|------------|------------|------------|
| mBERT           | 0.7131     | 0.7266     | 0.7131     | 0.7185     |
| **XLM-RoBERTa** | **0.7885** | **0.7877** | **0.7885** | **0.7866** |

### **OpenAI GPT-4o-mini**

`GPT-4o-mini` provides superior classification accuracy with contextual understanding, especially for:
- Multilingual content (English, Swahili, Sheng)
- Nuanced sentiment detection
- Context-aware hate speech identification

## **Key Visualizations**

* **Label Distribution**
![Label Distribution](images/labels_distribution.png)

* **Tweet Length Histogram**
![Tweet Length Distribution](images/tweet_length_distribution.png)

* **Word Cloud (Customer Care Terms)**
![Word Cloud](images/customer_care_wordcloud.png)

## **Data Pipeline**

### **Data Cleaning & Preprocessing**

Custom `TweetPreprocessor` class for streamlined preprocessing:

1. **Data Cleaning:** Remove URLs, mentions, hashtags, emojis, repeated characters
2. **Text Normalization:** Expand contractions, normalize punctuation
3. **Tokenization:** Lowercasing, stopword removal, lemmatization
4. **Feature Extraction:** TF-IDF Vectorization with configurable n-grams

## **Future Work**

- [ ] Real-time Twitter (X) stream integration
- [ ] Dashboard analytics with historical trends
- [ ] Webhook notifications for urgent issues
- [ ] Real-time chatbot response to specific tweet on Twitter (X)


## **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (\`git push origin feature/amazing-feature\`)
5. Open a Pull Request

## **License**

MIT License - see [LICENSE](LICENSE) file for details