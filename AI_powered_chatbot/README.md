# 🤖 Safaricom AI Chatbot (Rasa)

An intelligent conversational AI assistant for Safaricom customer support, powered by Rasa and integrated with the Safarimeter platform.

## ✨ Features

- 🗣️ **30+ Intents** covering network, MPESA, billing, and account services
- 🌍 **Multilingual** support (English + Swahili/Sheng)
- 🔄 **Intelligent Fallback** to ML models when Rasa is unavailable
- 🚀 **Easy Setup** with automated training and deployment scripts
- 🧪 **Comprehensive Testing** with automated test suite
- 📊 **Real-time Monitoring** through FastAPI integration
- 💬 **Beautiful UI** integrated with React frontend

## 🚀 Quick Start

### 1. Install Rasa

```bash
pip install rasa
```

### 2. Train the Chatbot

```bash
chmod +x train_rasa.sh
bash train_rasa.sh
```

### 3. Start the Server

```bash
chmod +x start_rasa.sh
bash start_rasa.sh
```

### 4. Test It

```bash
chmod +x test_rasa.sh
bash test_rasa.sh
```

### 5. Use via Frontend

Open the Safarimeter frontend → Click "AI Assistant" → Start chatting!

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `domain.yml` | Intents, entities, responses |
| `data/nlu.yml` | Training examples (500+ examples) |
| `data/stories.yml` | Conversation flows |
| `config.yml` | Rasa pipeline configuration |
| `train_rasa.sh` | Training automation script |
| `start_rasa.sh` | Server startup script |
| `test_rasa.sh` | Testing automation script |
| `stop_rasa.sh` | Server shutdown script |

## 🎯 Supported Use Cases

### Network Issues
- Slow network speeds
- Network outages
- No signal problems
- Call quality issues
- Data connectivity problems

### MPESA Services
- Stuck transactions
- Failed payments
- Reversal requests
- Statement generation
- PIN reset assistance

### Billing & Airtime
- Bill inquiries
- Charge disputes
- Airtime purchase
- Data bundle purchase
- Balance checks

### Customer Service
- Agent connection
- Complaint filing
- Service location
- Feedback submission

### Account Management
- SIM swap procedures
- Lost SIM reporting
- SIM unlocking
- Number changes

## 🔧 Configuration

### Rasa Settings

**Port:** 5005  
**Webhook:** `http://localhost:5005/webhooks/rest/webhook`  
**CORS:** Enabled for all origins

### Integration

The chatbot is automatically integrated with:
- ✅ FastAPI backend (middleware)
- ✅ React frontend (AI Assistant page)
- ✅ ML models (fallback system)

## 🧪 Testing Examples

Try these messages:

```
"Hello, I need help"
"My network is very slow today"
"My MPESA transaction is stuck"
"How can I buy data bundles?"
"I want to talk to customer service"
"Where is the nearest Safaricom shop?"
```

## 📊 API Endpoints

### Direct Rasa Endpoint

```bash
POST http://localhost:5005/webhooks/rest/webhook
Content-Type: application/json

{
  "sender": "user_id",
  "message": "your message"
}
```

### Via FastAPI (Recommended)

```bash
POST http://localhost:8000/chat
Content-Type: application/json

{
  "message": "your message",
  "sender_id": "user_id"
}
```

## 🛠️ Maintenance

### Update Training Data

1. Edit `data/nlu.yml` (add examples)
2. Edit `data/stories.yml` (add conversation flows)
3. Edit `domain.yml` (add responses)
4. Run `bash train_rasa.sh`
5. Run `bash stop_rasa.sh && bash start_rasa.sh`

### View Logs

```bash
tail -f /tmp/rasa.log
```

### Stop Server

```bash
bash stop_rasa.sh
# or
pkill -f "rasa run"
```

## 📚 Documentation

For detailed integration guide, see: `RASA_INTEGRATION_GUIDE.md`

## 🤝 Contributing

To add new intents or improve responses:

1. Add training examples to `data/nlu.yml`
2. Add conversation flows to `data/stories.yml`
3. Add responses to `domain.yml`
4. Retrain the model
5. Test thoroughly

## 📝 License

Part of the Safarimeter project.

## 🎉 Credits

Built with ❤️ using Rasa Open Source

