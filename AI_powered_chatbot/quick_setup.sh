#!/bin/bash

# Quick Setup Script for Safaricom Rasa Chatbot
# This script handles everything: installation, training, and startup

echo "========================================"
echo "  Safaricom Chatbot - Quick Setup"
echo "========================================"
echo ""

cd "$(dirname "$0")"

# Step 1: Check/Install Rasa
echo "📦 Step 1: Checking Rasa installation..."
if ! command -v rasa &> /dev/null; then
    echo "   Rasa not found. Installing..."
    pip install rasa
    if [ $? -eq 0 ]; then
        echo "   ✅ Rasa installed successfully!"
    else
        echo "   ❌ Failed to install Rasa. Please install manually:"
        echo "      pip install rasa"
        exit 1
    fi
else
    echo "   ✅ Rasa is already installed"
    rasa --version
fi
echo ""

# Step 2: Train the model
echo "🎓 Step 2: Training the chatbot..."
if [ ! -f "models/safaricom-chatbot.tar.gz" ]; then
    echo "   No trained model found. Training now..."
    bash train_rasa.sh
    if [ $? -ne 0 ]; then
        echo "   ❌ Training failed. Please check the errors above."
        exit 1
    fi
else
    read -p "   Model already exists. Retrain? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        bash train_rasa.sh
        if [ $? -ne 0 ]; then
            echo "   ❌ Training failed. Please check the errors above."
            exit 1
        fi
    else
        echo "   Using existing model."
    fi
fi
echo ""

# Step 3: Start the server
echo "🚀 Step 3: Starting Rasa server..."
bash start_rasa.sh
if [ $? -ne 0 ]; then
    echo "   ❌ Failed to start server."
    exit 1
fi
echo ""

# Step 4: Run tests
echo "🧪 Step 4: Running tests..."
read -p "   Run automated tests? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    sleep 5  # Give server time to fully initialize
    bash test_rasa.sh
fi
echo ""

# Success!
echo "========================================"
echo "  ✅ Setup Complete!"
echo "========================================"
echo ""
echo "📊 Next Steps:"
echo ""
echo "1. 🌐 Open frontend: http://localhost:5176"
echo "2. 💬 Click 'AI Assistant' in navigation"
echo "3. 🎯 Start chatting!"
echo ""
echo "📝 Useful Commands:"
echo "  - View logs: tail -f /tmp/rasa.log"
echo "  - Stop server: bash stop_rasa.sh"
echo "  - Test chatbot: bash test_rasa.sh"
echo ""
echo "📚 Documentation: RASA_INTEGRATION_GUIDE.md"
echo ""

