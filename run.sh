#!/bin/bash

# ENV_NAME="project-env"

# # Activate conda
# echo "Activating conda environment '$ENV_NAME'..."
# eval "$(conda shell.bash hook)"
# conda activate $ENV_NAME

# # Start FastAPI
# echo "Starting FastAPI server..."
# cd FastAPI/
# uvicorn main:app --host 0.0.0.0 --port 8000 &
# FASTAPI_PID=$!
# cd ..

# # Start Rasa actions
# echo "Starting Rasa action server..."
# cd Streamlit/AI_powered_chatbot/
# rasa run actions &
# RASA_ACTION_PID=$!
# cd ../..

# # Start Streamlit
# echo "Starting Streamlit app..."
# cd Streamlit/
# streamlit run app.py --server.port 8501 &
# STREAMLIT_PID=$!
# cd ..

# sleep 5

# # Start ngrok tunnels in background
# echo "Starting ngrok tunnels..."
# ngrok http 8000 --log=stdout > /dev/null &
# NGROK_FASTAPI_PID=$!
# ngrok http 8501 --log=stdout > /dev/null &
# NGROK_STREAMLIT_PID=$!

# # Give ngrok some time to register tunnels
# sleep 7

# # Fetch public URLs from ngrok's API
# FASTAPI_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | grep -o "https://[a-z0-9.-]*\.ngrok-free\.app" | head -n 1)
# STREAMLIT_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | grep -o "https://[a-z0-9.-]*\.ngrok-free\.app" | tail -n 1)

# echo "✅ FastAPI public URL: $FASTAPI_URL"
# echo "✅ Streamlit public URL: $STREAMLIT_URL"

# # Keep script alive
# echo "All services started. Press [CTRL+C] to stop."
# trap "echo 'Stopping services...'; kill $FASTAPI_PID $RASA_ACTION_PID $STREAMLIT_PID $NGROK_FASTAPI_PID $NGROK_STREAMLIT_PID; exit 0" SIGINT SIGTERM
# wait

ENV_NAME="project-env"

# Path to your ngrok config file
NGROK_CONFIG="$HOME/.config/ngrok/ngrok.yml"

# Activate conda
echo "Activating conda environment '$ENV_NAME'..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Start FastAPI
echo "Starting FastAPI server..."
cd FastAPI/
uvicorn main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!
cd ..

# Start Rasa actions
echo "Starting Rasa action server..."
cd Streamlit/AI_powered_chatbot/
rasa run actions &
RASA_ACTION_PID=$!
cd ../..

# Start Streamlit
echo "Starting Streamlit app..."
cd Streamlit/
streamlit run app.py --server.port 8501 &
STREAMLIT_PID=$!
cd ..

sleep 5

# Start ngrok with both tunnels from config file
echo "Starting ngrok tunnels..."
ngrok start --all --config "$NGROK_CONFIG" > /dev/null &
NGROK_PID=$!

# Wait for ngrok to start
sleep 7

# Get URLs for each named tunnel
FASTAPI_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | jq -r '.tunnels[] | select(.name=="fastapi") | .public_url')
STREAMLIT_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | jq -r '.tunnels[] | select(.name=="streamlit") | .public_url')

echo "✅ FastAPI public URL: $FASTAPI_URL"
echo "✅ Streamlit public URL: $STREAMLIT_URL"

# Keep script alive
echo "All services started. Press [CTRL+C] to stop."
trap "echo 'Stopping services...'; kill $FASTAPI_PID $RASA_ACTION_PID $STREAMLIT_PID $NGROK_PID; exit 0" SIGINT SIGTERM
wait
