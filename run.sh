#!/bin/bash

# Starts the FastAPI server, Rasa chatbot, and Streamlit app using conda environment

# setup the conda environment
ENV_NAME="project-env"

# Properly initialize Conda and activate environment
echo "Activating conda environment '$ENV_NAME'..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Start FastAPI endpoint
echo "Starting FastAPI server..."
cd FastAPI/
uvicorn main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!
cd ..

# Start the Rasa action server
echo "Starting Rasa action server..."
cd Streamlit/AI_powered_chatbot/
rasa run actions &
RASA_ACTION_PID=$!
cd ../..

# Start the Streamlit app
echo "Starting Streamlit app..."
cd Streamlit/
# streamlit run app.py &
streamlit run app_final.py &
STREAMLIT_PID=$!
cd ..

# Wait to keep the script running
echo "All services started. Press [CTRL+C] to stop."
wait $FASTAPI_PID $RASA_ACTION_PID $STREAMLIT_PID

# Cleanup on exit
trap "echo 'Stopping services...'; kill $FASTAPI_PID $RASA_ACTION_PID $STREAMLIT_PID; exit 0" SIGINT SIGTERM