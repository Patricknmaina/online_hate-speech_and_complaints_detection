#!/bin/bash
cd /home/patndirangu/git-repositories/online_hate-speech_and_complaints_detection/FastAPI
export HF_TOKEN="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"
export HF_MODEL_REPO="patrickmaina/safaricom-hatespeech-detector"
export SKLEARN_MODEL_PATH="models/best_model.pkl"
export VECTORIZER_PATH="models/vectorizer.pkl"
/home/patndirangu/miniconda3/envs/safaricom-env/bin/python main.py

