#!/bin/bash

"""
User Data script for AWS EC2 instance to set up the Safarimeter backend.

This script performs the following tasks:
1. Updates the system and installs necessary packages (Python 3.12, Git, Supervisor, Nginx).
2. Clones the Safarimeter repository from GitHub.
3. Sets up a Python virtual environment and installs dependencies.
4. Configures Supervisor to run the FastAPI application on startup.
5. Configures Nginx as a reverse proxy to forward requests to the FastAPI application.
6. Logs all output to /var/log/user_data.log for debugging purposes.
"""

set -euo pipefail

LOG_FILE="/var/log/user_data.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== Safarimeter EC2 bootstrap started at $(date) ==="

# --- System updates ---
apt-get update && apt-get upgrade -y

# --- Install Python 3.12 via deadsnakes PPA ---
apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update
apt-get install -y \
  python3.12 \
  python3.12-venv \
  python3.12-dev \
  python3-pip

# --- Install system dependencies ---
apt-get install -y \
  git \
  build-essential \
  libffi-dev \
  supervisor \
  nginx \
  curl

# --- Clone repository ---
cd /home/ubuntu
git clone ${github_repo_url} online_hate-speech_and_complaints_detection
cd online_hate-speech_and_complaints_detection

# --- Python virtual environment ---
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# --- Create FastAPI .env ---
cat > FastAPI/.env << 'ENVEOF'
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
OPENAI_API_KEY=${openai_api_key}
OPENAI_MODEL=gpt-4o-mini
USE_SKLEARN_ONLY=false
SKLEARN_MODEL_PATH=models/best_model.pkl
VECTORIZER_PATH=models/vectorizer.pkl
ENVEOF

# --- Configure Supervisor ---
cat > /etc/supervisor/conf.d/safarimeter.conf << 'SUPEOF'
[program:safarimeter]
directory=/home/ubuntu/online_hate-speech_and_complaints_detection/FastAPI
command=/home/ubuntu/online_hate-speech_and_complaints_detection/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
user=ubuntu
autostart=true
autorestart=true
stderr_logfile=/var/log/safarimeter.err.log
stdout_logfile=/var/log/safarimeter.out.log
environment=HOME="/home/ubuntu",USER="ubuntu"
SUPEOF

supervisorctl reread
supervisorctl update
supervisorctl start safarimeter

# --- Configure Nginx reverse proxy ---
INSTANCE_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 || echo "_")

cat > /etc/nginx/sites-available/safarimeter << NGXEOF
server {
    listen 80;
    server_name $INSTANCE_IP;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
NGXEOF

ln -sf /etc/nginx/sites-available/safarimeter /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx
systemctl enable nginx

# --- Fix ownership ---
chown -R ubuntu:ubuntu /home/ubuntu/online_hate-speech_and_complaints_detection

echo "=== Safarimeter EC2 bootstrap completed at $(date) ==="
