"""
This is the Streamlit app for the Safaricom Tweet Classifier,
modified to include a multi-layered UI with different pages.
"""

# Import necessary libraries
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import sys
import os
import html # Import the html module for escaping content
import re   # Import re for regular expressions

import zipfile
import requests

# --- Google Drive Model Downloader with Large File Support ---
# FILE_ID = "1CtiNyjbbYdO7pHdDPthaxRrCbnQ2jaoh"  # Replace with your Google Drive file ID
# URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# MODEL_ZIP_PATH = "Streamlit/models.zip"
# MODEL_DIR = "Streamlit/models"

# def download_file_from_google_drive(file_id, dest_path):
#     def get_confirm_token(response):
#         for key, value in response.cookies.items():
#             if key.startswith("download_warning"):
#                 return value
#         return None

#     def save_response_content(response, destination):
#         with open(destination, "wb") as f:
#             for chunk in response.iter_content(32768):
#                 if chunk:
#                     f.write(chunk)

#     session = requests.Session()
#     response = session.get(URL, params={'id': file_id}, stream=True)
#     token = get_confirm_token(response)

#     if token:
#         params = {'id': file_id, 'confirm': token}
#         response = session.get(URL, params=params, stream=True)

#     save_response_content(response, dest_path)

# def download_and_extract_model():
#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR, exist_ok=True)

#     if not os.path.exists(MODEL_ZIP_PATH):
#         print("📥 Downloading model from Google Drive...")
#         try:
#             download_file_from_google_drive(FILE_ID, MODEL_ZIP_PATH)
#             print("✅ Download complete. Extracting...")
#             with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
#                 zip_ref.extractall(MODEL_DIR)
#             print("✅ Model extracted successfully.")
#         except Exception as e:
#             print(f"❌ Failed to download model: {e}")

# # Download model before running the app
# download_and_extract_model()

# Add the parent directory to Python path to find AI_powered_chatbot
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from AI_powered_chatbot.rasa_client import RasaClient
    RASA_AVAILABLE = True
    RASA_UI_URL = "http://localhost:5055"  # Change this to your Rasa UI URL
except ImportError as e:
    print(f"Warning: Could not import RasaClient: {e}")
    RASA_AVAILABLE = False
    RASA_UI_URL = None

# Page configuration
st.set_page_config(
    page_title="Safarimeter: The Pulse of Public Opinion",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# --- Utility function to strip HTML tags ---
def strip_html_tags(text):
    """Remove all HTML tags and entities from a string."""
    if not isinstance(text, str):
        return str(text)
    
    # First decode HTML entities
    text = html.unescape(text)
    
    # Then remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove any remaining HTML entities
    text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Dynamic CSS based on theme ---
def get_css(theme):
    if theme == 'dark':
        return """
        <style>
            /* Dark Theme */
            .stApp {
                background-color: #0e1117;
                color: #f0f2f6;
            }
            .stMarkdown, .stText, .stWrite, p, label, .stDataFrame, .stTextInput > div > div > input, .stButton > button, .stDownloadButton > button, .stFileUploader > label, .stSelectbox > label {
                color: #f0f2f6 !important;
            }
            h1, h2, h3, h4, h5, h6, .st-bh, .st-bi {
                color: #64b5f6 !important;
                font-weight: bold;
            }
            .stSidebar {
                background-color: #1e2025;
            }
            .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
                color: #64b5f6 !important;
            }
            .stSidebar .stMarkdown p, .stSidebar .stText p, .stSidebar .stWrite p {
                color: #f0f2f6 !important;
                font-size: 14px;
                line-height: 1.4;
            }
            .info-box {
                background-color: #2e3035;
                border-left: 5px solid #64b5f6;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
                font-size: 14px;
                color: #f0f2f6;
            }
            .info-box .info-label {
                font-weight: bold;
                color: #64b5f6;
            }
            .status-box {
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
                font-weight: bold;
            }
            .status-connected { background-color: #1a4d2e; color: #81c784; }
            .status-disconnected { background-color: #6d1c24; color: #e57373; }
            .status-info { background-color: #1a4e5c; color: #81d4fa; }
            .status-warning { background-color: #6f5d2b; color: #ffd54f; }
            .main-header {
                font-size: 3rem;
                font-weight: bold;
                color: #64b5f6;
                text-align: center;
                margin-bottom: 2rem;
            }
            .sub-header {
                font-size: 1.5rem;
                color: #b0bec5;
                text-align: center;
                margin-bottom: 2rem;
            }
            .prediction-box {
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
            }
            .positive { background-color: #1a4d2e; border: 1px solid #81c784; color: #81c784; }
            .negative { background-color: #6d1c24; border: 1px solid #e57373; color: #e57373; }
            .neutral { background-color: #1a4e5c; border: 1px solid #81d4fa; color: #81d4fa; }
            .complaint { background-color: #6f5d2b; border: 1px solid #ffd54f; color: #ffd54f; }
            .hate-speech { background-color: #6d1c24; border: 1px solid #e57373; color: #e57373; }
            .stButton > button {
                width: 100%;
                background-color: #1f77b4;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 5px;
                font-size: 1.1rem;
                transition: background-color 0.2s;
            }
            .stButton > button:hover {
                background-color: #1565c0;
            }
            .chat-message {
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 10px;
                max-width: 80%;
                word-wrap: break-word;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .user-message {
                background-color: #424242;
                border: 1px solid #64b5f6;
                margin-left: auto;
                text-align: right;
                color: #f0f2f6;
            }
            .bot-message {
                background-color: #212121;
                border: 1px solid #bdbdbd;
                margin-right: auto;
                color: #f0f2f6;
                text-align: left;
            }
            .chat-container {
                height: 400px;
                overflow-y: auto;
                border: 2px solid #333;
                border-radius: 10px;
                padding: 1rem;
                background-color: #212121;
                margin-bottom: 1rem;
            }
            .stTextInput > div > div > input {
                background-color: #333;
                color: #f0f2f6;
                border: 2px solid #555;
                border-radius: 8px;
                padding: 0.5rem;
            }
            .stTextInput > div > div > input:focus {
                border-color: #64b5f6;
                box-shadow: 0 0 0 2px rgba(100, 181, 246, 0.2);
            }
            .proactive-response {
                background-color: #2e3035;
                border: 2px solid #81c784;
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
                color: #81c784;
            }
            .suggestion-box {
                background-color: #3e3e3e;
                border: 1px solid #888;
                border-radius: 8px;
                padding: 0.8rem;
                margin: 0.5rem 0;
                color: #f0f2f6;
            }
            .sample-question {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 0.5rem;
                margin: 0.25rem 0;
                color: #f0f2f6;
                cursor: pointer;
                transition: background-color 0.2s;
            }
            .sample-question:hover { background-color: #444; }
            .chat-container::-webkit-scrollbar { width: 8px; }
            .chat-container::-webkit-scrollbar-track { background: #222; border-radius: 4px; }
            .chat-container::-webkit-scrollbar-thumb { background: #555; border-radius: 4px; }
            .chat-container::-webkit-scrollbar-thumb:hover { background: #777; }

            /* New styles for homepage appeal */
            .hero-section {
                background: linear-gradient(135deg, #1a4e5c 0%, #0e1117 100%); /* Gradient background */
                padding: 2.5rem 2rem; /* Adjusted padding */
                border-radius: 15px;
                text-align: center;
                margin-bottom: 2.5rem; /* Adjusted margin */
                box-shadow: 0 6px 20px rgba(0,0,0,0.4); /* Stronger shadow */
            }
            .hero-section h1 {
                font-size: 3.2rem; /* Slightly larger font */
                color: #e0f2f7; /* Lighter blue for contrast */
                margin-bottom: 0.6rem; /* Adjusted margin */
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5); /* Text shadow */
            }
            .hero-section p {
                font-size: 1.15rem; /* Slightly larger font */
                color: #b3e5fc; /* Slightly darker blue for subtext */
                max-width: 750px; /* Adjusted max-width */
                margin: 0 auto 2rem auto; /* Adjusted margin */
            }
            .feature-card {
                background-color: #2e3035;
                padding: 1.4rem; /* Adjusted padding */
                border-radius: 12px; /* Slightly more rounded */
                margin-bottom: 1.2rem; /* Adjusted margin */
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-shadow: 0 4px 12px rgba(0,0,0,0.25); /* Stronger shadow */
                border: 1px solid #3a3f4a;
                transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; /* Smooth transition */
            }
            .feature-card:hover {
                transform: translateY(-5px); /* Lift effect on hover */
                box-shadow: 0 8px 25px rgba(0,0,0,0.4); /* More prominent shadow on hover */
            }
            .feature-card h3 {
                color: #81d4fa; /* Light blue for feature titles */
                margin-bottom: 0.8rem; /* Adjusted margin */
                font-size: 1.35rem; /* Adjusted font size */
            }
            .feature-card p {
                font-size: 1rem; /* Adjusted font size */
                line-height: 1.6; /* Adjusted line height */
                color: #c0c0c0;
            }
            .how-it-works-section {
                background-color: #1e2025;
                padding: 2rem; /* Adjusted padding */
                border-radius: 15px;
                margin-top: 2.5rem; /* Adjusted margin */
                box-shadow: 0 6px 20px rgba(0,0,0,0.3); /* Stronger shadow */
            }
            .how-it-works-section h2 {
                color: #64b5f6;
                text-align: center;
                margin-bottom: 2rem; /* Adjusted margin */
            }
            .how-it-works-step {
                background-color: #2e3035;
                padding: 1.2rem; /* Adjusted padding */
                border-radius: 10px;
                margin-bottom: 1rem; /* Adjusted margin */
                border-left: 6px solid #64b5f6; /* Thicker border */
                transition: background-color 0.2s ease-in-out;
            }
            .how-it-works-step:hover {
                background-color: #3a3f4a; /* Slightly lighter background on hover */
            }
            .how-it-works-step p {
                font-size: 1rem; /* Adjusted font size */
                color: #f0f2f6;
            }
            .call-to-action-button {
                background-color: #4CAF50; /* Green for call to action */
                color: white;
                padding: 0.9rem 1.8rem; /* Larger padding */
                border-radius: 10px; /* More rounded */
                font-size: 1.3rem; /* Larger font */
                text-align: center;
                display: block;
                margin: 2.5rem auto 0 auto; /* Adjusted margin */
                width: 60%; /* Slightly wider */
                max-width: 350px; /* Adjusted max-width */
                transition: background-color 0.3s ease, transform 0.2s ease;
            }
            .call-to-action-button:hover {
                background-color: #45a049;
                transform: translateY(-3px); /* Subtle lift on hover */
                cursor: pointer;
            }

        </style>
        """
    else: # Light Theme
        return """
        <style>
            /* Light Theme */
            .stApp {
                background-color: #ffffff;
                color: #333333;
            }
            .stMarkdown, .stText, .stWrite, p, label, .stDataFrame, .stTextInput > div > div > input, .stButton > button, .stDownloadButton > button, .stFileUploader > label, .stSelectbox > label {
                color: #333333 !important;
            }
            h1, h2, h3, h4, h5, h6, .st-bh, .st-bi {
                color: #1f77b4 !important;
                font-weight: bold;
            }
            .stSidebar {
                background-color: #f0f2f6;
            }
            .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
                color: #1f77b4 !important;
            }
            .stSidebar .stMarkdown p, .stSidebar .stText p, .stSidebar .stWrite p {
                color: #333333 !important;
                font-size: 14px;
                line-height: 1.4;
            }
            .info-box {
                background-color: #e6e8eb;
                border-left: 5px solid #1f77b4;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
                font-size: 14px;
                color: #333333;
            }
            .info-box .info-label {
                font-weight: bold;
                color: #1f77b4;
            }
            .status-box {
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
                font-weight: bold;
            }
            .status-connected { background-color: #d4edda; color: #155724; }
            .status-disconnected { background-color: #f8d7da; color: #721c24; }
            .status-info { background-color: #d1ecf1; color: #0c5460; }
            .status-warning { background-color: #fff3cd; color: #856404; }
            .main-header {
                font-size: 3rem;
                font-weight: bold;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
            }
            .sub-header {
                font-size: 1.5rem;
                color: #555555;
                text-align: center;
                margin-bottom: 2rem;
            }
            .prediction-box {
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
                color: white;
            }
            .positive { background-color: #28a745; border: 1px solid #1c7430; color: white; }
            .negative { background-color: #dc3545; border: 1px solid #a71d2a; color: white; }
            .neutral { background-color: #17a2b8; border: 1px solid #117a8b; color: white; }
            .complaint { background-color: #ffc107; border: 1px solid #d39e00; color: #333333; }
            .hate-speech { background-color: #dc3545; border: 1px solid #a71d2a; color: white; }
            .stButton > button {
                width: 100%;
                background-color: #1f77b4;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 5px;
                font-size: 1.1rem;
                transition: background-color 0.2s;
            }
            .stButton > button:hover {
                background-color: #1565c0;
            }
            .chat-message {
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 10px;
                max-width: 80%;
                word-wrap: break-word;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .user-message {
                background-color: #e1f5fe;
                border: 1px solid #b3e5fc;
                margin-left: auto;
                text-align: right;
                color: #333333;
            }
            .bot-message {
                background-color: #f1f3f5;
                border: 1px solid #e0e0e0;
                margin-right: auto;
                color: #333333;
                text-align: left;
            }
            .chat-container {
                height: 400px;
                overflow-y: auto;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 1rem;
                background-color: #fafafa;
                margin-bottom: 1rem;
            }
            .stTextInput > div > div > input {
                background-color: #ffffff;
                color: #333333;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 0.5rem;
            }
            .stTextInput > div > div > input:focus {
                border-color: #1f77b4;
                box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.2);
            }
            .proactive-response {
                background-color: #e8f5e9;
                border: 2px solid #4caf50;
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
                color: #333333;
            }
            .suggestion-box {
                background-color: #e0e0e0;
                border: 1px solid #bdbdbd;
                border-radius: 8px;
                padding: 0.8rem;
                margin: 0.5rem 0;
                color: #333333;
            }
            .sample-question {
                background-color: #f1f3f5;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 0.5rem;
                margin: 0.25rem 0;
                color: #333333;
                cursor: pointer;
                transition: background-color 0.2s;
            }
            .sample-question:hover { background-color: #e6e8eb; }
            .chat-container::-webkit-scrollbar { width: 8px; }
            .chat-container::-webkit-scrollbar-track { background: #f1f3f5; border-radius: 4px; }
            .chat-container::-webkit-scrollbar-thumb { background: #bdbdbd; border-radius: 4px; }
            .chat-container::-webkit-scrollbar-thumb:hover { background: #a0a0a0; }

            /* New styles for homepage appeal */
            .hero-section {
                background: linear-gradient(135deg, #e0f2f7 0%, #ffffff 100%); /* Gradient background */
                padding: 2.5rem 2rem; /* Adjusted padding */
                border-radius: 15px;
                text-align: center;
                margin-bottom: 2.5rem; /* Adjusted margin */
                box-shadow: 0 6px 20px rgba(0,0,0,0.15); /* Stronger shadow */
            }
            .hero-section h1 {
                font-size: 3.2rem; /* Slightly larger font */
                color: #1f77b4;
                margin-bottom: 0.6rem; /* Adjusted margin */
                text-shadow: 1px 1px 2px rgba(0,0,0,0.1); /* Text shadow */
            }
            .hero-section p {
                font-size: 1.15rem; /* Slightly larger font */
                color: #555555;
                max-width: 750px; /* Adjusted max-width */
                margin: 0 auto 2rem auto; /* Adjusted margin */
            }
            .feature-card {
                background-color: #f8f9fa;
                padding: 1.4rem; /* Adjusted padding */
                border-radius: 12px; /* Slightly more rounded */
                margin-bottom: 1.2rem; /* Adjusted margin */
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1); /* Stronger shadow */
                border: 1px solid #e0e0e0;
                transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; /* Smooth transition */
            }
            .feature-card:hover {
                transform: translateY(-5px); /* Lift effect on hover */
                box-shadow: 0 8px 25px rgba(0,0,0,0.2); /* More prominent shadow on hover */
            }
            .feature-card h3 {
                color: #17a2b8; /* Teal for feature titles */
                margin-bottom: 0.8rem; /* Adjusted margin */
                font-size: 1.35rem; /* Adjusted font size */
            }
            .feature-card p {
                font-size: 1rem; /* Adjusted font size */
                line-height: 1.6; /* Adjusted line height */
                color: #444444;
            }
            .how-it-works-section {
                background-color: #f1f3f5;
                padding: 2rem; /* Adjusted padding */
                border-radius: 15px;
                margin-top: 2.5rem; /* Adjusted margin */
                box-shadow: 0 6px 20px rgba(0,0,0,0.15); /* Stronger shadow */
            }
            .how-it-works-section h2 {
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem; /* Adjusted margin */
            }
            .how-it-works-step {
                background-color: #ffffff;
                padding: 1.2rem; /* Adjusted padding */
                border-radius: 10px;
                margin-bottom: 1rem; /* Adjusted margin */
                border-left: 6px solid #1f77b4; /* Thicker border */
                transition: background-color 0.2s ease-in-out;
            }
            .how-it-works-step:hover {
                background-color: #e6e8eb; /* Slightly darker background on hover */
            }
            .how-it-works-step p {
                font-size: 1rem; /* Adjusted font size */
                color: #333333;
            }
            .call-to-action-button {
                background-color: #28a745; /* Green for call to action */
                color: white;
                padding: 0.9rem 1.8rem; /* Larger padding */
                border-radius: 10px; /* More rounded */
                font-size: 1.3rem; /* Larger font */
                text-align: center;
                display: block;
                margin: 2.5rem auto 0 auto; /* Adjusted margin */
                width: 60%; /* Slightly wider */
                max-width: 350px; /* Adjusted max-width */
                transition: background-color 0.3s ease, transform 0.2s ease;
            }
            .call-to-action-button:hover {
                background-color: #218838;
                transform: translateY(-3px); /* Subtle lift on hover */
                cursor: pointer;
            }
        </style>
        """

# Apply the selected theme
st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)
# --- End of Dynamic CSS ---

# API configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None

def predict_tweet(tweet_text, user_id=None, use_transformer=True):
    """Make prediction using the API"""
    try:
        payload = {
            "text": tweet_text,
            "user_id": user_id
        }
        endpoint = "/predict/transformer" if use_transformer else "/predict"
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def get_model_info():
    """Get model information from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def create_probability_chart(probabilities):
    """Create a bar chart for prediction probabilities"""
    df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])
    df['Probability'] = df['Probability'] * 100
    
    fig = px.bar(
        df, 
        x='Class', 
        y='Probability',
        color='Probability',
        color_continuous_scale='RdYlGn',
        title="Prediction Probabilities",
        labels={'Probability': 'Probability (%)', 'Class': 'Classification'}
    )
    
    fig.update_layout(
        xaxis_title="Classification",
        yaxis_title="Probability (%)",
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f0f2f6' if st.session_state.theme == 'dark' else '#333333')
    )
    
    return fig


def get_sentiment_color(prediction):
    """Get color class based on prediction label"""
    prediction_lower = prediction.lower()
    if "positive" in prediction_lower:
        return "positive"
    elif "complaint" in prediction_lower:
        return "complaint"
    elif "network" in prediction_lower or "reliability" in prediction_lower:
        return "negative"
    elif "privacy" in prediction_lower or "hate" in prediction_lower:
        return "hate-speech"
    else:
        return "neutral"

def generate_proactive_response(tweet_text, prediction, confidence, probabilities):
    """Generate AI chatbot-style response based on classification"""
    response_templates = {
        "MPESA complaint": f"📢 It looks like you're having an MPESA issue. We're sorry for the inconvenience. Please rest assured that your transaction is being reviewed and we'll get back to you shortly.",
        "Customer care complaint": f"🙋‍♂️ Thank you for reaching out to Safaricom Care. A customer representative will assist you shortly.",
        "Network reliability problem": f"📶 Our network is currently experiencing technical issues in some areas. Our technical team is working round the clock to restore full service.",
        "Data protection and privacy concern": f"🔐 Thank you for raising this concern. Safaricom takes data protection seriously and we are reviewing the matter.",
        "Internet or airtime bundle complaint": f"📲 We acknowledge the reported internet bundles problem. Our team is looking to improve the data deals and coverage for ease of using internet bundles.",
        "Neutral": f"🎉 We're glad you're enjoying our services! Your positive feedback keeps us going. Thank you!",
        "Hate Speech": f"🤖 We are sorry if our services are not up to per with your expectations. We are working round the clock to provide reliable services."
    }

    # If the prediction is not in the templates, default to 'Neutral'
    message = response_templates.get(prediction, response_templates["Neutral"])

    return {
        "message": message,
        "category": prediction,
        "confidence": confidence,
        "tweet_text": tweet_text
    }


def chatbot_response(user_input, use_transformer=True):
    """Generate chatbot response using Rasa or fallback"""
    bot_message = ""
    analysis_result = None

    if RASA_AVAILABLE:
        rasa_client = RasaClient()
        if rasa_client.is_available():
            rasa_response = rasa_client.send_message(user_input)
            if rasa_response and len(rasa_response) > 0:
                bot_message = rasa_response[0].get('text', 'Sorry, I couldn\'t process that.')
                # Ensure thorough cleaning of the response
                bot_message = strip_html_tags(bot_message)
                analysis_result = predict_tweet(user_input, use_transformer=use_transformer)
            else:
                # Fallback to proactive response if Rasa is available but doesn't respond
                analysis_result = predict_tweet(user_input, use_transformer=use_transformer)
                if analysis_result:
                    proactive_response = generate_proactive_response(
                        user_input, analysis_result['prediction'], analysis_result['confidence'], analysis_result['probabilities']
                    )
                    bot_message = strip_html_tags(proactive_response['message'])
                else:
                    bot_message = "Sorry, I couldn't analyze that tweet. Please try again."
        else:
            # Fallback if Rasa client is not available
            analysis_result = predict_tweet(user_input, use_transformer=use_transformer)
            if analysis_result:
                proactive_response = generate_proactive_response(
                    user_input, analysis_result['prediction'], analysis_result['confidence'], analysis_result['probabilities']
                )
                bot_message = strip_html_tags(proactive_response['message'])
            else:
                bot_message = "Sorry, I couldn't analyze that tweet. Please try again."
    else:
        # If Rasa is not available at all
        analysis_result = predict_tweet(user_input, use_transformer=use_transformer)
        if analysis_result:
            proactive_response = generate_proactive_response(
                user_input, analysis_result['prediction'], analysis_result['confidence'], analysis_result['probabilities']
            )
            bot_message = strip_html_tags(proactive_response['message'])
        else:
            bot_message = "Sorry, I couldn't analyze that tweet. Please try again."
    
    return bot_message, analysis_result
    
# --- Callback function to handle chat submission ---
def handle_chat_submit():
    """Handles the 'Send' button click to process user input and clear the text box."""
    if st.session_state.chat_input.strip():
        user_message = st.session_state.chat_input.strip()
        st.session_state.chat_history.append({'role': 'user', 'content': user_message})
        
        # Determine which model to use from the radio button state
        use_transformer_flag = (st.session_state.get('model_choice', 'Transformer') == "Transformer")
        
        bot_response, analysis_result = chatbot_response(user_message, use_transformer_flag)
        st.session_state.chat_history.append({'role': 'assistant', 'content': bot_response})
        
        # Clear the chat input after submission
        st.session_state.chat_input = ""
        # We don't need `st.experimental_rerun()` here, as Streamlit automatically re-runs
        # after a session state change initiated by a widget.

# --- Callback for clearing chat ---
def clear_chat_callback():
    st.session_state.chat_history = []
    # Set a flag to clear the input on the next rerun
    st.session_state.clear_input_on_next_run = True
    st.rerun()

# --- Page Functions ---

# Callback for "Get Started Now!" button
def navigate_to_tweet_analysis():
    st.session_state.current_page = "📝 Tweet Analysis"
    st.rerun() # Force a rerun to immediately update the page

def home_page():
    """Displays the home page with overall app information and enhanced styling."""
    
    # Hero Section
    st.markdown(
        """
        <div class="hero-section">
            <h1>Safarimeter: The Pulse of Public Opinion 📲</h1>
            <p>AI that listens. Instantly spot complaints, understand sentiment, and take action — all from Twitter conversations.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # The button needs to be outside the markdown for Streamlit to handle its click
    st.button("Get Started Now!", key="get_started_button", on_click=navigate_to_tweet_analysis, use_container_width=True)


    st.markdown("---")
    st.markdown('<h2 class="sub-header">🪄 Platform Capabilities</h2>', unsafe_allow_html=True)
    
    # Key Functionalities presented in columns/cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📝 Tweet Analysis</h3>
                <p>Classify individual tweets to understand their sentiment, and identify specific issues like network reliability complaints, MPESA complaints, Customer care issues or hate speech towards Safaricom. Get instant proactive responses.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📋 Batch Analysis</h3>
                <p>Upload CSV files containing multiple tweets for bulk classification. Visualize the distribution of the predictions and download detailed results for further analysis.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🤖 AI Assistant</h3>
                <p>Interact with an AI-powered chatbot that can provide insights and responses based on tweet classifications, helping with automating customer service interactions.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    
    # Modifying the CSS class for the heading to remove the large bottom margin.
    # Instead of `sub-header`, a new class `tight-header` or a direct style can be used.
    # Let's create a new CSS class to avoid affecting other sub-headers.
    st.markdown('<style>.tight-header { font-size: 1.5rem; color: #b0bec5; text-align: center; margin-bottom: 1rem; }</style>', unsafe_allow_html=True)
    st.markdown('<h2 class="tight-header">⛏️ Basic Workflow</h2>', unsafe_allow_html=True)

    # How it Works section with emojis
    st.markdown(
        """
        <div class="how-it-works-section">
            <div class="how-it-works-step">
                <p><strong>1. Data Ingestion:</strong> 📥 Tweets are fed into the system either individually or in batches via CSV uploads.</p>
            </div>
            <div class="how-it-works-step">
                <p><strong>2. AI-Powered Classification:</strong> 🧠 Our robust FastAPI backend, powered by Transformer-based and Scikit-learn models, processes the tweets to classify their sentiment and intent.</p>
            </div>
            <div class="how-it-works-step">
                <p><strong>3. Instant Insights:</strong> 💡 Get immediate predictions, confidence scores, and probability distributions for each tweet.</p>
            </div>
            <div class="how-it-works-step">
                <p><strong>4. Proactive Engagement:</strong> 💬 The AI Assistant generates automated, context-aware responses, streamlining your customer service workflow.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def tweet_analysis_page():
    """Handles single tweet analysis and displays results."""
    st.markdown('<h2 class="main-header">📝 Single Tweet Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter a tweet to get an instant classification and proactive response.</p>', unsafe_allow_html=True)

    use_transformer_flag = (st.session_state.get('model_choice', 'Transformer') == "Transformer")

    tweet_input = st.text_area("Enter a tweet to analyze:", placeholder="Type your tweet here...", height=120)
    user_id = st.text_input("User ID (optional):", placeholder="Enter user ID")
    
    if st.button("🔍 Analyze Tweet", type="primary"):
        if tweet_input.strip():
            with st.spinner("Analyzing tweet..."):
                result = predict_tweet(tweet_input, user_id, use_transformer_flag)
                if result:
                    st.success("✅ Analysis completed!")
                    prediction = result['prediction']
                    confidence = result['confidence']
                    probabilities = result['probabilities']
                    color_class = get_sentiment_color(prediction)
                    st.markdown(f"""
                    <div class="prediction-box {color_class}">
                        <h3>🎯 Prediction: {prediction.upper()}</h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    proactive_response = generate_proactive_response(
                        tweet_input, 
                        result['prediction'], 
                        result['confidence'], 
                        result['probabilities']
                    )
                    st.markdown(f"""
                    <div class="proactive-response">
                        <h4>🤖 AI Assistant Response</h4>
                        <p>{proactive_response['message']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(create_probability_chart(probabilities), use_container_width=True)
                    st.subheader("📊 Detailed Probabilities")
                    prob_df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])
                    prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(prob_df, use_container_width=True)
                else:
                    st.error("❌ Failed to analyze tweet. Please try again.")
        else:
            st.warning("⚠️ Please enter a tweet to analyze.")

def batch_analysis_page():
    """Handles batch tweet analysis from a CSV file."""
    st.markdown('<h2 class="main-header">📋 Batch Tweet Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a CSV file containing tweets for bulk classification and download the results.</p>', unsafe_allow_html=True)

    use_transformer_flag = (st.session_state.get('model_choice', 'Transformer') == "Transformer")

    uploaded_file = st.file_uploader("Upload CSV file with tweets (should have a 'text' column)", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'text' in df.columns:
                st.write(f"📊 Found {len(df)} tweets to analyze")
                if st.button("🔍 Analyze All Tweets", type="primary"):
                    results = []
                    with st.spinner("Analyzing tweets..."):
                        for index, row in df.iterrows():
                            tweet_text = row['text']
                            bot_response, analysis_result = chatbot_response(tweet_text, use_transformer=use_transformer_flag)
                            
                            if analysis_result:
                                results.append({
                                    'text': tweet_text,
                                    'prediction': analysis_result['prediction'],
                                    'confidence': analysis_result['confidence'],
                                    'probabilities': analysis_result['probabilities'],
                                    'chatbot_response': bot_response # Store the chatbot response
                                })
                    if results:
                        results_df = pd.DataFrame(results)
                        st.success(f"✅ Analyzed {len(results)} tweets successfully!")
                        st.subheader("📊 Analysis Summary")
                        summary = results_df['prediction'].value_counts()
                        fig = px.pie(values=summary.values, names=summary.index, title="Prediction Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                        st.subheader("📋 Detailed Results")
                        display_df = results_df[['text', 'prediction', 'confidence', 'chatbot_response']].copy()
                        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
                        st.dataframe(display_df, use_container_width=True)
                        
                        st.subheader("📥 Download Results")
                        
                        # CSV Download
                        csv_data = results_df[['text', 'prediction', 'confidence', 'chatbot_response']].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download as CSV",
                            data=csv_data,
                            file_name=f"safaricom_tweet_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_csv"
                        )

                        # PDF-like Text Report Download
                        pdf_report_content = "Safaricom Tweet Analysis Report\n"
                        pdf_report_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        
                        for idx, row in results_df.iterrows():
                            pdf_report_content += f"--- Tweet {idx + 1} ---\n"
                            pdf_report_content += f"Tweet: {row['text']}\n"
                            pdf_report_content += f"Prediction: {row['prediction']} (Confidence: {row['confidence']:.2%})\n"
                            pdf_report_content += f"Chatbot Response: {row['chatbot_response']}\n\n"
                        
                        st.download_button(
                            label="Download as Text Report (for PDF conversion)",
                            data=pdf_report_content.encode('utf-8'),
                            file_name=f"safaricom_tweet_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            key="download_txt_report",
                            help="Download this text file and use your browser's 'Print to PDF' option for a PDF document."
                        )
                    else:
                        st.error("❌ Failed to analyze tweets. Please check your data.")
            else:
                st.error("❌ CSV file must contain a 'text' column")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

def ai_assistant_page():
    """Provides an AI chatbot interface with proper HTML tag stripping."""
    st.markdown('<h2 class="main-header">🤖 AI Assistant</h2>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Chat with the AI about Safaricom tweets and get instant responses.</p>', unsafe_allow_html=True)

    # Initialize chat history if not present
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'last_proactive_response' not in st.session_state:
        st.session_state.last_proactive_response = None

    # Initialize the flag for clearing input if not already present
    if 'clear_input_on_next_run' not in st.session_state:
        st.session_state.clear_input_on_next_run = False

    # Display chat messages
    with st.container():
        for message in st.session_state.chat_history:
            # Create columns for alignment (user on right, bot on left)
            cols = st.columns([4, 1] if message['role'] == 'user' else [1, 4])
            
            with cols[0] if message['role'] == 'assistant' else cols[1]:
                # Clean the message content thoroughly
                clean_content = strip_html_tags(message['content'])
                
                # Apply appropriate styling class
                message_class = "user-message" if message['role'] == 'user' else "bot-message"
                
                # Display the message with proper styling but no HTML
                st.markdown(
                    f"""
                    <div class="chat-message {message_class}">
                        {clean_content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # Chat input and buttons
    st.markdown("---")
    
    # Determine the current value for the chat input, and clear if flag is set
    current_chat_input_value = st.session_state.get('chat_input', '')
    if st.session_state.clear_input_on_next_run:
        current_chat_input_value = ""
        st.session_state.clear_input_on_next_run = False  # Reset the flag immediately

    # Use the `value` parameter to control the input's content
    st.text_input(
        "Ask me about a tweet:",
        value=current_chat_input_value,
        placeholder="Type a tweet or question...",
        key="chat_input",
        on_change=handle_chat_submit
    )

    col_send, col_clear = st.columns([1, 1])

    with col_send:
        st.button(
            "📤 Send",
            key="chat_send",
            on_click=handle_chat_submit,
            use_container_width=True
        )
    
    with col_clear:
        if st.button("🗑️ Clear", key="clear_chat", use_container_width=True, on_click=clear_chat_callback):
            pass  # The clearing logic is in the callback

    st.markdown("---")
    st.write("**💡 Try these examples:**")
    sample_questions = [
        "Safaricom network is very slow today",
        "Thank you Safaricom for the excellent service!",
        "Safaricom customer service was very helpful",
        "I hate Safaricom, they are stealing our money"
    ]
    
    for i, question in enumerate(sample_questions):
        if st.button(
            f"📝 {question[:30]}{'...' if len(question) > 30 else ''}",
            key=f"sample_{i}",
            use_container_width=True
        ):
            # Add user question to chat history (cleaned)
            st.session_state.chat_history.append({
                'role': 'user',
                'content': strip_html_tags(question)
            })
            
            # Get bot response (which will be cleaned by chatbot_response)
            use_transformer_flag = (st.session_state.get('model_choice', 'Transformer') == "Transformer")
            bot_response, analysis_result = chatbot_response(question, use_transformer_flag)
            
            # Add bot response to chat history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': bot_response
            })
            
            st.rerun()

# --- New System Info Page Function ---
def system_info_page():
    """Displays system information including API health and model details."""
    st.markdown('<h2 class="main-header">⚙️ System Information</h2>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">View the current status of the API, chatbot, and model details.</p>', unsafe_allow_html=True)

    # Fetch system information
    api_healthy, health_info = check_api_health()
    model_info = get_model_info()

    st.subheader("🚀 API Status")
    if api_healthy:
        st.markdown("""
        <div class="status-box status-connected">
            ✅ <strong>FastAPI API Endpoint Connected</strong>
        </div>
        """, unsafe_allow_html=True)
        if health_info:
            st.markdown(f"""
            <div class="info-box">
                <div><span class="info-label">Status:</span> {html.escape(health_info.get('status', 'Unknown'))}</div>
                <div><span class="info-label">Version:</span> {html.escape(health_info.get('version', 'N/A'))}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-box status-disconnected">
            ❌ <strong>API Endpoint Disconnected</strong>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🤖 Chatbot Status")
    if RASA_AVAILABLE:
        st.markdown("""
        <div class="status-box status-info">
            ✅ <strong>Enhanced Chatbot Available (Rasa)</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-box status-warning">
            ⚠️ <strong>Basic Chatbot Mode (Rasa not available)</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🧠 Model Details")
    if model_info:
        st.markdown(f"""
        <div class="info-box">
            <div><span class="info-label">Transformer Model Type:</span> {html.escape(model_info.get('transformer_model_type', 'N/A'))}</div>
            <div><span class="info-label">Transformer Classes:</span> {html.escape(', '.join(model_info.get('transformer_classes', {}).values()) if isinstance(model_info.get('transformer_classes'), dict) else 'N/A')}</div>
            <div><span class="info-label">Sklearn Model Type:</span> {html.escape(model_info.get('sklearn_model_type', 'N/A'))}</div>
            <div><span class="info-label">Sklearn Classes:</span> {html.escape(', '.join(model_info.get('classes', [])) if isinstance(model_info.get('classes'), list) else 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No model information available. Ensure API is running and accessible.")
    
    st.markdown("---")
    st.subheader("📈 App Statistics")
    st.markdown(f"""
    <div class="info-box">
        <div><span class="info-label">Total Chat Messages:</span> {len(st.session_state.get('chat_history', []))}</div>
        <div><span class="info-label">Current App Time:</span> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        <div><span class="info-label">App Version:</span> 1.0.0</div>
    </div>
    """, unsafe_allow_html=True)


# --- Main Application Logic ---
def main():
    # Initialize session state for page navigation if not already set
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Home"
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = 'Transformer' # Default model choice

    # Sidebar for navigation and global settings
    with st.sidebar:
        st.title("🚀 Navigation Panel")
        st.markdown("---")
        
        # Page Navigation
        page_options = ["🏠 Home", "📝 Tweet Analysis", "📋 Batch Analysis", "🤖 AI Assistant", "⚙️ System Info"]
        page_selection = st.radio(
            "Navigate",
            page_options,
            key="page_selector",
            index=page_options.index(st.session_state.current_page)
        )
        st.session_state.current_page = page_selection

        st.markdown("---")
        st.subheader("⚙️ Global Settings")
        
        # Model Choice (moved to sidebar as a global setting)
        st.session_state.model_choice = st.radio(
            "Choose Model", 
            ["Transformer", "Sklearn"], 
            index=0 if st.session_state.model_choice == 'Transformer' else 1,
            key="model_choice_radio"
        )

        # Theme Toggle
        theme_icon = "☀️" if st.session_state.theme == 'dark' else "🌙"
        if st.button(f"{theme_icon} Toggle Theme", use_container_width=True):
            st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
            st.rerun()
        
        st.markdown("---")
        st.info("This app helps classify Safaricom tweets into categories like complaints, positive feedback, and more.")


    # Display the selected page content
    if st.session_state.current_page == "🏠 Home":
        home_page()
    elif st.session_state.current_page == "📝 Tweet Analysis":
        tweet_analysis_page()
    elif st.session_state.current_page == "📋 Batch Analysis":
        batch_analysis_page()
    elif st.session_state.current_page == "🤖 AI Assistant":
        ai_assistant_page()
    elif st.session_state.current_page == "⚙️ System Info":
        system_info_page()

    # --- Add Footer ---
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.9em;">
            AI-enabled Tweet classifier, powered by Scikit-Learn and Hugging Face Transformers<br>
            Developed by Patrick Maina, Christine Ndungu, Teresia Njoki and George Nyandusi<br>
            &copy; 2025 All Rights Reserved.
        </div>
        """,
    unsafe_allow_html=True
)
# --- End of Footer ---

if __name__ == "__main__":
    main()
