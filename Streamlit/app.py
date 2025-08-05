"""
This is the Streamlit app for the Safaricom Tweet Classifier.
It contains the user interface for the app, as well as the FastAPI API endpoint integration.
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

# Add the parent directory to Python path to find AI_powered_chatbot
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from AI_powered_chatbot.rasa_client import RasaClient
    RASA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import RasaClient: {e}")
    RASA_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Safaricom Tweet Classifier",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

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
    if RASA_AVAILABLE:
        rasa_client = RasaClient()
        if rasa_client.is_available():
            rasa_response = rasa_client.send_message(user_input)
            if rasa_response and len(rasa_response) > 0:
                bot_message = rasa_response[0].get('text', 'Sorry, I couldn\'t process that.')
                result = predict_tweet(user_input, use_transformer=use_transformer)
                return bot_message, result
            else:
                result = predict_tweet(user_input, use_transformer=use_transformer)
                if result:
                    proactive_response = generate_proactive_response(
                        user_input, result['prediction'], result['confidence'], result['probabilities']
                    )
                    return proactive_response['message'], result
                else:
                    return "Sorry, I couldn't analyze that tweet. Please try again.", None
    result = predict_tweet(user_input, use_transformer=use_transformer)
    if result:
        proactive_response = generate_proactive_response(
            user_input, result['prediction'], result['confidence'], result['probabilities']
        )
        return proactive_response['message'], result
    else:
        return "Sorry, I couldn't analyze that tweet. Please try again.", None
    
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
        
        # This line is the key fix: We set the state of the input widget to an empty string.
        # This is allowed within a callback and will clear the box on the next re-run.
        st.session_state.chat_input = ""
        # We don't need `st.experimental_rerun()` here, as Streamlit automatically re-runs
        # after a session state change initiated by a widget.

def main():
    # Header
    st.markdown('<h1 class="main-header">📱 Safaricom Tweet Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze tweets directed towards Safaricom for sentiment and classification</p>', unsafe_allow_html=True)

    # Check API health
    api_healthy, health_info = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ API is not running. Please initialize the FastAPI server.")
        return
    
    # Sidebar with enhanced visibility
    use_transformer = st.radio("⚙️ Choose Model", ["Transformer", "Sklearn"], index=0)
    use_transformer_flag = (use_transformer == "Transformer")

    with st.sidebar:
        st.title("🔧 Settings")
        
        # Theme Toggle
        theme_icon = "☀️" if st.session_state.theme == 'dark' else "🌙"
        if st.button(f"{theme_icon} Toggle Theme", use_container_width=True):
            st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
            st.rerun()

    # Call get_model_info()
    model_info = get_model_info()

    # Determine selected model name from toggle
    selected_model = "Transformer" if use_transformer_flag else "Sklearn"
    selected_model_type = model_info.get('transformer_model_type') if use_transformer_flag else model_info.get('sklearn_model_type')
    selected_classes = model_info.get('transformer_classes') if use_transformer_flag else model_info.get('classes')

    # Convert class list or dict to comma-separated string
    if isinstance(selected_classes, dict):  # transformer_classes is a dict like {0: "MPESA Complaint", ...}
        class_names = ', '.join(selected_classes.values())
    elif isinstance(selected_classes, list):
        class_names = ', '.join(selected_classes)
    else:
        class_names = "Unknown"

    # Display in sidebar with highlight for selected model
    with st.sidebar:
        st.subheader("Model Information")
        st.markdown(f"""
        <div class="info-box">
            <div><span class="info-label">Selected Model:</span> <span style="color:#81d4fa;"><strong>{selected_model}</strong></span></div>
            <div><span class="info-label">Model Type:</span> {selected_model_type or 'Unknown'}</div>
            <div><span class="info-label">Classes:</span> {class_names}</div>
        </div>
        """, unsafe_allow_html=True)

        # API status
        st.subheader("API Status")
        if api_healthy:
            st.markdown("""
            <div class="status-box status-connected">
                ✅ <strong>FastAPI API Endpoint Connected</strong>
            </div>
            """, unsafe_allow_html=True)
            if health_info:
                st.markdown(f"""
                <div class="info-box">
                    <div><span class="info-label">Status:</span> {health_info.get('status', 'Unknown')}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-box status-disconnected">
                ❌ <strong>API Endpoint Disconnected</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Rasa status
        st.subheader("Chatbot Status")
        if RASA_AVAILABLE:
            st.markdown("""
            <div class="status-box status-info">
                🤖 <strong>Enhanced Chatbot Available</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-box status-warning">
                ⚠️ <strong>Basic Chatbot Mode</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional sidebar information
        st.markdown("---")
        st.subheader("📊 Quick Stats")
        st.markdown(f"""
        <div class="info-box">
            <div><span class="info-label">Total Chat Messages:</span> {len(st.session_state.get('chat_history', []))}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # System information
        st.markdown("---")
        st.subheader("ℹ️ System Info")
        st.markdown(f"""
        <div class="info-box">
            <div><span class="info-label">Current Time:</span> {datetime.now().strftime('%H:%M:%S')}</div>
            <div><span class="info-label">Version:</span> 1.0.0</div>
        </div>
        """, unsafe_allow_html=True)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'last_proactive_response' not in st.session_state:
        st.session_state.last_proactive_response = None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Tweet Analysis")
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
        
        st.markdown("---")
        st.subheader("📋 Batch Analysis")
        uploaded_file = st.file_uploader("Upload CSV file with tweets (should have a 'text' column)", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    st.write(f"📊 Found {len(df)} tweets to analyze")
                    if st.button("🔍 Analyze All Tweets"):
                        results = []
                        with st.spinner("Analyzing tweets..."):
                            for index, row in df.iterrows():
                                result = predict_tweet(row['text'], use_transformer=use_transformer_flag)
                                if result:
                                    results.append({
                                        'text': row['text'],
                                        'prediction': result['prediction'],
                                        'confidence': result['confidence'],
                                        'probabilities': result['probabilities']
                                    })
                        if results:
                            results_df = pd.DataFrame(results)
                            st.success(f"✅ Analyzed {len(results)} tweets successfully!")
                            st.subheader("📊 Analysis Summary")
                            summary = results_df['prediction'].value_counts()
                            fig = px.pie(values=summary.values, names=summary.index, title="Prediction Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                            st.subheader("📋 Detailed Results")
                            display_df = results_df[['text', 'prediction', 'confidence']].copy()
                            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
                            st.dataframe(display_df, use_container_width=True)
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name=f"safaricom_tweet_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error("❌ Failed to analyze tweets. Please check your data.")
                else:
                    st.error("❌ CSV file must contain a 'text' column")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with col2:
        st.subheader("🤖 AI Assistant")
        st.write("**Chat with me about Safaricom tweets!**")
        
        # Chat container to display messages
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>AI Assistant:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)

        # Chat input and buttons
        # The key to clearing the input is the use of the `on_change` parameter or a callback.
        # We will use a callback function on the `text_input` and the `button`.
        
        # Using a text_input with an on_change callback is the cleanest way.
        chat_input = st.text_input(
            "Ask me about a tweet:", 
            placeholder="Type a tweet or question...", 
            key="chat_input",
            on_change=handle_chat_submit
        )

        col_send, col_clear = st.columns([1, 1])

        with col_send:
            # We add a button that also triggers the callback.
            # `on_click` is the recommended way to associate a button with a function.
            st.button(
                "📤 Send", 
                key="chat_send", 
                on_click=handle_chat_submit, 
                use_container_width=True
            )
        
        with col_clear:
            if st.button("🗑️ Clear", key="clear_chat", use_container_width=True):
                st.session_state.chat_history = []
                # Clear the chat input as well.
                st.session_state.chat_input = ""
                st.rerun()
        
        st.markdown("---")
        st.write("**💡 Try these examples:**")
        sample_questions = ["Safaricom network is very slow today", "Thank you Safaricom for the excellent service!", "Safaricom customer service was very helpful", "I hate Safaricom, they are stealing our money"]
        for i, question in enumerate(sample_questions):
            if st.button(f"📝 {question[:30]}{'...' if len(question) > 30 else ''}", key=f"sample_{i}", use_container_width=True):
                # When a sample question is clicked, we directly update the state and rerun.
                st.session_state.chat_history.append({'role': 'user', 'content': question})
                use_transformer_flag = (st.session_state.get('model_choice', 'Transformer') == "Transformer")
                bot_response, analysis_result = chatbot_response(question, use_transformer_flag)
                st.session_state.chat_history.append({'role': 'assistant', 'content': bot_response})
                st.rerun()

if __name__ == "__main__":
    main()