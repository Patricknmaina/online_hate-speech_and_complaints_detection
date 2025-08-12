## ONLINE HATE SPEECH AND COMPLAINTS DETECTION AND CLASSIFICATION

### Project summary
This NLP project automates the detection and classification of hate speech and customer complaints directed at Safaricom via Twitter. It uses both machine learning and transformer-based models to turn tweets into actionable categories, helping Safaricom improve customer care, protect its brand, and respond more efficiently to complaints.

### Data understanding
The data used in this project contained 6,146 tweets that were scrapped from X using N8N and TwitterAPI.io
The data included features like Tweet ID, content, likes, retweets, replies, quotes, views, date. we generated labels using OpenAI GPT-4 and Zero-Shot Classification and we came up with the following classes:
- Customer Care Complaint
- MPESA Complaint
- Network Reliability Problem
- Internet/Airtime Bundle Complaint
- Data Protection/Privacy Concern
- Neutral
- Hate Speech

### Problem satatement
Safaricom faces growing challenges in managing hate speech directed at its brand, products, and employees on digital platforms. Customers frequently use social media to express frustrations with services such as M-PESA, customer care, and network coverage—often using emotionally charged or hostile language. The sheer volume of such content makes it difficult to manually monitor and respond effectively, exposing the company to reputational damage and delayed customer resolution. Moreover, distinguishing hate from genuine complaints is complicated by the informal, multilingual nature of online discourse in Kenya. Current moderation practices lack the scalability and contextual nuance required to handle these dynamics.

### Business Objectives
The primary client for this NLP project is Safaricom plc. By analyzing and classifying tweets about their products, safaricom can gain authentic feedback that traditional methods might miss. This real-time access to customer sentiment will enable them to quickly identify trends, preferences, and potential issues, facilitating proactive engagement and timely adjustments to their strategies.

### Project objectives

• Identify and flag hate speech and user complaints in real time.
• Distinguish between negative feedback/complaints and harmful speech, ensuring genuine concerns are not misclassified.
• Support brand protection strategies through early detection of online hostility.
• Contribute to safer, more respectful online interactions between the organization and the general public.

### Exploratory Data Analysis
We performed a systematic investigation of the dataset to extract insights, evaluate feature distributions, assess the relationship between the feature and target variables, and identify anomalies, outliers or data quality issues. This was helpful in choosing the right modelling techniques.
We discovered that there was a class imbalance as the neutral tweets dominated the dataset
We employed visualizations and descriptive statistics to show the undelying patterns, trends and correlations within the data

![download%20%281%29.png](attachment:download%20%281%29.png)

![Picture1.png](attachment:Picture1.png)

![download 2.png](<attachment:download 2.png>)

### Data Cleaning And Preprocessing
We introduced a custom TweetPreprocessor class designed to automate and standardize the text cleaning and feature extraction process for Twitter sentiment analysis. the class performed tasks such as:
Data Cleaning:

-Removal of URLs, user mentions, hashtags, special characters, and repeated characters.

-Expansion of common English contractions (e.g., "can't" → "cannot").

-Normalization of whitespace and punctuation.

Text Preprocessing:

-Conversion to lowercase for consistency.

-Tokenization of text into words.

-Removal of stopwords and short words.

-Lemmatization to reduce words to their base forms.

Feature Extraction:

-Supports both TF-IDF and Count Vectorization for transforming cleaned tweets into numerical feature vectors.

-Configurable options for n-gram range, vocabulary size, and document frequency thresholds.

Pipeline Integration:

The class is compatible with scikit-learn pipelines, enabling seamless integration with machine learning workflows.

### Modelling

In the modeling phase of this NLP project, the goal was to evaluate multiple approaches for classifying Safaricom tweets into their respective categories, combining both traditional Machine Learning models and SOTA (state-of-the-art) transformer-based models.
We trained and evaluated three classical methods: Logistic Regression, Naive Bayes, and Random Forest. These models were chosen for their proven effectiveness in text classification tasks and their relatively low computational cost.
To leverage advances in deep learning for multilingual NLP, we also fine-tuned two Hugging Face transformer models: XLM-RoBERTa and mBERT.
Model evaluation was carried out using precision, recall, F1-score (main performance metrics)
and model accuracy to provide a balanced view of performance.

### Evaluation


|          | Logistic Regression | Naive Bayes | Random Forest |   mBERT  | XLM RoBERTa |
|----------|---------------------|-------------|---------------|----------|-------------|
| Accuracy |    0.6959           | 0.6846      | 0.6886        | 0.7131   | 0.7885      |
| Precision|    0.7151           | 0.6770      | 0.6606        | 0.7266   | 0.7877      |
| Recall   |    0.6959           | 0.6846      | 0.6886        | 0.7131   | 0.7885      |
| F1 score |    0.7027           | 0.6752      | 0.6537        | 0.7185   | 0.7866      |

### Deployment

In this project, deployment focuses on enabling Safaricom’s customer service team to automatically classify tweets directed at @Safaricom, ultimately improving response efficiency and customer satisfaction.

This project is deployed as a full-stack NLP application that allows users (e.g., Safaricom moderators or analysts) to input tweets and receive real-time classification results. The deployment stack includes a user-friendly dashboard, a backend API for model inference, and optional chatbot integration for conversational triage.
 1. Frontend: Streamlit Dashboard
- Purpose: Provides an intuitive interface for users to input tweets, view classification results, and explore model insights.
- Features:
- Text input box for manual tweet entry
- CSV upload for batch classification
- Real-time prediction display (class label + confidence score)
- Visualizations: class distribution, word clouds, tweet length histograms
- Tech: Built with Streamlit, enabling rapid prototyping and deployment with minimal overhead.
 2. Backend: FastAPI Inference Server
- Purpose: Handles incoming requests from the dashboard and returns model predictions.
- Workflow:
- Receives tweet text via POST request
- Applies preprocessing using the TweetPreprocessor class
- Feeds cleaned text into selected model (Logistic Regression or XLM-RoBERTa)
- Returns predicted class and confidence score
- Tech: Built with FastAPI, chosen for its speed, scalability, and easy integration with Python ML pipelines.
 3. Model Serving
- Models Deployed:
- Logistic Regression: Lightweight, fast, and interpretable
- XLM-RoBERTa: Deep multilingual transformer for nuanced classification
- Model Selection: Users can toggle between models based on speed vs. accuracy trade-offs.
- Storage: Models are serialized using joblib or pickle and loaded into memory at API startup.
 4. Optional Chatbot: Rasa Integration
- Purpose: Enables conversational triage of tweets or user queries.
- Use Case: A moderator can ask, “Is this tweet hate speech?” and receive a natural language response.
- Tech: Rasa is integrated into the Streamlit app via REST API or WebSocket.
 5. End-to-End Architecture
User Input (Tweet) 
   ↓
Streamlit Dashboard (Frontend)
   ↓
FastAPI Server (Backend)
   ↓
TweetPreprocessor → ML Model (LogReg or XLM-RoBERTa)
   ↓
Prediction Output (Class + Confidence)


 6. Hosting Options
- Local Deployment: Run Streamlit and FastAPI locally for internal testing.
- Cloud Deployment:
- Streamlit Sharing or Render for dashboard hosting
- Railway, Heroku, or Azure App Service for FastAPI backend
- Docker containers for reproducible deployment
- Security: API endpoints can be secured with token-based authentication if exposed publicly.

### Conclussion

The final deployment architecture integrates Streamlit for a user-friendly dashboard, FastAPI for a high-performance backend layer, and Rasa for conversational AI support, enabling interactive and accessible customer engagement. This system not only automates classification, but also provides actionable insights through visual analytics, empowering Safaricom’s customer care team to respond more efficiently, identifying emerging issues in real-time, and foster a healthier online environment.
Through these capabilities, the project provides a scalable foundation for proactive brand reputation management and customer relationship improvement, positioning Safaricom as a leader in data-driven customer engagement within the telecommunications sector.






```python

```
