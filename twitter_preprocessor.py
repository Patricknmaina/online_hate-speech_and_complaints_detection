
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class TwitterTextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Comprehensive text preprocessing pipeline for Twitter data with feature extraction.
    Compatible with scikit-learn pipelines.
    """
    
    def __init__(self, 
                 vectorizer_type='tfidf',  # 'tfidf' or 'count'
                 ngram_range=(1, 2),
                 max_features=5000,
                 min_df=2,
                 max_df=0.95,
                 min_word_length=2,
                 remove_urls=True,
                 remove_mentions=True,
                 remove_hashtags=False):
        
        self.vectorizer_type = vectorizer_type
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.min_word_length = min_word_length
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        
        # Initialize components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        
    def clean_text(self, text):
        """Clean individual text with various preprocessing steps"""
        if pd.isna(text):
            return ""
            
        text = str(text).lower()
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
            
        # Remove hashtags (optional)
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize text and apply lemmatization"""
        if not text:
            return []
            
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens: remove stopwords and short words
        filtered_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words 
            and len(token) >= self.min_word_length
            and token.isalpha()
        ]
        
        return filtered_tokens
    
    def preprocess_text(self, text):
        """Complete preprocessing pipeline for single text"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        return ' '.join(tokens)
    
    def fit(self, X, y=None):
        """Fit the preprocessor and vectorizer"""
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in X]
        
        # Initialize vectorizer
        if self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df
            )
        else:
            self.vectorizer = CountVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df
            )
        
        # Fit vectorizer
        self.vectorizer.fit(processed_texts)
        
        return self
    
    def transform(self, X):
        """Transform texts to feature vectors"""
        processed_texts = [self.preprocess_text(text) for text in X]
        return self.vectorizer.transform(processed_texts)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self):
        """Get feature names from vectorizer"""
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            return self.vectorizer.get_feature_names_out()
        else:
            return self.vectorizer.get_feature_names()
    
    def get_vocabulary(self):
        """Get vocabulary from vectorizer"""
        return self.vectorizer.vocabulary_

# Example usage:
# preprocessor = TwitterTextPreprocessor(vectorizer_type='tfidf', ngram_range=(1,2), max_features=1000)
# X_features = preprocessor.fit_transform(tweet_texts)
# feature_names = preprocessor.get_feature_names()
