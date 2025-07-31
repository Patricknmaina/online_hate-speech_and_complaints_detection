import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List
import warnings

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass


class FeatureEngineering:
    """
    A class for feature engineering, data cleaning, preprocessing, and vectorization
    for hate speech and complaints detection.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the FeatureEngineering class with the dataset.
        
        Args:
            data (pd.DataFrame): The input dataset
        """
        self.data = data.copy()
        self.vectorizer = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_text_features(self, text_column: str = 'Content') -> pd.DataFrame:
        """
        Generate basic text features like character count, word count, and sentence count.
        
        Args:
            text_column (str): Name of the text column to analyze
            
        Returns:
            pd.DataFrame: DataFrame with additional text features
        """
        if text_column not in self.data.columns:
            print(f"Column '{text_column}' not found in the dataset.")
            return self.data
            
        print("Generating text features...")
        
        # Character count
        self.data['chars'] = self.data[text_column].apply(len)
        
        # Word count
        self.data['words'] = self.data[text_column].apply(lambda x: len(word_tokenize(x)))
        
        # Sentence count
        self.data['sentences'] = self.data[text_column].apply(lambda x: len(nltk.sent_tokenize(x)))
        
        print(f"Added features: chars, words, sentences")
        return self.data
    
    def remove_columns(self, columns_to_remove: List[str]) -> pd.DataFrame:
        """
        Remove specified columns from the dataset.
        
        Args:
            columns_to_remove (List[str]): List of column names to remove
            
        Returns:
            pd.DataFrame: DataFrame with specified columns removed
        """
        existing_columns = [col for col in columns_to_remove if col in self.data.columns]
        if existing_columns:
            self.data = self.data.drop(columns=existing_columns, axis=1)
            print(f"Removed columns: {existing_columns}")
        else:
            print("No specified columns found in the dataset.")
            
        return self.data
    
    def clean_text(self, text_column: str = 'Content') -> pd.DataFrame:
        """
        Clean the text data by removing URLs, special characters, and normalizing text.
        
        Args:
            text_column (str): Name of the text column to clean
            
        Returns:
            pd.DataFrame: DataFrame with cleaned text
        """
        if text_column not in self.data.columns:
            print(f"Column '{text_column}' not found in the dataset.")
            return self.data
            
        print("Cleaning text data...")
        
        def clean_data(text):
            # Remove URLs
            text = re.sub(r'http\S+|www.\S+', '', text)
            
            # Remove only the @ and # signs, keep the words
            text = re.sub(r'[@#]', '', text)
            
            # Remove special characters (but keep numbers)
            text = re.sub(r'[^A-Za-z0-9\s]', '', text)
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove additional whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        self.data[text_column] = self.data[text_column].apply(clean_data)
        print("Text cleaning completed.")
        
        return self.data
    
    def tokenize_text(self, text_column: str = 'Content', tokenized_column: str = 'Tokenized Text') -> pd.DataFrame:
        """
        Tokenize the text data.
        
        Args:
            text_column (str): Name of the source text column
            tokenized_column (str): Name for the tokenized column
            
        Returns:
            pd.DataFrame: DataFrame with tokenized text
        """
        if text_column not in self.data.columns:
            print(f"Column '{text_column}' not found in the dataset.")
            return self.data
            
        print("Tokenizing text...")
        
        self.data[tokenized_column] = self.data[text_column].apply(
            lambda document: word_tokenize(document.lower())
        )
        
        print("Tokenization completed.")
        return self.data
    
    def lemmatize_text(self, tokenized_column: str = 'Tokenized Text', 
                       lemmatized_column: str = 'Lematized Text') -> pd.DataFrame:
        """
        Lemmatize the tokenized text.
        
        Args:
            tokenized_column (str): Name of the tokenized text column
            lemmatized_column (str): Name for the lemmatized column
            
        Returns:
            pd.DataFrame: DataFrame with lemmatized text
        """
        if tokenized_column not in self.data.columns:
            print(f"Column '{tokenized_column}' not found in the dataset.")
            return self.data
            
        print("Lemmatizing text...")
        
        lemmatizer = WordNetLemmatizer()
        self.data[lemmatized_column] = self.data[tokenized_column].apply(
            lambda word_tokens: [lemmatizer.lemmatize(token) for token in word_tokens]
        )
        
        print("Lemmatization completed.")
        return self.data
    
    def create_processed_text(self, lemmatized_column: str = 'Lematized Text', 
                             processed_column: str = 'processed_text') -> pd.DataFrame:
        """
        Join lemmatized tokens back into strings for vectorization.
        
        Args:
            lemmatized_column (str): Name of the lemmatized text column
            processed_column (str): Name for the processed text column
            
        Returns:
            pd.DataFrame: DataFrame with processed text
        """
        if lemmatized_column not in self.data.columns:
            print(f"Column '{lemmatized_column}' not found in the dataset.")
            return self.data
            
        print("Creating processed text...")
        
        self.data[processed_column] = self.data[lemmatized_column].apply(lambda x: ' '.join(x))
        
        print("Processed text creation completed.")
        return self.data
    
    def handle_missing_labels(self, label_column: str = 'Labels', 
                             fill_value: str = 'unknown') -> pd.DataFrame:
        """
        Handle missing values in the label column.
        
        Args:
            label_column (str): Name of the label column
            fill_value (str): Value to fill missing labels with
            
        Returns:
            pd.DataFrame: DataFrame with handled missing labels
        """
        if label_column not in self.data.columns:
            print(f"Column '{label_column}' not found in the dataset.")
            return self.data
            
        print(f"Handling missing values in '{label_column}' column...")
        
        # Fill missing values
        self.data[label_column].fillna(fill_value, inplace=True)
        
        # Filter out unknown labels
        initial_count = len(self.data)
        self.data = self.data[self.data[label_column] != fill_value]
        final_count = len(self.data)
        
        print(f"Removed {initial_count - final_count} rows with '{fill_value}' labels.")
        
        return self.data
    
    def prepare_data_for_modeling(self, text_column: str = 'processed_text', 
                                 label_column: str = 'Labels',
                                 exclude_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare the data for modeling by separating features and target.
        
        Args:
            text_column (str): Name of the text column to use as features
            label_column (str): Name of the label column
            exclude_columns (List[str]): Columns to exclude from features
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target variables
        """
        if exclude_columns is None:
            exclude_columns = [label_column, 'Date']
        
        # Create feature matrix
        X = self.data.drop(exclude_columns, axis=1)
        y = self.data[label_column]
        
        print(f"Prepared data for modeling:")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split the data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"Splitting data with test_size={test_size}, random_state={random_state}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def vectorize_text(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      text_column: str = 'processed_text',
                      vectorizer_type: str = 'count',
                      max_features: int = 5000,
                      stop_words: str = 'english') -> Tuple:
        """
        Vectorize the text data using CountVectorizer or TfidfVectorizer.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Testing features
            text_column (str): Name of the text column to vectorize
            vectorizer_type (str): Type of vectorizer ('count' or 'tfidf')
            max_features (int): Maximum number of features
            stop_words (str): Stop words to remove
            
        Returns:
            Tuple: (X_train_vectorized, X_test_vectorized, vectorizer)
        """
        print(f"Vectorizing text using {vectorizer_type.upper()} vectorizer...")
        
        if vectorizer_type.lower() == 'count':
            self.vectorizer = CountVectorizer(
                max_features=max_features, 
                stop_words=stop_words
            )
        elif vectorizer_type.lower() == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features, 
                stop_words=stop_words
            )
        else:
            raise ValueError("vectorizer_type must be 'count' or 'tfidf'")
        
        # Fit on training data and transform both training and testing data
        X_train_vectorized = self.vectorizer.fit_transform(X_train[text_column])
        X_test_vectorized = self.vectorizer.transform(X_test[text_column])
        
        print(f"Training features shape: {X_train_vectorized.shape}")
        print(f"Testing features shape: {X_test_vectorized.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        return X_train_vectorized, X_test_vectorized, self.vectorizer
    
    def save_processed_data(self, output_path: str = 'data/cleaned_safaricom_data.csv') -> None:
        """
        Save the processed dataset to a CSV file.
        
        Args:
            output_path (str): Path where to save the processed data
        """
        try:
            self.data.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def get_processing_summary(self) -> dict:
        """
        Get a summary of the processing steps performed.
        
        Returns:
            dict: Summary of processing steps
        """
        summary = {
            'original_shape': self.data.shape if self.data is not None else None,
            'current_columns': list(self.data.columns) if self.data is not None else [],
            'vectorizer_type': type(self.vectorizer).__name__ if self.vectorizer else None,
            'vocabulary_size': len(self.vectorizer.vocabulary_) if self.vectorizer else None
        }
        
        return summary
    
    def process_pipeline(self, text_column: str = 'Content', 
                        label_column: str = 'Labels',
                        vectorizer_type: str = 'count',
                        max_features: int = 5000,
                        test_size: float = 0.2,
                        random_state: int = 42,
                        save_processed: bool = True,
                        output_path: str = 'data/cleaned_safaricom_data.csv') -> Tuple:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            text_column (str): Name of the text column
            label_column (str): Name of the label column
            vectorizer_type (str): Type of vectorizer to use
            max_features (int): Maximum number of features for vectorization
            test_size (float): Proportion of data for testing
            random_state (int): Random seed
            save_processed (bool): Whether to save processed data
            output_path (str): Path to save processed data
            
        Returns:
            Tuple: (X_train_vectorized, X_test_vectorized, y_train, y_test, vectorizer)
        """
        print("Starting feature engineering pipeline...")
        print("=" * 50)
        
        # Step 1: Generate text features
        self.generate_text_features(text_column)
        
        # Step 2: Remove unnecessary columns
        self.remove_columns(['Tweet ID', 'URL'])
        
        # Step 3: Clean text
        self.clean_text(text_column)
        
        # Step 4: Tokenize text
        self.tokenize_text(text_column)
        
        # Step 5: Lemmatize text
        self.lemmatize_text()
        
        # Step 6: Create processed text
        self.create_processed_text()
        
        # Step 7: Handle missing labels
        self.handle_missing_labels(label_column)
        
        # Step 8: Save processed data if requested
        if save_processed:
            self.save_processed_data(output_path)
        
        # Step 9: Prepare data for modeling
        X, y = self.prepare_data_for_modeling()
        
        # Step 10: Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size, random_state)
        
        # Step 11: Vectorize text
        X_train_vectorized, X_test_vectorized, vectorizer = self.vectorize_text(
            X_train, X_test, vectorizer_type=vectorizer_type, max_features=max_features
        )
        
        print("Feature engineering pipeline completed!")
        print("=" * 50)
        
        return X_train_vectorized, X_test_vectorized, y_train, y_test, vectorizer 