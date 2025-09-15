# train_scripts/ml_train.py

"""
Script to train the model and save it for the API to use.
"""

import pandas as pd
import numpy as np
import joblib
import os
from data_prep.data_loader import DataLoader
from data_prep.feature_engineering import FeatureEngineering
from data_prep.modeling import Modeling

def train_and_save_model():
    """
    Train the model and save it for the API to use.
    """
    print("Starting model training...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    data_loader = DataLoader('data/raw/labeled_data_openai.csv')
    data = data_loader.load_data()
    
    if data is None:
        print("Failed to load dataset. Exiting.")
        return False
    
    # Step 2: Feature engineering
    print("\n2. Feature engineering...")
    feature_eng = FeatureEngineering(data)
    
    # Run the complete feature engineering pipeline
    X_train_vectorized, X_test_vectorized, y_train, y_test, vectorizer = feature_eng.process_pipeline(
        text_column='Content',
        label_column='Labels',
        vectorizer_type='count',
        max_features=5000,
        test_size=0.2,
        random_state=42,
        save_processed=True,
        output_path='data/raw/cleaned_and_processed_safaricom_data.csv'
    )
    
    # Step 3: Train models
    print("\n3. Training models...")
    modeling = Modeling()
    
    # Train and evaluate all models
    results = modeling.train_and_evaluate_all_models(
        X_train_vectorized, y_train, X_test_vectorized, y_test
    )
    
    # Get the best model
    best_model = modeling.best_model
    best_model_name = modeling.best_model_name
    
    if best_model is None:
        print("No best model found. Exiting.")
        return False
    
    print(f"\nBest model: {best_model_name}")
    
    # Step 4: Save the model and vectorizer
    print("\n4. Saving model and vectorizer...")
    
    # Save the best model
    model_path = "models/best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save the vectorizer
    vectorizer_path = "models/vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to {vectorizer_path}")
    
    # Save model metadata
    metadata = {
        "best_model_name": best_model_name,
        "model_type": type(best_model).__name__,
        "vectorizer_type": type(vectorizer).__name__,
        "classes": best_model.classes_.tolist() if hasattr(best_model, 'classes_') else None,
        "feature_count": len(vectorizer.vocabulary_) if vectorizer else None,
        "training_results": results
    }
    
    metadata_path = "models/model_metadata.pkl"
    joblib.dump(metadata, metadata_path)
    print(f"Model metadata saved to {metadata_path}")
    
    # Step 5: Test the saved model
    print("\n5. Testing saved model...")
    
    # Test with a sample tweet
    test_tweet = "Safaricom network is very slow today, I can't even browse properly"
    
    # Load the saved model and vectorizer
    loaded_model = joblib.load(model_path)
    loaded_vectorizer = joblib.load(vectorizer_path)
    
    # Test prediction
    prediction = modeling.predict_new_text(test_tweet, loaded_vectorizer, loaded_model)
    print(f"Test tweet: {test_tweet}")
    print(f"Prediction: {prediction}")
    
    print("\nModel training and saving completed successfully!")
    return True

if __name__ == "__main__":
    success = train_and_save_model()
    if success:
        print("\n✅ Model is ready for the API!")
        print("You can now start the API with: python api/main.py")
    else:
        print("\n❌ Model training failed!") 