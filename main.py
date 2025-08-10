#!/usr/bin/env python3
"""
Main script for hate speech and complaints detection using the modular classes.
This script demonstrates how to use the DataLoader, FeatureEngineering, and Modeling classes.
"""

import pandas as pd
import numpy as np
from data_prep.data_loader import DataLoader
from data_prep.feature_engineering import FeatureEngineering
from data_prep.modeling import Modeling


def main():
    """
    Main function to run the complete pipeline.
    """

    # Load and analyze the dataset
    print("\n1. Loading and Analyzing the Dataset")
    
    # Initialize data loader
    data_loader = DataLoader('data/safaricom_data.csv')
    
    # Load the dataset
    data = data_loader.load_data()
    
    if data is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Check dataset information
    data_loader.check_dataset_info()
    
    # Check for duplicates
    duplicates = data_loader.check_duplicates()
    
    # Analyze target distribution
    distribution = data_loader.analyze_target_distribution()
    
    # Plot target distribution
    # data_loader.plot_target_distribution()
    
    # Get dataset summary
    summary = data_loader.get_dataset_summary()
    
    # Feature Engineering and Preprocessing
    print("\n2. Feature Engineering and Data Preprocessing")

    # Initialize feature engineering
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
        output_path='data/cleaned_safaricom_data.csv'
    )
    
    # Get processing summary
    processing_summary = feature_eng.get_processing_summary()
    
    # Step 3: Modeling
    print("\n3. Machine Learning Training and Evaluation")
    
    # Initialize modeling
    modeling = Modeling()
    
    # Train and evaluate all models
    results = modeling.train_and_evaluate_all_models(
        X_train_vectorized, y_train, X_test_vectorized, y_test
    )
    
    # Compare models
    comparison_df = modeling.compare_models()
    
    # Plot confusion matrix for the best model
    # modeling.plot_confusion_matrix()


    # Step 4: Test on new data
    print("\n4. Testing on New Tweets")
    
    # Test the best model on a new tweet
    new_tweet = "There have been so many abductions in the country the last couple of months, and I bet on my life that Safaricom is taking part in it by sharing our information with those involved"
    
    prediction = modeling.predict_new_text(new_tweet, vectorizer)
    print(f"New tweet: {new_tweet}")
    print(f"Predicted label: {prediction}")
    
    # Test another example
    new_tweet2 = "Safaricom network is very slow today, I can't even browse properly"
    prediction2 = modeling.predict_new_text(new_tweet2, vectorizer)
    print(f"\nNew tweet: {new_tweet2}")
    print(f"Predicted label: {prediction2}")


def run_individual_steps():
    """
    Function to run individual steps for demonstration purposes.
    """
    print("\n" + "=" * 60)
    print("INDIVIDUAL STEPS DEMONSTRATION")
    print("=" * 60)
    
    # Step 1: Data Loading
    print("\nStep 1: Data Loading")
    print("-" * 30)
    
    loader = DataLoader('data/safaricom_data.csv')
    data = loader.load_data()
    loader.check_dataset_info()
    loader.check_duplicates()
    
    # Step 2: Feature Engineering
    print("\nStep 2: Feature Engineering")
    print("-" * 30)
    
    fe = FeatureEngineering(data)
    
    # Individual steps
    fe.generate_text_features()
    fe.remove_columns(['Tweet ID', 'URL'])
    fe.clean_text()
    fe.tokenize_text()
    fe.lemmatize_text()
    fe.create_processed_text()
    fe.handle_missing_labels()
    
    # Prepare for modeling
    X, y = fe.prepare_data_for_modeling()
    X_train, X_test, y_train, y_test = fe.split_data(X, y)
    X_train_vec, X_test_vec, vectorizer = fe.vectorize_text(X_train, X_test)
    
    # Step 3: Modeling
    print("\nStep 3: Modeling")
    print("-" * 30)
    
    modeler = Modeling()
    
    # Train individual models
    lr_model = modeler.train_logistic_regression(X_train_vec, y_train)
    lr_results = modeler.evaluate_model(lr_model, X_test_vec, y_test, 'Logistic Regression')
    
    nb_model = modeler.train_naive_bayes(X_train_vec, y_train)
    nb_results = modeler.evaluate_model(nb_model, X_test_vec, y_test, 'Naive Bayes')
    
    # Compare results
    comparison = modeler.compare_models()
    
    print("\nIndividual steps demonstration completed!")


if __name__ == "__main__":
    # Run the complete pipeline
    main()
    
    # Uncomment the line below to run individual steps demonstration
    # run_individual_steps() 