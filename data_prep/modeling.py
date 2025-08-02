import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional, Any
import warnings
import re


class Modeling:
    """
    A class for training and evaluating machine learning models for hate speech and complaints detection.
    """
    
    def __init__(self):
        """
        Initialize the Modeling class.
        """
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_logistic_regression(self, X_train, y_train, 
                                 max_iter: int = 1000, 
                                 random_state: int = 42,
                                 class_weight: str = 'balanced',
                                 **kwargs) -> LogisticRegression:
        """
        Train a Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            max_iter (int): Maximum iterations for convergence
            random_state (int): Random seed
            class_weight (str): Class weight strategy
            **kwargs: Additional parameters for LogisticRegression
            
        Returns:
            LogisticRegression: Trained model
        """
        print("Training Logistic Regression model...")
        
        model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight,
            **kwargs
        )
        
        model.fit(X_train, y_train)
        
        self.models['logistic_regression'] = model
        print("Logistic Regression model trained successfully.")
        
        return model
    
    def train_naive_bayes(self, X_train, y_train, 
                          alpha: float = 1.0,
                          **kwargs) -> MultinomialNB:
        """
        Train a Multinomial Naive Bayes model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            alpha (float): Smoothing parameter
            **kwargs: Additional parameters for MultinomialNB
            
        Returns:
            MultinomialNB: Trained model
        """
        print("Training Naive Bayes model...")
        
        model = MultinomialNB(alpha=alpha, **kwargs)
        model.fit(X_train, y_train)
        
        self.models['naive_bayes'] = model
        print("Naive Bayes model trained successfully.")
        
        return model
    
    def train_random_forest(self, X_train, y_train,
                           n_estimators: int = 100,
                           random_state: int = 42,
                           class_weight: str = 'balanced',
                           **kwargs) -> RandomForestClassifier:
        """
        Train a Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators (int): Number of trees
            random_state (int): Random seed
            class_weight (str): Class weight strategy
            **kwargs: Additional parameters for RandomForestClassifier
            
        Returns:
            RandomForestClassifier: Trained model
        """
        print("Training Random Forest model...")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
            **kwargs
        )
        
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        print("Random Forest model trained successfully.")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name: str = None) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test labels
            model_name (str): Name of the model for identification
            
        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics
        """
        if model_name is None:
            model_name = type(model).__name__
        
        print(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        self.results[model_name] = results
        
        # Print results
        print(f"{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(results['classification_report'])
        
        return results
    
    def train_and_evaluate_all_models(self, X_train, y_train, X_test, y_test) -> Dict[str, Dict]:
        """
        Train and evaluate all models (Logistic Regression, Naive Bayes, Random Forest).
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict[str, Dict]: Dictionary containing results for all models
        """
        print("Training and evaluating all models...")
        print("=" * 50)
        
        # Train and evaluate Logistic Regression
        lr_model = self.train_logistic_regression(X_train, y_train)
        lr_results = self.evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
        
        # Train and evaluate Naive Bayes
        nb_model = self.train_naive_bayes(X_train, y_train)
        nb_results = self.evaluate_model(nb_model, X_test, y_test, 'Naive Bayes')
        
        # Train and evaluate Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        rf_results = self.evaluate_model(rf_model, X_test, y_test, 'Random Forest')
        
        # Find the best model
        self._find_best_model()
        
        return self.results
    
    def _find_best_model(self) -> None:
        """
        Find the best performing model based on F1-score.
        """
        if not self.results:
            print("No results available to find best model.")
            return
        
        best_f1 = -1
        best_model_name = None
        
        for model_name, results in self.results.items():
            if results['f1_score'] > best_f1:
                best_f1 = results['f1_score']
                best_model_name = model_name
        
        self.best_model_name = best_model_name
        self.best_model = self.models.get(best_model_name.lower().replace(' ', '_'))
        
        print(f"\nBest model: {best_model_name} (F1-Score: {best_f1:.4f})")
    
    def plot_confusion_matrix(self, model_name: str = None, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot confusion matrix for a specific model.
        
        Args:
            model_name (str): Name of the model to plot
            figsize (Tuple[int, int]): Figure size
        """
        if model_name is None:
            if self.best_model_name:
                model_name = self.best_model_name
            else:
                print("No model specified and no best model found.")
                return
        
        if model_name not in self.results:
            print(f"No results found for model: {model_name}")
            return
        
        results = self.results[model_name]
        cm = results['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=sorted(set(results.get('y_test', []))),
                   yticklabels=sorted(set(results.get('y_test', []))),
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def compare_models(self, metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare all trained models based on specified metrics.
        
        Args:
            metrics (List[str]): List of metrics to compare (default: ['accuracy', 'precision', 'recall', 'f1_score'])
            
        Returns:
            pd.DataFrame: DataFrame with comparison results
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        if not self.results:
            print("No results available for comparison.")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            row = {'Model': model_name}
            for metric in metrics:
                if metric in results:
                    row[metric] = results[metric]
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("Model Comparison:")
        print("=" * 50)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        return comparison_df
    
    def predict_new_text(self, text: str, vectorizer, model_name_or_model = None) -> str:
        """
        Predict the label for a new text using the best model or specified model.
        
        Args:
            text (str): New text to classify
            vectorizer: Fitted vectorizer to transform the text
            model_name_or_model: Name of the model to use or the actual model object (if None, uses best model)
            
        Returns:
            str: Predicted label
        """
        # Determine which model to use
        if model_name_or_model is None:
            if self.best_model is None:
                print("No best model available. Please specify a model name or model object.")
                return None
            model = self.best_model
        elif isinstance(model_name_or_model, str):
            # If it's a string, treat it as a model name
            if model_name_or_model not in self.models:
                print(f"Model '{model_name_or_model}' not found.")
                return None
            model = self.models[model_name_or_model]
        else:
            # If it's not a string, treat it as a model object
            model = model_name_or_model
        
        # Clean the text (using the same cleaning function from feature engineering)
        cleaned_text = self._clean_text(text)
        
        # Vectorize the text
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        
        return prediction
    
    
    # define the clean text function for cleaning new tweets
    def _clean_text(self, text: str) -> str:
        """
        Clean text using the same cleaning function as in feature engineering.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
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


    # def cross_validate_model(self, model, X, y, cv: int = 5, scoring: str = 'f1_weighted') -> Dict[str, float]:
    #     """
    #     Perform cross-validation on a model.
        
    #     Args:
    #         model: Model to cross-validate
    #         X: Features
    #         y: Labels
    #         cv (int): Number of cross-validation folds
    #         scoring (str): Scoring metric
            
    #     Returns:
    #         Dict[str, float]: Cross-validation results
    #     """
    #     print(f"Performing {cv}-fold cross-validation...")
        
    #     cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
    #     results = {
    #         'mean_score': cv_scores.mean(),
    #         'std_score': cv_scores.std(),
    #         'scores': cv_scores
    #     }
        
    #     print(f"Cross-validation {scoring}: {results['mean_score']:.4f} (+/- {results['std_score'] * 2:.4f})")
        
    #     return results
    
    # def hyperparameter_tuning(self, model_type: str, X_train, y_train, 
    #                          param_grid: Dict = None, cv: int = 5) -> Any:
    #     """
    #     Perform hyperparameter tuning using GridSearchCV.
        
    #     Args:
    #         model_type (str): Type of model ('logistic_regression', 'naive_bayes', 'random_forest')
    #         X_train: Training features
    #         y_train: Training labels
    #         param_grid (Dict): Parameter grid for tuning
    #         cv (int): Number of cross-validation folds
            
    #     Returns:
    #         Any: Best model with tuned parameters
    #     """
    #     if param_grid is None:
    #         if model_type == 'logistic_regression':
    #             param_grid = {
    #                 'C': [0.1, 1, 10, 100],
    #                 'max_iter': [500, 1000],
    #                 'class_weight': ['balanced', None]
    #             }
    #         elif model_type == 'naive_bayes':
    #             param_grid = {
    #                 'alpha': [0.1, 0.5, 1.0, 2.0]
    #             }
    #         elif model_type == 'random_forest':
    #             param_grid = {
    #                 'n_estimators': [50, 100, 200],
    #                 'max_depth': [10, 20, None],
    #                 'min_samples_split': [2, 5, 10]
    #             }
        
    #     # Create base model
    #     if model_type == 'logistic_regression':
    #         base_model = LogisticRegression(random_state=42)
    #     elif model_type == 'naive_bayes':
    #         base_model = MultinomialNB()
    #     elif model_type == 'random_forest':
    #         base_model = RandomForestClassifier(random_state=42)
    #     else:
    #         raise ValueError(f"Unknown model type: {model_type}")
        
    #     # Perform grid search
    #     grid_search = GridSearchCV(
    #         base_model, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1
    #     )
        
    #     print(f"Performing hyperparameter tuning for {model_type}...")
    #     grid_search.fit(X_train, y_train)
        
    #     print(f"Best parameters: {grid_search.best_params_}")
    #     print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
    #     # Store the best model
    #     self.models[f'{model_type}_tuned'] = grid_search.best_estimator_
        
    #     return grid_search.best_estimator_
    
    # def save_model(self, model_name: str, filepath: str) -> None:
    #     """
    #     Save a trained model to disk.
        
    #     Args:
    #         model_name (str): Name of the model to save
    #         filepath (str): Path where to save the model
    #     """
    #     import joblib
        
    #     if model_name not in self.models:
    #         print(f"Model '{model_name}' not found.")
    #         return
        
    #     try:
    #         joblib.dump(self.models[model_name], filepath)
    #         print(f"Model saved to {filepath}")
    #     except Exception as e:
    #         print(f"Error saving model: {e}")
    
    # def load_model(self, model_name: str, filepath: str) -> None:
    #     """
    #     Load a trained model from disk.
        
    #     Args:
    #         model_name (str): Name to assign to the loaded model
    #         filepath (str): Path to the saved model file
    #     """
    #     import joblib
        
    #     try:
    #         model = joblib.load(filepath)
    #         self.models[model_name] = model
    #         print(f"Model loaded from {filepath}")
    #     except Exception as e:
    #         print(f"Error loading model: {e}")
    
    # def get_model_summary(self) -> Dict[str, Any]:
    #     """
    #     Get a summary of all trained models and their performance.
        
    #     Returns:
    #         Dict[str, Any]: Summary of models and results
    #     """
    #     summary = {
    #         'trained_models': list(self.models.keys()),
    #         'evaluated_models': list(self.results.keys()),
    #         'best_model': self.best_model_name,
    #         'best_f1_score': self.results.get(self.best_model_name, {}).get('f1_score', None) if self.best_model_name else None
    #     }
        
    #     print("Model Summary:")
    #     print("=" * 30)
    #     print(f"Trained models: {summary['trained_models']}")
    #     print(f"Evaluated models: {summary['evaluated_models']}")
    #     print(f"Best model: {summary['best_model']}")
    #     if summary['best_f1_score']:
    #         print(f"Best F1-score: {summary['best_f1_score']:.4f}")
        
    #     return summary 