# train_scripts/transformers.py

"""
This module implements a text classification model using Hugging Face's Transformers library.

It includes data loading, preprocessing, model training, evaluation, and inference functionalities.
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import Dataset


class ModelTrainer:
    def __init__(self, model_name, num_labels, train_path, val_path, test_path, output_dir="./models/transformer_results"):
        """
        Initialize the trainer with model, tokenizer, and datasets.

        Args:
            model_name (str): Hugging Face model checkpoint (e.g., "bert-base-uncased")
            num_labels (int): Number of labels for classification
            train_path (str): Path to train.csv
            val_path (str): Path to val.csv
            test_path (str): Path to test.csv
            output_dir (str): Directory to save outputs
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = output_dir

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

        # Load datasets
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)

        # Convert to HuggingFace datasets
        self.train_dataset = Dataset.from_pandas(self.train_df)
        self.val_dataset = Dataset.from_pandas(self.val_df)
        self.test_dataset = Dataset.from_pandas(self.test_df)

        # Tokenized datasets will be set later
        self.tokenized_train = None
        self.tokenized_val = None
        self.tokenized_test = None

    def tokenize_function(self, examples):
        """
        Tokenize the text column using the model's tokenizer.
        """
        return self.tokenizer(
            examples["Text"], padding="max_length", truncation=True, max_length=128
        )

    def tokenize_data(self):
        """
        Apply tokenization to train, validation, and test datasets.
        """
        self.tokenized_train = self.train_dataset.map(self.tokenize_function, batched=True)
        self.tokenized_val = self.val_dataset.map(self.tokenize_function, batched=True)
        self.tokenized_test = self.test_dataset.map(self.tokenize_function, batched=True)

        # Set format for PyTorch
        self.tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "Labels"])
        self.tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "Labels"])
        self.tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "Labels"])

    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics: accuracy, precision, recall, f1.
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="weighted")
        f1 = f1_score(labels, predictions, average="weighted")

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def setup_training_args(self, batch_size=16, epochs=3, lr=5e-5):
        """
        Setup training arguments for the HuggingFace Trainer API.
        """
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir=os.path.join(self.output_dir, "logs"),
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )

    def train(self):
        """
        Train the model using HuggingFace Trainer.
        """
        if self.tokenized_train is None:
            raise ValueError("Data not tokenized. Run `tokenize_data()` first.")

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_val,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()

    def evaluate(self):
        """
        Evaluate the model on the validation set.
        """
        if not hasattr(self, "trainer"):
            raise ValueError("Trainer not initialized. Run `train()` first.")

        results = self.trainer.evaluate()
        print("\nValidation Results:", results)
        return results

    def test(self):
        """
        Evaluate the model on the test set.
        """
        if not hasattr(self, "trainer"):
            raise ValueError("Trainer not initialized. Run `train()` first.")

        results = self.trainer.evaluate(eval_dataset=self.tokenized_test)
        print("\nTest Results:", results)
        return results
    
    def save_model(self):
        """
        Save the trained model and tokenizer to output_dir.
        """
        save_path = os.path.join(self.output_dir, "final_model")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, path=None):
        """
        Reload a trained model from the given path (or default output_dir).
        """
        if path is None:
            path = os.path.join(self.output_dir, "final_model")

        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"Model loaded from {path}")

    def predict(self, texts):
        """
        Run inference on new samples.

        Args:
            texts (list[str]): A list of text samples to classify.

        Returns:
            list[int]: Predicted class labels
        """
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.tolist()


# implementation
if __name__ == "__main__":
    trainer = ModelTrainer(
        model_name="roberta-base", # implement the XLM-RoBERTa model
        num_labels=5,  # adjust to match your dataset
        train_path="data/processed/train.csv",
        val_path="data/processed/val.csv",
        test_path="data/processed/test.csv",
        output_dir="./results"
    )

    trainer.tokenize_data()
    trainer.setup_training_args(batch_size=16, epochs=3, lr=2e-5)
    trainer.train()
    trainer.evaluate()
    trainer.test()
    trainer.save_model()

    # Example inference
    sample_texts = ["This service is terrible!", "I love using this product."]
    preds = trainer.predict(sample_texts)
    print("\nPredictions:", preds)



