# data_prep/zero_shot.py

"""
Zero-shot classification using Hugging Face Transformers and OpenAI's GPT-4o-mini API

This script demonstrates how to perform zero-shot classification using both a pre-trained transformer model from Hugging Face and OpenAI's GPT-4o-mini API.

"""

import pandas as pd
from transformers import pipeline
from openai import OpenAI

# load the dataset
df = pd.read_csv("test.csv")
print("Dataset loaded with shape:", df.shape)
print(df.head())

# hugging face zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels
candidate_labels = [
    "customer care complaint",
    "MPESA complaint",
    "network reliability problem",
    "data protection issue",
    "general inquiry",
    "positive feedback",
    "hate speech",
]

# run the zero-shot classification
hf_results = []
for text in df["Text"].tolist():
    prediction = classifier(text, candidate_labels)
    hf_results.append(prediction)

# Convert results to a DataFrame
hf_preds_df = pd.DataFrame(hf_results)
print("\nHugging Face predictions:")
print(hf_preds_df.head())

# Extract best label predictions
df["hf_predicted_label"] = [res["labels"][0] for res in hf_results]
df["hf_predicted_score"] = [res["scores"][0] for res in hf_results]


# OpenAI zero-shot classification

# Initialize the OpenAI client
client = OpenAI()

# Function to classify text using OpenAI's GPT-4o-mini
def classify_with_openai(text, labels):
    """
    Use OpenAI to classify text into one of the given candidate labels.
    """
    prompt = f"""
    You are a classifier. Assign the following text to one of the given categories.

    Text: "{text}"

    Categories: {labels}

    Reply with only the category name.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # change if needed
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0
    )
    return response.choices[0].message.content.strip()

# run classification with OpenAI
openai_results = []
for text in df["Text"].tolist():
    label = classify_with_openai(text, candidate_labels)
    openai_results.append(label)

df["openai_predicted_label"] = openai_results

# save the results
df.to_csv("data/raw/labeled_data_openai.csv", index=False)
print("\nPredictions saved to zero_shot_predictions_full.csv")

# predict for sample texts
sample_texts = [
    "My MPESA transaction failed and my money is missing",
    "Safaricom's internet is very reliable these days",
    "This is hateful speech against the community",
]

print("\nCustom Text Predictions:")
for text in sample_texts:
    hf_pred = classifier(text, candidate_labels)
    openai_pred = classify_with_openai(text, candidate_labels)

    print(f"\nText: {text}")
    print("HF Predicted:", hf_pred["labels"][0], " (score:", round(hf_pred["scores"][0], 3), ")")
    print("OpenAI Predicted:", openai_pred)
