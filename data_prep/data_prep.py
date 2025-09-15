# data_prep/data_preprocessing.py

"""
Data preprocessing script for Twitter dataset

Data containing text and labels is read from a CSV file and different preprocessing techniques are applied.

The cleaned data is then split into training, validation, and test sets, ensuring stratification of labels.
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split
import collections


class SafaricomPreprocessor:
    def __init__(self, filepath):
        """Initialize with a CSV file containing 'Text' and 'Labels' columns."""
        self.filepath = filepath
        self.df = pd.read_csv(filepath)

        # contractions dictionary
        self.contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
            "'cause": "because", "could've": "could have", "couldn't": "could not",
            "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
            "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have",
            "he'll": "he will", "he'll've": "he will have", "he's": "he is", "how'd": "how did",
            "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would",
            "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
            "I've": "I have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
            "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
            "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
            "mightn't've": "might not have", "must've": "must have"
        }

    # exploratory data analysis
    def basic_info(self):
        """Show dataset info, nulls, duplicates, label distribution."""
        print("\nDataset Info")
        print(self.df.info())
        print("\nMissing Values")
        print(self.df.isnull().sum())
        print("\nDuplicate Rows")
        print(self.df.duplicated().sum())
        print("\nLabel Distribution")
        print(self.df['Labels'].value_counts())
        return self.df.head()

    # data cleaning
    def expand_contractions(self, text):
        """Expand contractions in text."""
        for contraction, expansion in self.contractions.items():
            text = re.sub(contraction, expansion, text)
        return text

    def clean_text(self, text):
        """Clean individual text string."""
        text = self.expand_contractions(str(text))
        text = re.sub(r'http\S+', '', text)      # remove URLs
        text = re.sub(r'@\w+', '', text)         # remove mentions
        text = re.sub(r'#\w+', '', text)         # remove hashtags
        text = re.sub(r'\d+', '', text)          # remove digits
        text = re.sub(r'[^\w\s]', '', text)      # remove punctuation
        text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces
        return text.lower()

    def preprocess(self, save_path="cleaned_data.csv"):
        """Apply cleaning to dataset and save results."""
        self.df['Text'] = self.df['Text'].apply(self.clean_text)
        self.df.to_csv(save_path, index=False)
        print(f"\nCleaned dataset saved to {save_path}")
        return self.df

    # data splitting
    def split_data(self, test_size=0.3, val_size=0.5, random_state=42,
                   train_path="train.csv", val_path="val.csv", test_path="test.csv"):
        """Split dataset into train, validation, and test sets and save them."""
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            self.df['Text'].tolist(),
            self.df['Labels'].tolist(),
            test_size=test_size,
            random_state=random_state,
            stratify=self.df['Labels'].tolist()
        )

        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts,
            temp_labels,
            test_size=val_size,
            random_state=random_state,
            stratify=temp_labels
        )

        train_df = pd.DataFrame({'Text': train_texts, 'Labels': train_labels})
        val_df = pd.DataFrame({'Text': val_texts, 'Labels': val_labels})
        test_df = pd.DataFrame({'Text': test_texts, 'Labels': test_labels})

        # save splits
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print("\nData splits saved to:", train_path, val_path, test_path)
        print("Train size:", len(train_texts), "Validation size:", len(val_texts), "Test size:", len(test_texts))
        print("Train label distribution:", collections.Counter(train_labels))
        print("Validation label distribution:", collections.Counter(val_labels))
        print("Test label distribution:", collections.Counter(test_labels))

        return train_df, val_df, test_df

    # reload and verify splits
    def reload_splits(self, train_path="data/processed/train.csv", val_path="data/processed/val.csv", test_path="data/processed/test.csv"):
        """Reload train, validation, and test sets."""
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        print("\n--- Reloaded Datasets ---")
        print("Train:", train_df.shape, "Validation:", val_df.shape, "Test:", test_df.shape)
        print("\nTrain label distribution:", train_df['Labels'].value_counts())
        print("Validation label distribution:", val_df['Labels'].value_counts())
        print("Test label distribution:", test_df['Labels'].value_counts())

        return train_df, val_df, test_df


# implementation
if __name__ == "__main__":
    processor = SafaricomPreprocessor("data/labeled_data_openai.csv")

    # exploratory check
    processor.basic_info()

    # clean and save dataset
    processor.preprocess("data/cleaned_and_processed_safaricom_data.csv")

    # split dataset
    train_df, val_df, test_df = processor.split_data()

    # reload & verify splits
    processor.reload_splits()
