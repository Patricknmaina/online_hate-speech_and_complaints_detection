import pandas as pd
import numpy as np
import plotly.express as px
from typing import Optional, Tuple


class DataLoader:
    """
    A class for loading and analyzing datasets for hate speech and complaints detection.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the DataLoader with the path to the dataset.
        
        Args:
            file_path (str): Path to the CSV file containing the dataset
        """
        self.file_path = file_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from the specified file path.
        
        Returns:
            pd.DataFrame: The loaded dataset
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully from {self.file_path}")
            return self.data
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found.")
            return None
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def check_dataset_info(self) -> None:
        """
        Display basic information about the dataset including info, shape, and data types.
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
            
        print("Dataset Information:")
        print("=" * 50)
        print(self.data.info())
        print(f"\nDataset shape: {self.data.shape}")
        print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Drop the one missing value in the 'Labels' column, if present
        # missing_labels = self.data['Labels'].isnull().sum() if 'Labels' in self.data.columns else 0
        # if missing_labels > 0:
        #     print(f"\nFound {missing_labels} missing value(s) in 'Labels' column. Dropping them.")
        #     self.data = self.data[self.data['Labels'].notnull()]
        #     print(f"New dataset shape after dropping missing 'Labels': {self.data.shape}")
        # else:
        #     print("\nNo missing values found in 'Labels' column.")

    # Add this method to your data_prep/data_loader.py file

    def handle_missing_values(self, strategy: str = 'comprehensive') -> pd.DataFrame:
        """
        Handle missing values in the dataset using different strategies.
        
        Args:
            strategy (str): Strategy to use ('comprehensive', 'drop_all', 'fill_numeric')
            
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
        
        print("Handling missing values...")
        print("=" * 40)
        
        original_shape = self.data.shape
        print(f"Original dataset shape: {original_shape}")
        
        # Analyze missing values
        missing_summary = self.data.isnull().sum()
        columns_with_missing = missing_summary[missing_summary > 0]
        
        if len(columns_with_missing) == 0:
            print("✓ No missing values found!")
            return self.data
        
        print(f"Found missing values in {len(columns_with_missing)} columns:")
        for col, count in columns_with_missing.items():
            percentage = (count / len(self.data)) * 100
            print(f"  {col}: {count} ({percentage:.2f}%)")
        
        if strategy == 'comprehensive':
            # Comprehensive strategy: handle each column appropriately
            for column in columns_with_missing.index:
                missing_count = columns_with_missing[column]
                
                if column == 'Labels':
                    # Drop rows with missing labels (target variable)
                    print(f"Dropping {missing_count} rows with missing labels")
                    self.data = self.data[self.data[column].notnull()]
                    
                elif column in ['Tweet ID', 'URL', 'Date', 'Content']:
                    # Drop rows with missing essential data
                    print(f"Dropping {missing_count} rows with missing {column}")
                    self.data = self.data[self.data[column].notnull()]
                    
                elif column in ['Likes', 'Retweets', 'Replies', 'Quotes', 'Views']:
                    # Fill numeric engagement metrics with 0
                    print(f"Filling {missing_count} missing values in {column} with 0")
                    self.data[column] = self.data[column].fillna(0)
                    
                else:
                    # For any other columns, drop rows
                    print(f"Dropping {missing_count} rows with missing {column}")
                    self.data = self.data[self.data[column].notnull()]
        
        elif strategy == 'drop_all':
            # Drop all rows with any missing values
            print("Dropping all rows with any missing values")
            self.data = self.data.dropna()
        
        elif strategy == 'fill_numeric':
            # Fill numeric columns with 0, drop rows with missing categorical/text
            print("Filling numeric columns with 0, dropping rows with missing categorical/text")
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            categorical_columns = self.data.select_dtypes(include=['object']).columns
            
            # Fill numeric columns
            for col in numeric_columns:
                if col in columns_with_missing.index:
                    self.data[col] = self.data[col].fillna(0)
            
            # Drop rows with missing categorical/text
            for col in categorical_columns:
                if col in columns_with_missing.index:
                    self.data = self.data[self.data[col].notnull()]
        
        # Verify results
        final_missing = self.data.isnull().sum()
        remaining_missing = final_missing[final_missing > 0]
        
        if len(remaining_missing) == 0:
            print("✓ All missing values successfully handled!")
        else:
            print("⚠ Remaining missing values:")
            for col, count in remaining_missing.items():
                print(f"  {col}: {count}")
        
        # Summary
        final_shape = self.data.shape
        rows_removed = original_shape[0] - final_shape[0]
        print(f"\nSummary:")
        print(f"  Original rows: {original_shape[0]}")
        print(f"  Final rows: {final_shape[0]}")
        print(f"  Rows removed: {rows_removed}")
        print(f"  Data retention: {((final_shape[0] / original_shape[0]) * 100):.2f}%")
        
        return self.data
    
    def check_duplicates(self) -> int:
        """
        Check for duplicate rows in the dataset.
        
        Returns:
            int: Number of duplicate rows found
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return 0
            
        duplicates = self.data.duplicated().sum()
        print(f"Number of duplicate rows: {duplicates}")
        
        if duplicates > 0:
            print("Duplicate rows found. Consider removing them.")
        else:
            print("No duplicate rows found.")
            
        return duplicates
    
    def remove_duplicates(self, inplace: bool = True) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            inplace (bool): If True, modify the original dataframe. If False, return a copy.
            
        Returns:
            pd.DataFrame: Dataset with duplicates removed
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
            
        initial_shape = self.data.shape
        if inplace:
            self.data.drop_duplicates(inplace=True)
            print(f"Removed {initial_shape[0] - self.data.shape[0]} duplicate rows.")
            return self.data
        else:
            cleaned_data = self.data.drop_duplicates()
            print(f"Removed {initial_shape[0] - cleaned_data.shape[0]} duplicate rows.")
            return cleaned_data
    
    def analyze_target_distribution(self, target_column: str = 'Labels') -> pd.Series:
        """
        Analyze the distribution of the target variable.
        
        Args:
            target_column (str): Name of the target column
            
        Returns:
            pd.Series: Distribution of the target variable
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
            
        if target_column not in self.data.columns:
            print(f"Column '{target_column}' not found in the dataset.")
            return None
            
        distribution = self.data[target_column].value_counts()
        print("Distribution of the target variable:")
        print("=" * 50)
        print(distribution)
        
        return distribution
    
    def plot_target_distribution(self, target_column: str = 'Labels') -> None:
        """
        Create a bar plot of the target variable distribution.
        
        Args:
            target_column (str): Name of the target column
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
            
        if target_column not in self.data.columns:
            print(f"Column '{target_column}' not found in the dataset.")
            return
            
        distribution = self.data[target_column].value_counts().reset_index()
        distribution.columns = ['Labels', 'count']
        
        fig = px.bar(
            distribution,
            x='Labels',
            y='count',
            labels={'Labels': 'Label', 'count': 'Count'},
            title='Distribution of Target Variable'
        )
        
        # Get unique labels to assign a color to each
        unique_labels = distribution['Labels']
        color_discrete_sequence = px.colors.qualitative.Plotly
        
        # If there are more labels than colors in the palette, repeat the palette
        if len(unique_labels) > len(color_discrete_sequence):
            color_discrete_sequence = color_discrete_sequence * (len(unique_labels) // len(color_discrete_sequence) + 1)
        
        fig.update_traces(marker_color=color_discrete_sequence[:len(unique_labels)])
        fig.show()
    
    def get_dataset_summary(self) -> dict:
        """
        Get a comprehensive summary of the dataset.
        
        Returns:
            dict: Dictionary containing dataset summary information
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return {}
            
        summary = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicates': self.data.duplicated().sum(),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2
        }
        
        print("Dataset Summary:")
        print("=" * 50)
        print(f"Shape: {summary['shape']}")
        print(f"Columns: {len(summary['columns'])}")
        print(f"Missing values: {sum(summary['missing_values'].values())}")
        print(f"Duplicates: {summary['duplicates']}")
        print(f"Memory usage: {summary['memory_usage_mb']:.2f} MB")
        
        return summary 