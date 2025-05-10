# data_preparation.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

def load_data(file_path):
    """
    Load data from the specified CSV file path.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Preprocess the dataset by separating features and target, 
    encoding categorical variables, and scaling numerical variables.
    Args:
        df (pd.DataFrame): The dataset to preprocess.
    Returns:
        X (pd.DataFrame): Features (independent variables).
        y (pd.Series): Target (dependent variable).
        preprocessor (ColumnTransformer): Preprocessing pipeline.
    """
    # Define the target variable and features
    y = df['no_show']  # Adjust according to your dataset
    X = df.drop(columns=['no_show', 'booking_id'])  # Drop target and any irrelevant columns
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # Create the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns),  # Scaling numerical columns
            ('cat', OneHotEncoder(), categorical_cols)  # One-hot encoding categorical columns
        ])
    
    return X, y, preprocessor

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        test_size (float): Proportion of the data to use as test set.
        random_state (int): Random state for reproducibility.
    Returns:
        X_train, X_test, y_train, y_test: Split data.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def prepare_pipeline(preprocessor, model):
    """
    Create a machine learning pipeline that applies preprocessing and model training.
    Args:
        preprocessor (ColumnTransformer): Preprocessing pipeline.
        model: A machine learning model to train.
    Returns:
        pipeline: A combined pipeline for preprocessing and model training.
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    return pipeline
