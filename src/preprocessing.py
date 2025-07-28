# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the hotel no-show dataset:
    - Encode categorical variables
    - Scale numerical features
    - Split into features and target

    Args:
        df (pd.DataFrame): Raw DataFrame

    Returns:
        X_train, X_test, y_train, y_test: Split and processed datasets
    """

    print("🔍 Starting preprocessing...")

    # 1. Drop irrelevant columns (based on EDA)
    drop_cols = ['reservation_status', 'reservation_status_date']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # 2. Encode categorical features using LabelEncoder (simpler than OneHot for DNN/RF)
    cat_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"🔠 Encoded: {col}")

    # 3. Separate features and target
    X = df.drop("no_show", axis=1)
    y = df["no_show"]

    # 4. Scale numerical features
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    print("📏 Numerical features scaled.")

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("✅ Preprocessing complete.")
    return X_train, X_test, y_train, y_test
