# src/main.py

from src.input_handler import get_file_path
from src.data_preparation import (
    load_data, preprocess_data, split_data, prepare_pipeline
)
from sklearn.ensemble import RandomForestClassifier
from src.model_evaluation import evaluate_model

def main():
    file_path = get_file_path()
    
    # Load and preprocess
    df = load_data(file_path)
    X, y, preprocessor = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Choose model and build pipeline
    model = RandomForestClassifier(random_state=42)
    pipeline = prepare_pipeline(preprocessor, model)

    # Train and evaluate
    pipeline.fit(X_train, y_train)
    evaluate_model(pipeline, X_test, y_test)

if __name__ == "__main__":
    main()
