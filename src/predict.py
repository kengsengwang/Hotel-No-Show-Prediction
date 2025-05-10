# predict.py
import joblib
import pandas as pd
from data_preparation import preprocess_data

def load_model(filename):
    """
    Load a trained model from a file.
    """
    return joblib.load(filename)

def predict(model, input_data):
    """
    Make predictions using the trained model.
    Args:
        model: The trained model.
        input_data (pd.DataFrame): Data to predict on.
    Returns:
        predictions (array-like): Predicted values.
    """
    return model.predict(input_data)

def main():
    # Load the model
    model_type = input("Enter model type (Random Forest or XGBoost): ").strip()
    model_filename = f"{model_type.lower()}_model.joblib"
    
    # Load the trained model
    model = load_model(model_filename)
    
    # Load and preprocess input data
    file_path = input("Enter the path to the input data for prediction: ")
    df = pd.read_csv(file_path)
    X, _, _ = preprocess_data(df)  # Preprocess data
    
    # Make predictions
    predictions = predict(model, X)
    
    # Output predictions
    print("Predictions: ", predictions)

if __name__ == "__main__":
    main()
