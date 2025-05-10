# model_evaluation.py
import joblib
from data_preparation import load_data, preprocess_data, split_data
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(y_true, y_pred, y_prob):
    """
    Evaluate the model's performance.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    return precision, recall, f1, roc_auc

def load_model(filename):
    """
    Load a saved model from file.
    """
    return joblib.load(filename)

def main():
    # Corrected file path (ensure no extra spaces or typos)
    file_path = r'C:\Users\DELL\Documents\Hotel-No-Show-Prediction\data\cleaned_noshow_data.csv'

    # Load and preprocess data
    df = load_data(file_path)
    X, y, preprocessor = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Load the trained models
    rf_model = load_model('random_forest_model.joblib')
    xgb_model = load_model('xgboost_model.joblib')

    # Predict and evaluate Random Forest model
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_model(y_test, rf_pred, rf_prob)
    print(f"Random Forest - Precision: {rf_metrics[0]}, Recall: {rf_metrics[1]}, F1-Score: {rf_metrics[2]}, ROC AUC: {rf_metrics[3]}")

    # Predict and evaluate XGBoost model
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate_model(y_test, xgb_pred, xgb_prob)
    print(f"XGBoost - Precision: {xgb_metrics[0]}, Recall: {xgb_metrics[1]}, F1-Score: {xgb_metrics[2]}, ROC AUC: {xgb_metrics[3]}")

if __name__ == "__main__":
    main()
