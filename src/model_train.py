# model_train.py
from data_preparation import load_data, preprocess_data, split_data, prepare_pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib  # For saving models

def evaluate_model(y_true, y_pred, y_prob):
    """
    Evaluate a model based on precision, recall, F1 score, and ROC AUC.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    return precision, recall, f1, roc_auc

def save_model(model, filename):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, filename)

def main():
    # Corrected file path (ensure no extra spaces or typos)
    file_path = r'C:\Users\DELL\Documents\Hotel-No-Show-Prediction\data\cleaned_noshow_data.csv'

    # Load and preprocess data
    df = load_data(file_path)
    X, y, preprocessor = preprocess_data(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Prepare and train Random Forest model
    rf_pipeline = prepare_pipeline(preprocessor, RandomForestClassifier(random_state=42))
    rf_pipeline.fit(X_train, y_train)

    # Predict and evaluate Random Forest model
    rf_pred = rf_pipeline.predict(X_test)
    rf_prob = rf_pipeline.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_model(y_test, rf_pred, rf_prob)
    print(f"Random Forest - Precision: {rf_metrics[0]}, Recall: {rf_metrics[1]}, F1-Score: {rf_metrics[2]}, ROC AUC: {rf_metrics[3]}")

    # Save Random Forest model
    save_model(rf_pipeline, 'random_forest_model.joblib')

    # Prepare and train XGBoost model
    xgb_pipeline = prepare_pipeline(preprocessor, xgb.XGBClassifier(random_state=42))
    xgb_pipeline.fit(X_train, y_train)

    # Predict and evaluate XGBoost model
    xgb_pred = xgb_pipeline.predict(X_test)
    xgb_prob = xgb_pipeline.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate_model(y_test, xgb_pred, xgb_prob)
    print(f"XGBoost - Precision: {xgb_metrics[0]}, Recall: {xgb_metrics[1]}, F1-Score: {xgb_metrics[2]}, ROC AUC: {xgb_metrics[3]}")

    # Save XGBoost model
    save_model(xgb_pipeline, 'xgboost_model.joblib')

if __name__ == "__main__":
    main()
