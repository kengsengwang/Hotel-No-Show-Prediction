from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

# Load data
data = pd.read_csv('cleaned_noshow_data.csv')

# Define features and target
X = data.drop(columns=['no_show', 'booking_id'])  # Features
y = data['no_show']  # Target (1 = no-show, 0 = showed up)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing: Numeric vs. Categorical
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Pipeline setup
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__class_weight': [None, 'balanced']  # Handles class imbalance
}

# Setting up GridSearchCV
rf_grid = GridSearchCV(
    rf_pipe,
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# Fitting the model
print("Training Random Forest with GridSearchCV...")
rf_grid.fit(X_train, y_train)

# Best model after hyperparameter tuning
best_rf = rf_grid.best_estimator_
print(f"\nBest parameters found: {rf_grid.best_params_}")

# Evaluate model
def evaluate_model(model, X_test, y_test):
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class
    
    # Calculate metrics
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "classification_report": report,
        "roc_auc_score": roc_auc,
        "confusion_matrix": cm,
        "feature_importances": model.named_steps['classifier'].feature_importances_
    }

# Get model performance metrics
print("\nEvaluating model performance...")
rf_metrics = evaluate_model(best_rf, X_test, y_test)

# Print Random Forest Performance
print("\nRandom Forest Performance:")
print(rf_metrics["classification_report"])
print(f"ROC AUC Score: {rf_metrics['roc_auc_score']:.4f}")
print("Confusion Matrix:")
print(rf_metrics["confusion_matrix"])

# Feature Importance (if needed)
feature_names = (numeric_features.tolist() + 
                 list(best_rf.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .get_feature_names_out(categorical_features)))

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_metrics['feature_importances']
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance_df.head(10))