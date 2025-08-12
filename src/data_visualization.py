# data_visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plot a confusion matrix for the model predictions.
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        model_name (str): Name of the model for the plot title.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Scam', 'Scam'], yticklabels=['Not Scam', 'Scam'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def plot_feature_importance(model, X):
    """
    Plot feature importance for the given model.
    Args:
        model: Trained model (RandomForest or XGBoost).
        X (pd.DataFrame): Feature set.
    """
    importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.get_booster().get_score(importance_type='weight')
    feature_names = X.columns if hasattr(model, 'feature_importances_') else list(importance.keys())
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_names, y=importance)
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.show()
