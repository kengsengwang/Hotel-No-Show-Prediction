from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_and_evaluate(preprocessor, X_train, X_test, y_train, y_test):
    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(solver="liblinear", random_state=42))
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(clf, "outputs/logistic_model.pkl")
