"""
Train five classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset.
Run: python train_models.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Dataset: UCI Breast Cancer Wisconsin (Diagnostic), distributed with scikit-learn.
dataset = load_breast_cancer(as_frame=True)
frame = dataset.frame.rename(columns={"target": "diagnosis"})
X = frame.drop(columns=["diagnosis"])
y = frame["diagnosis"].astype(int)  # 0 = malignant, 1 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Export the held-out test dataset for Streamlit evaluation
test_data = X_test.copy()
test_data["diagnosis"] = y_test.values

test_data.to_csv(
    ROOT / "test_data.csv",
    index=False
)

print("Test dataset exported successfully.")
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
    ])
}

results = []
for display_name, estimator in models.items():
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)

    # All five selected estimators expose predict_proba; positive class is 1 (benign).
    probabilities = estimator.predict_proba(X_test)[:, 1]
    row = {
        "Model": display_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions)
    }
    results.append(row)
    safe_name = display_name.lower().replace(" ", "_").replace("-", "")
    joblib.dump(estimator, MODEL_DIR / f"{safe_name}.joblib")
    print(f"Saved {display_name}")

metrics = pd.DataFrame(results).sort_values("F1", ascending=False)
metrics.to_csv(ROOT / "model_metrics.csv", index=False)
with open(ROOT / "model_metrics.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print("\nEvaluation metrics on the held-out 20% test set:")
print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print("\nTest data file: test_data.csv")
print("Saved model files in: model/")
