from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="ML Classifier Comparison", page_icon="🧪", layout="wide")
st.title("Machine Learning Classification Model Evaluator")
st.caption("Breast Cancer Wisconsin (Diagnostic) • Five-model comparison")

st.info(
    "Upload a labelled test CSV containing the 30 feature columns and a "
    "'diagnosis' target column. Labels: 0 = malignant, 1 = benign."
)

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
model_name = st.selectbox("Choose a classification model", list(MODEL_FILES))

if uploaded is not None:
    try:
        test_df = pd.read_csv(uploaded)
        if "diagnosis" not in test_df.columns:
            st.error("The uploaded CSV must contain a 'diagnosis' column.")
            st.stop()

        y_true = test_df["diagnosis"].astype(int)
        X_test = test_df.drop(columns=["diagnosis"])

        model_path = MODEL_DIR / MODEL_FILES[model_name]
        if not model_path.exists():
            st.error("Model files are missing. Run `python train_models.py` first.")
            st.stop()

        model = joblib.load(model_path)
        expected_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
        if expected_features is not None:
            missing = sorted(set(expected_features) - set(X_test.columns))
            extra = sorted(set(X_test.columns) - set(expected_features))
            if missing or extra:
                st.error(f"Feature mismatch. Missing: {missing}; unexpected: {extra}")
                st.stop()
            X_test = X_test[expected_features]

        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        st.subheader(f"Results: {model_name}")
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_score),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }
        cols = st.columns(3)
        for index, (name, value) in enumerate(metrics.items()):
            cols[index % 3].metric(name, f"{value:.4f}")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Confusion matrix")
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            st.dataframe(
                pd.DataFrame(cm,
                    index=["Actual malignant (0)", "Actual benign (1)"],
                    columns=["Predicted malignant (0)", "Predicted benign (1)"])
            )
        with right:
            st.markdown("#### Classification report")
            st.code(classification_report(
                y_true, y_pred, target_names=["malignant", "benign"], zero_division=0
            ))

        with st.expander("Preview uploaded test data"):
            st.dataframe(test_df.head(10), use_container_width=True)
    except Exception as error:
        st.error(f"Could not evaluate this file: {error}")
else:
    st.warning("Upload test_data.csv from the project folder to display evaluation results.")
