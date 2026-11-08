# Machine Learning Assignment 2 — Classification Model Comparison

## a. Problem statement

The objective is to build and compare five supervised classification algorithms on a binary classification dataset. The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

A Streamlit interface allows evaluators to upload labelled test data, select a model, and inspect its performance metrics, confusion matrix, classification report, and uploaded-data preview.

## b. Dataset description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Scikit-learn reference:** `sklearn.datasets.load_breast_cancer`
- **Instances:** 569
- **Input features:** 30 numeric measurements
- **Target:** `diagnosis`
- **Problem type:** Binary classification

Target encoding:

- `0` = Malignant
- `1` = Benign

The project uses an 80:20 stratified train/test split with `random_state=42`.

Feature scaling is applied inside pipelines for Logistic Regression, KNN, and SVM. This ensures that preprocessing parameters are learned from training data only, preventing test-data leakage.

## c. GitHub repository link

**Public GitHub Repository:**

https://github.com/Sonu19011/ML

**Live Streamlit Application:**

https://bq3vov4spwpcdb58bdwhkv.streamlit.app/

## d. Models used

The following five supervised classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Evaluation metrics

The project evaluates each model using:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model performance comparison

The following results are taken from the project's recorded `model_metrics.csv` and `model_metrics.json` files. Values are rounded to four decimal places.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.9825** | **0.9954** | 0.9861 | 0.9861 | 0.9861 | **0.9623** |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | **0.9825** | 0.9950 | 0.9861 | 0.9861 | 0.9861 | **0.9623** |

**Verification note:** Logistic Regression, Decision Tree, Gaussian Naive Bayes, and SVM were manually tested through the deployed Streamlit application. KNN's results above are from the saved training evaluation metrics; it was not separately re-tested manually in the live application during final verification.

### Observations about model performance

**1. Logistic Regression**

Logistic Regression achieved 98.25% accuracy, 0.9954 AUC, 0.9861 F1 score, and 0.9623 MCC. It tied with SVM for accuracy and MCC and achieved the highest AUC. Feature scaling is important for this model because its optimization is sensitive to feature magnitude.

**2. Decision Tree**

Decision Tree achieved 92.11% accuracy and 0.9163 AUC, the lowest values among the five models in this experiment. Decision Trees do not require feature standardization. Further validation would be required to determine whether the lower test performance reflects underfitting, overfitting, or model configuration.

**3. K-Nearest Neighbors**

KNN achieved 95.61% accuracy, 0.9788 AUC, 0.9655 F1 score, and 0.9054 MCC. Since KNN uses distances between observations, feature scaling is important to prevent large-scale features from dominating distance calculations.

**4. Gaussian Naive Bayes**

Gaussian Naive Bayes achieved 93.86% accuracy and 0.9878 AUC. Its AUC is high despite lower accuracy than Logistic Regression, SVM, and KNN. Correlations among diagnostic features may challenge its conditional-independence assumption.

**5. Support Vector Machine**

SVM achieved 98.25% accuracy, 0.9950 AUC, 0.9861 F1 score, and 0.9623 MCC. It performed very similarly to Logistic Regression. Feature standardization is important for SVM because feature scales influence margin and kernel calculations.

### Overall winner

**Logistic Regression** is the overall winner based on the highest AUC (0.9954), tied-highest accuracy (0.9825), and tied-highest MCC (0.9623).

SVM is a very close alternative, with nearly identical performance.

For the application's binary metrics, class `1` (benign) is treated as the positive class. The classification report separately presents results for malignant and benign cases.

## e. How to run the project

Clone the repository:

```bash
git clone https://github.com/Sonu19011/ML.git
cd ML
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

To regenerate the trained models and evaluation metrics:

```bash
python train_models.py
```

The training script saves the trained model files in the `model/` directory and generates `model_metrics.csv` and `model_metrics.json`.

## f. Repository structure

```text
ML/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── breast_cancer_wisconsin.csv
├── test_data.csv
├── model_metrics.csv
├── model_metrics.json
└── model/
    ├── decision_tree.joblib
    ├── gaussian_naive_bayes.joblib
    ├── knearest_neighbors.joblib
    ├── logistic_regression.joblib
    └── support_vector_machine.joblib
```

## g. Deployment

The application is deployed using Streamlit Community Cloud.

- **GitHub:** https://github.com/Sonu19011/ML
- **Live application:** https://bq3vov4spwpcdb58bdwhkv.streamlit.app/

The deployed application accepts a labelled CSV containing the 30 expected feature columns and a `diagnosis` target column.

## h. Conclusion

Five supervised classification algorithms were implemented and evaluated using the Breast Cancer Wisconsin (Diagnostic) dataset.

Logistic Regression achieved the strongest overall recorded performance, with 98.25% accuracy, 0.9954 AUC, and 0.9623 MCC. SVM achieved closely matching results.

The Streamlit application provides an interactive interface for uploading labelled test data and examining model evaluation metrics.

**Academic-use note:** This project is intended for educational purposes only. It is not a medical diagnostic or clinical decision-support tool. The reported metrics should not be interpreted as evidence of clinical performance.
