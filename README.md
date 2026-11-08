# Machine Learning Assignment 2 — Classification Model Comparison

## a. Problem statement
The objective is to build and compare five supervised classification algorithms on one binary classification dataset. The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). A Streamlit interface allows an evaluator to upload labelled test data, select a model, and inspect its metrics and confusion matrix.

## b. Dataset description
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic  
**Dataset reference:** The dataset is also available through `sklearn.datasets.load_breast_cancer`.  
**Instances:** 569  
**Input features:** 30 numeric measurements computed from digitized images of fine needle aspirates of breast masses.  
**Target:** `diagnosis` — 0 = malignant, 1 = benign.  
**Problem type:** Binary classification.

The project uses an 80:20 stratified train/test split with `random_state=42`. Scaling is applied inside pipelines for Logistic Regression, KNN, and SVM to prevent preprocessing leakage from the test set.

## c. GitHub Repository Link
**Add your public GitHub repository URL here after creating and pushing your repository.**

## d. Models used

The following five models are implemented on the same training split:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Evaluation metrics
Run `python train_models.py`. The program writes the actual results to `model_metrics.csv` and `model_metrics.json`. Copy the values from your own run into this table; do not invent or reuse another student's results.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Fill from model_metrics.csv | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Gaussian Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

### Observations about model performance
Complete this section after running the experiments. Discuss the actual metrics, not assumptions.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Compare its Accuracy, AUC, F1 and MCC with the other models. Mention whether scaling helped and whether the model appears to separate the two classes well. |
| Decision Tree | Discuss its test performance and whether its metrics suggest a possible bias/variance trade-off. |
| KNN | Discuss the effect of distance-based classification and feature scaling on its test metrics. |
| Gaussian Naive Bayes | Discuss its results in light of the conditional-independence assumption. |
| Support Vector Machine | Discuss its AUC and classification metrics; note that features were standardized. |
| Overall winner | Select the model supported by your chosen priority metric(s), and justify the choice using the measured results. |

## Step-by-step execution

### 1. Set up the environment
Use the BITS Virtual Lab as required by the assignment. Open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
```

### 2. Train and evaluate all models

```bash
python train_models.py
```

This creates five saved model files in `model/`, plus `model_metrics.csv` and `model_metrics.json`.

### 3. Run the Streamlit application locally

```bash
streamlit run app.py
```

Open the local URL printed in the terminal. Upload `test_data.csv`, choose a model, and review the metrics, confusion matrix, and classification report.

### 4. Create the GitHub repository
Create a new repository under your own GitHub account. Add `app.py`, `train_models.py`, `requirements.txt`, `README.md`, `test_data.csv`, `breast_cancer_wisconsin.csv`, `model_metrics.csv`, `model_metrics.json`, and all five files in `model/`. Commit your own work with meaningful messages. Add the repository URL to this README.

### 5. Deploy on Streamlit Community Cloud
1. Push the completed project to GitHub.
2. Visit https://streamlit.io/cloud and sign in with GitHub.
3. Select **New app**, choose your repository and `main` branch.
4. Set the main file path to `app.py`.
5. Deploy and wait for the app to finish building.
6. Open the public app URL and test it by uploading `test_data.csv`.

### 6. Prepare the final submission PDF
Maintain the order specified in the assignment:
1. GitHub repository link.
2. Live Streamlit app link.
3. One screenshot showing execution on BITS Virtual Lab.
4. The completed README content, including the dataset description, metrics comparison table, model observations, and overall winner.

## Important notes
- The assignment says AI tools may be used for learning support, not direct copy-paste submissions. Read, run, understand, and customize the code; write observations based on your own measured results.
- The BITS Virtual Lab execution screenshot must be captured by you.
- Verify that your GitHub and deployed Streamlit links are accessible before submitting.
- Submission deadline in the assignment: 8 November 2026, 23:59 PM.
