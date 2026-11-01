# Breast Cancer Classification — BITS Pilani M.Tech AIML Assignment 2

## 1. Project Overview

This project implements and compares five supervised machine learning classification algorithms to predict whether a breast tumour is Benign or Malignant.

The project covers dataset exploration, preprocessing, model training, performance evaluation, and development of an interactive Streamlit application.

This project is developed as part of the BITS Pilani M.Tech Artificial Intelligence and Machine Learning programme.

**Note:** This is an academic machine learning project and is not intended for clinical diagnosis or medical decision-making.

---

## 2. Problem Statement

The objective is to develop and compare five supervised machine learning classification models using the Breast Cancer Wisconsin (Diagnostic) dataset.

The project aims to:

- Load and validate a public classification dataset.
- Perform appropriate data preprocessing.
- Train five supervised machine learning algorithms.
- Evaluate the models using six classification metrics.
- Compare the performance of the trained models.
- Develop an interactive Streamlit application for model evaluation.

---

## 3. Dataset Description

| Attribute | Description |
|---|---|
| Dataset Name | Breast Cancer Wisconsin (Diagnostic) |
| Dataset Source | UCI Machine Learning Repository |
| Dataset ID | 17 |
| Dataset URL | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| Problem Type | Binary Classification |
| Total Observations | 569 |
| Input Features | 30 |
| Target Classes | Benign and Malignant |

### Target Variable

The target variable is mapped as follows:

- `0` = Benign
- `1` = Malignant

### Dataset Validation

The dataset is validated for:

- Minimum 500 observations.
- Minimum 12 input features.
- Duplicate records.
- Target class distribution.

### Data Preprocessing

The following preprocessing techniques are implemented:

1. Median imputation for missing numerical values.
2. Feature standardization using `StandardScaler`.
3. Integration of preprocessing and classification within machine learning pipelines.
4. Fitting preprocessing transformations on training data only to prevent data leakage.

### Train-Test Split

The dataset is divided using a stratified 75:25 train-test split with `random_state=42`.

The generated `test_data.csv` contains the input features and the actual target label.

---

## 4. Machine Learning Models

The following five supervised classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

All five models use the same training and testing split.

The trained models are saved using Joblib for reuse in the Streamlit application.

---

## 5. Evaluation Metrics

The models are evaluated using the following six classification metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall classification correctness |
| AUC | Measures the ability to distinguish between classes |
| Precision | Measures the correctness of positive predictions |
| Recall | Measures how many actual positive cases are identified |
| F1-score | Harmonic mean of precision and recall |
| MCC | Measures binary classification quality using all confusion matrix values |

The comparative evaluation results will be documented after completing the model evaluation stage.

---

## 6. Streamlit Application

An interactive web application is developed using Streamlit.

### Application Features

- Upload a CSV test dataset.
- Select one of the five trained machine learning models.
- Validate uploaded input features.
- Generate predictions using saved models.
- Display all six evaluation metrics.
- Display a confusion matrix.
- Display a classification report.
- View sample predictions.
- Download prediction results.

The application uses previously trained models and evaluates uploaded test data without retraining.

### Run the Application

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
python -m streamlit run app.py
```

---

## 7. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning algorithms and evaluation |
| Joblib | Model persistence |
| Streamlit | Interactive web application |
| Git | Version control |
| GitHub | Source code management |

---

## 8. Repository Structure

```text
breast_cacer_assignment/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── START_HERE.txt
│
├── breast_cancer.csv
├── test_data.csv
├── model_comparison.csv
├── evaluation_details.json
├── dataset_metadata.json
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```
