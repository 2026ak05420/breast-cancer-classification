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
| Dataset URL | [UCI Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) |
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

- Training observations: 426
- Testing observations: 143

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

## 5. Evaluation Metrics and Model Comparison

The models are evaluated using the following six classification metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall classification correctness |
| AUC | Measures the ability to distinguish between classes |
| Precision | Measures the correctness of positive predictions |
| Recall | Measures how many actual positive cases are identified |
| F1-score | Harmonic mean of precision and recall |
| MCC | Measures binary classification quality using all confusion matrix values |

### Model Comparison Results

The following table presents the evaluation results obtained on the test dataset.

| ML Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9720 | 0.9950 | 0.9804 | 0.9434 | 0.9615 | 0.9400 |
| Decision Tree | 0.9231 | 0.9117 | 0.9200 | 0.8679 | 0.8932 | 0.8340 |
| KNN | 0.9720 | 0.9870 | 1.0000 | 0.9245 | 0.9608 | 0.9408 |
| Naive Bayes | 0.9441 | 0.9925 | 0.9592 | 0.8868 | 0.9216 | 0.8798 |
| Support Vector Machine | **0.9930** | **0.9969** | **1.0000** | **0.9811** | **0.9905** | **0.9851** |

### Observations on Model Performance

**1. Logistic Regression**

Logistic Regression achieved an accuracy of 97.20% and an AUC of 0.9950. It demonstrated strong classification performance with high precision and recall.

**2. Decision Tree Classifier**

The Decision Tree achieved an accuracy of 92.31%. Compared with the other models, it recorded lower AUC, recall, F1-score, and MCC values on this test dataset.

**3. K-Nearest Neighbors (KNN)**

KNN achieved an accuracy of 97.20% and a precision of 1.0000. It correctly identified all predicted malignant cases, although its recall was lower than that of SVM.

**4. Gaussian Naive Bayes**

Naive Bayes achieved an accuracy of 94.41% and an AUC of 0.9925. It demonstrated good classification performance, although its recall and F1-score were lower than those of Logistic Regression, KNN, and SVM.

**5. Support Vector Machine (SVM)**

SVM achieved the highest overall performance among the five models. It recorded 99.30% accuracy, 99.69% AUC, 100% precision, and 98.11% recall. It also achieved the highest F1-score and MCC.

### Overall Winner

**Support Vector Machine (SVM)**

SVM is the best-performing model on the selected test split based on its overall evaluation metrics. Its high precision and recall indicate strong performance in identifying the two target classes.

This result applies to the current train-test split and does not establish clinical suitability or guarantee performance on unseen external datasets.

---

## 6. Streamlit Application

An interactive web application is developed using Streamlit.

### Application Features

- Upload a CSV test dataset.
- Select one of the five trained machine learning models.
- Validate uploaded input features.
- Generate predictions using saved models.
- Display all six evaluation metrics.
- Compare the performance of all five models on the uploaded test dataset.
- Display a confusion matrix.
- Display a classification report.
- View sample predictions.
- Download prediction results.

The application uses previously trained models and evaluates uploaded test data without retraining the models.

The uploaded test CSV must contain all required input features and the actual target column to calculate evaluation metrics.

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

## 8. Project Links

**GitHub Repository**

[Breast Cancer Classification Repository](https://github.com/2026ak05420/breast-cancer-classification)

**Live Streamlit Application**

[Open the deployed application](ADD_YOUR_STREAMLIT_APP_URL_HERE)

---

## 9. Conclusion

This project implements and evaluates five supervised machine learning classification algorithms using the Breast Cancer Wisconsin (Diagnostic) dataset.

The models were evaluated using Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient.

Among the five algorithms, Support Vector Machine achieved the best overall results on the selected test dataset.

The project also demonstrates the use of saved machine learning pipelines in an interactive Streamlit application, allowing users to upload test data, generate predictions, and evaluate model performance.