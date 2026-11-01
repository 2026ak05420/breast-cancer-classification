
"""
BITS Pilani M.Tech AIML - Machine Learning Assignment 2

Project: Breast Cancer Wisconsin (Diagnostic) Classification

Five classifiers:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

Evaluation metrics:
Accuracy, AUC, Precision, Recall, F1, MCC

Positive class: Malignant (1)
Negative class: Benign (0)
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25
TARGET = "target"

DATASET_NAME = "Breast Cancer Wisconsin (Diagnostic)"
DATASET_SOURCE = (
    "https://archive.ics.uci.edu/dataset/17/"
    "breast+cancer+wisconsin+diagnostic"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset():

    print("\nLoading Breast Cancer Wisconsin dataset...")

    # Load the original UCI dataset through scikit-learn
    dataset = load_breast_cancer(as_frame=True)

    X = dataset.data.copy()

    # Scikit-learn's original labels:
    # 0 = malignant
    # 1 = benign
    #
    # Convert them so that:
    # 0 = benign
    # 1 = malignant
    y = (dataset.target == 0).astype(int)

    data = X.copy()
    data[TARGET] = y

    # Save a local copy of the dataset
    data.to_csv(
        ROOT / "breast_cancer.csv",
        index=False
    )

    print("Dataset shape:", data.shape)
    print("Number of observations:", len(data))
    print("Number of input features:", X.shape[1])

    print("\nTarget distribution:")
    print(
        y.map({
            0: "Benign",
            1: "Malignant"
        }).value_counts()
    )

    # Validate assignment requirements
    if len(data) < 500:
        raise ValueError(
            "Dataset must contain at least 500 observations."
        )

    if X.shape[1] < 12:
        raise ValueError(
            "Dataset must contain at least 12 input features."
        )

    if data.isnull().any().any():
        print("\nMissing values detected. They will be imputed.")

    # Check duplicates without silently removing records
    duplicate_count = int(data.duplicated().sum())

    print("\nExact duplicate rows:", duplicate_count)

    if duplicate_count > 0:
        raise ValueError(
            "Duplicate records detected. Investigate the dataset "
            "before proceeding with model evaluation."
        )

    print("\nDataset validation successful!")

    return X, y


# ============================================================
# 3. PREPROCESSING
# ============================================================

def create_preprocessor():

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


# ============================================================
# 4. DEFINE FIVE CLASSIFIERS
# ============================================================

def create_models():

    return {

        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=7
        ),

        "Naive Bayes": GaussianNB(),

        "Support Vector Machine": SVC(
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    }


# ============================================================
# 5. EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    # Probability of malignant class (class 1)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {

        "Accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "AUC": float(
            roc_auc_score(y_test, probabilities)
        ),

        "Precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "Recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "F1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "MCC": float(
            matthews_corrcoef(y_test, predictions)
        )
    }

    # Class order: 0 = Benign, 1 = Malignant
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0
    )

    return metrics, matrix, report


# ============================================================
# 6. MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 65)
    print("BITS PILANI - MACHINE LEARNING ASSIGNMENT 2")
    print("BREAST CANCER CLASSIFICATION")
    print("=" * 65)

    # Load dataset
    X, y = load_dataset()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    # Save test dataset with target column
    test_data = X_test.copy()
    test_data[TARGET] = y_test.values

    test_data.to_csv(
        ROOT / "test_data.csv",
        index=False
    )

    print("\nSaved test_data.csv")

    # Preprocessing
    preprocessor = create_preprocessor()

    # Models
    estimators = create_models()

    metrics_rows = []
    evaluation_details = {}

    # ========================================================
    # TRAIN AND EVALUATE FIVE MODELS
    # ========================================================

    for model_name, estimator in estimators.items():

        print("\n" + "-" * 65)
        print("Training:", model_name)
        print("-" * 65)

        pipeline = Pipeline([
            ("preprocess", create_preprocessor()),
            ("classifier", estimator)
        ])

        # Fit only on training data
        pipeline.fit(X_train, y_train)

        # Evaluate on unseen test data
        metrics, matrix, report = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        metrics_rows.append({
            "Model": model_name,
            **metrics
        })

        evaluation_details[model_name] = {
            "metrics": metrics,
            "confusion_matrix": matrix.tolist(),
            "classification_report": report
        }

        # Save model
        filename = (
            model_name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        model_path = MODEL_DIR / filename

        joblib.dump(
            pipeline,
            model_path
        )

        print("\nEvaluation results:")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        print("\nConfusion Matrix:")
        print(matrix)

        print("\nSaved model:", model_path)

    # ========================================================
    # 7. MODEL COMPARISON
    # ========================================================

    comparison = pd.DataFrame(metrics_rows)

    comparison = comparison.sort_values(
        by=["F1", "AUC"],
        ascending=False
    ).reset_index(drop=True)

    comparison.insert(
        0,
        "Rank",
        range(1, len(comparison) + 1)
    )

    comparison.to_csv(
        ROOT / "model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    best_model = comparison.iloc[0]["Model"]

    print("\nBest model based on F1 and AUC:", best_model)

    # ========================================================
    # 8. SAVE EVALUATION DETAILS
    # ========================================================

    with open(
        ROOT / "evaluation_details.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation_details,
            file,
            indent=4
        )

    # ========================================================
    # 9. SAVE DATASET METADATA
    # ========================================================

    metadata = {

        "dataset": DATASET_NAME,

        "dataset_source": DATASET_SOURCE,

        "total_rows": int(len(X)),

        "feature_count": int(X.shape[1]),

        "features": list(X.columns),

        "target": TARGET,

        "target_mapping": {
            "0": "Benign",
            "1": "Malignant"
        },

        "positive_class": "Malignant",

        "train_rows": int(len(X_train)),

        "test_rows": int(len(X_test)),

        "random_state": RANDOM_STATE,

        "test_size": TEST_SIZE,

        "best_model": best_model
    }

    with open(
        ROOT / "dataset_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # ========================================================
    # 10. COMPLETION
    # ========================================================

    print("\n" + "=" * 65)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("\nGenerated files:")
    print("- breast_cancer.csv")
    print("- test_data.csv")
    print("- model_comparison.csv")
    print("- evaluation_details.json")
    print("- dataset_metadata.json")

    print("\nSaved models:")

    for model_file in sorted(MODEL_DIR.glob("*.joblib")):
        print("-", model_file.name)

    print("\nBest model:", best_model)


if __name__ == "__main__":
    main()
