
"""
BITS Pilani M.Tech AIML - Machine Learning Assignment 2

Streamlit application for Breast Cancer Classification.

Upload test_data.csv, select a saved model,
and view six evaluation metrics and detailed results.

Target:
0 = Benign
1 = Malignant
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

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


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

st.title("Breast Cancer Classification")
st.subheader("BITS Pilani - M.Tech AIML - Machine Learning Assignment 2")

st.markdown(
    """
    This application evaluates five trained machine learning
    classifiers using the Breast Cancer Wisconsin Diagnostic dataset.

    **Target classes:**
    - 0 = Benign
    - 1 = Malignant

    Upload a test CSV containing all 30 input features and
    the actual target column.
    """
)

st.info(
    "This application is for academic machine learning evaluation "
    "only. It is not a medical diagnostic tool."
)


# ============================================================
# 3. MODEL SELECTION
# ============================================================

selected_model = st.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys())
)


# ============================================================
# 4. UPLOAD TEST DATA
# ============================================================

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        # Read uploaded dataset
        test_data = pd.read_csv(uploaded_file)

        st.divider()
        st.subheader("Uploaded Dataset")

        col1, col2 = st.columns(2)

        col1.metric("Total Records", len(test_data))
        col2.metric("Total Columns", len(test_data.columns))

        st.dataframe(
            test_data.head(),
            use_container_width=True
        )

        # Validate target
        if TARGET not in test_data.columns:

            st.error(
                "Target column 'target' is missing. "
                "Upload a CSV containing the actual target values."
            )

            st.stop()

        # Separate features and target
        X_test = test_data.drop(columns=[TARGET])
        y_test = pd.to_numeric(
            test_data[TARGET],
            errors="coerce"
        )

        # Validate target values
        if y_test.isna().any():

            st.error("Target contains missing or invalid values.")
            st.stop()

        if not set(y_test.unique()).issubset({0, 1}):

            st.error(
                "Invalid target values. Expected 0 = Benign "
                "and 1 = Malignant."
            )

            st.stop()

        if len(y_test.unique()) < 2:

            st.error(
                "The uploaded test data must contain both classes "
                "to calculate AUC."
            )

            st.stop()

        y_test = y_test.astype(int)

        # ====================================================
        # 5. LOAD SELECTED MODEL
        # ====================================================

        model_path = MODEL_DIR / MODEL_FILES[selected_model]

        if not model_path.exists():

            st.error(
                f"Model file not found: {model_path}. "
                "Run train.py first."
            )

            st.stop()

        # Load model without retraining
        model = joblib.load(model_path)

        # Retrieve expected feature names from fitted pipeline
        expected_features = list(
            model.named_steps["preprocess"].feature_names_in_
        )

        # Check uploaded feature columns
        missing_features = [
            feature for feature in expected_features
            if feature not in X_test.columns
        ]

        extra_features = [
            feature for feature in X_test.columns
            if feature not in expected_features
        ]

        if missing_features or extra_features:

            st.error(
                f"Feature mismatch.\n\n"
                f"Missing features: {missing_features}\n\n"
                f"Unexpected features: {extra_features}"
            )

            st.stop()

        # Ensure correct feature order
        X_test = X_test[expected_features]

        # ====================================================
        # 6. PREDICTIONS
        # ====================================================

        predictions = model.predict(X_test)

        # Probability for malignant class (1)
        malignant_index = list(model.classes_).index(1)

        probabilities = model.predict_proba(X_test)[
            :, malignant_index
        ]

        # ====================================================
        # 7. CALCULATE SIX METRICS
        # ====================================================

        accuracy = accuracy_score(
            y_test, predictions
        )

        auc = roc_auc_score(
            y_test, probabilities
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test,
            predictions
        )

        # ====================================================
        # 8. DISPLAY EVALUATION METRICS
        # ====================================================

        st.divider()

        st.header(f"Evaluation Results: {selected_model}")

        st.caption(
            "Positive class: Malignant"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # ====================================================
        # 9. CONFUSION MATRIX
        # ====================================================

        st.divider()
        st.subheader("Confusion Matrix")

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual Benign", "Actual Malignant"],
            columns=["Predicted Benign", "Predicted Malignant"]
        )

        st.dataframe(
            matrix_df,
            use_container_width=True
        )

        # ====================================================
        # 10. CLASSIFICATION REPORT
        # ====================================================

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df,
            use_container_width=True
        )

        # ====================================================
        # 11. SAMPLE PREDICTIONS
        # ====================================================

        st.subheader("Sample Predictions")

        results = X_test.copy()

        results["Actual Diagnosis"] = y_test.map({
            0: "Benign",
            1: "Malignant"
        }).values

        results["Predicted Diagnosis"] = pd.Series(
            predictions
        ).map({
            0: "Benign",
            1: "Malignant"
        }).values

        results["Malignant Probability"] = probabilities

        st.dataframe(
            results.head(20),
            use_container_width=True
        )

        # Download evaluation results
        st.download_button(
            label="Download Predictions CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="breast_cancer_predictions.csv",
            mime="text/csv"
        )

        st.success(
            f"{selected_model} evaluation completed successfully!"
        )

    except Exception as error:

        st.error(
            f"An error occurred while processing the dataset: {error}"
        )

else:

    st.info(
        "Upload test_data.csv to start model evaluation."
    )

    st.write("Available trained models:")

    for model_name in MODEL_FILES:
        st.write(f"- {model_name}")
