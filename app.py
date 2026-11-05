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
# 2. HELPER FUNCTIONS
# ============================================================

def get_expected_features(model):

    if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
        preprocess = model.named_steps["preprocess"]

        if hasattr(preprocess, "feature_names_in_"):
            return list(preprocess.feature_names_in_)

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    raise ValueError("Could not determine expected feature names.")


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support probability predictions.")

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Malignant class 1 not found in model.")

    malignant_index = classes.index(1)

    probabilities = model.predict_proba(X_test)[:, malignant_index]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),

        "AUC": roc_auc_score(
            y_test, probabilities
        ),

        "Precision": precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),

        "Recall": recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),

        "F1 Score": f1_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_test, predictions
        )
    }

    return predictions, metrics


# ============================================================
# 3. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide",
    page_icon="🩺"
)

st.title("Breast Cancer Classification")

st.subheader("Machine Learning Assignment 2")

st.markdown(
    """
    **Target classes:**

    - 0 = Benign
    - 1 = Malignant

    Upload the test dataset to evaluate the five trained
    classification models.
    """
)


# ============================================================
# 4. MODEL SELECTION
# ============================================================

selected_model = st.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys())
)


# ============================================================
# 5. UPLOAD TEST DATA
# ============================================================

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


if uploaded_file is None:

    st.info("Upload test_data.csv to start model evaluation.")

    st.write("Available trained models:")

    for model_name in MODEL_FILES:
        st.write(f"- {model_name}")

    st.stop()


# ============================================================
# 6. READ AND VALIDATE TEST DATA
# ============================================================

try:

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

    if TARGET not in test_data.columns:

        st.error(
            "Target column 'target' is missing."
        )

        st.stop()

    X_test = test_data.drop(columns=[TARGET])

    y_test = pd.to_numeric(
        test_data[TARGET],
        errors="coerce"
    )

    if y_test.isna().any():

        st.error("Target contains missing or invalid values.")

        st.stop()

    if not set(y_test.unique()).issubset({0, 1}):

        st.error("Expected target values: 0 = Benign, 1 = Malignant.")

        st.stop()

    if len(y_test.unique()) < 2:

        st.error(
            "Test dataset must contain both classes to calculate AUC."
        )

        st.stop()

    y_test = y_test.astype(int)


    # ========================================================
    # 7. LOAD ALL FIVE SAVED MODELS
    # ========================================================

    loaded_models = {}
    model_errors = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = MODEL_DIR / filename

        if not model_path.exists():

            model_errors[model_name] = "Model file not found."

            continue

        try:

            loaded_models[model_name] = joblib.load(model_path)

        except Exception as error:

            model_errors[model_name] = str(error)


    if not loaded_models:

        st.error("No saved models could be loaded.")

        st.stop()


    # ========================================================
    # 8. VALIDATE INPUT FEATURES
    # ========================================================

    expected_features = get_expected_features(
        next(iter(loaded_models.values()))
    )

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

    X_test = X_test[expected_features]


    # ========================================================
    # 9. EVALUATE ALL FIVE MODELS
    # ========================================================

    st.divider()

    st.header("Comparison of All Five Models")

    st.caption(
        "All models are evaluated using the uploaded test dataset. "
        "Saved models are loaded without retraining."
    )

    comparison_rows = []
    evaluation_outputs = {}

    for model_name, model in loaded_models.items():

        try:

            predictions, metrics = evaluate_model(
                model,
                X_test,
                y_test
            )

            comparison_rows.append({
                "Model": model_name,
                **metrics
            })

            evaluation_outputs[model_name] = predictions

        except Exception as error:

            model_errors[model_name] = str(error)


    # ========================================================
    # 10. DISPLAY MODEL COMPARISON
    # ========================================================

    if comparison_rows:

        comparison_df = pd.DataFrame(comparison_rows)

        st.dataframe(
            comparison_df.style.format({
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "MCC": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.error("No models could be evaluated.")

        st.stop()


    if model_errors:

        with st.expander("Model errors"):

            for model_name, error in model_errors.items():

                st.warning(f"{model_name}: {error}")


    # ========================================================
    # 11. SELECTED MODEL EVALUATION
    # ========================================================

    st.divider()

    st.header(f"Detailed Evaluation: {selected_model}")

    if selected_model not in evaluation_outputs:

        st.error(f"{selected_model} could not be evaluated.")

        st.stop()

    predictions = evaluation_outputs[selected_model]

    selected_metrics = comparison_df[
        comparison_df["Model"] == selected_model
    ].iloc[0]


    # ========================================================
    # 12. DISPLAY SIX EVALUATION METRICS
    # ========================================================

    st.caption("Positive class: Malignant (1)")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{selected_metrics['Accuracy']:.4f}")
    col2.metric("AUC", f"{selected_metrics['AUC']:.4f}")
    col3.metric("Precision", f"{selected_metrics['Precision']:.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{selected_metrics['Recall']:.4f}")
    col5.metric("F1 Score", f"{selected_metrics['F1 Score']:.4f}")
    col6.metric("MCC", f"{selected_metrics['MCC']:.4f}")


    # ========================================================
    # 13. CONFUSION MATRIX
    # ========================================================

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


    # ========================================================
    # 14. CLASSIFICATION REPORT
    # ========================================================

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

    st.success(
        f"{selected_model} evaluation completed successfully!"
    )


except Exception as error:

    st.error(
        f"An error occurred while processing the dataset: {error}"
    )