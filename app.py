
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

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


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

st.title("Breast Cancer Classification")
st.write(
    "Evaluation of five trained machine learning classifiers "
    "using uploaded test data."
)

st.write(
    "**Target mapping:** 0 = Benign | 1 = Malignant"
)


# --------------------------------------------------
# 2. MODEL FILE PATHS
# --------------------------------------------------

MODEL_DIR = Path("model")

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "K-Nearest Neighbors": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Support Vector Machine": MODEL_DIR / "support_vector_machine.joblib"
}

TARGET = "target"


# --------------------------------------------------
# 3. HELPER FUNCTION: LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# --------------------------------------------------
# 4. HELPER FUNCTION: GET EXPECTED FEATURES
# --------------------------------------------------

def get_expected_features(model):

    if (
        hasattr(model, "named_steps")
        and "preprocess" in model.named_steps
    ):
        return list(
            model.named_steps["preprocess"].feature_names_in_
        )



# --------------------------------------------------
# 5. HELPER FUNCTION: EVALUATE MODEL
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The model does not support probability predictions."
        )

    probabilities = model.predict_proba(X_test)

    if 1 not in model.classes_:
        raise ValueError(
            "The model does not contain class 1 (Malignant)."
        )

    positive_class_index = list(model.classes_).index(1)

    malignant_probabilities = probabilities[
        :, positive_class_index
    ]

    results = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(
            y_test,
            malignant_probabilities
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0
        ),
        "MCC": matthews_corrcoef(
            y_test,
            predictions
        )
    }

    return results, predictions


# --------------------------------------------------
# 6. UPLOAD TEST DATASET
# --------------------------------------------------

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test_data.csv file",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Please upload test_data.csv to evaluate the trained models."
    )
    st.stop()


# --------------------------------------------------
# 7. READ AND DISPLAY DATASET
# --------------------------------------------------

try:
    test_data = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(f"Unable to read the uploaded CSV: {error}")
    st.stop()


st.subheader("Uploaded Test Dataset")

st.write(f"Number of observations: {test_data.shape[0]}")
st.write(f"Number of columns: {test_data.shape[1]}")

st.dataframe(
    test_data.head(),
    use_container_width=True
)


# --------------------------------------------------
# 8. VALIDATE DATASET
# --------------------------------------------------

if TARGET not in test_data.columns:

    st.error(
        " The uploaded CSV must contain the target column named 'target'."
    )
    st.stop()


if test_data.empty:

    st.error(" The uploaded dataset is empty.")
    st.stop()


y_test = test_data[TARGET]

X_test = test_data.drop(columns=[TARGET])


if y_test.isnull().any():

    st.error("❌ The target column contains missing values.")
    st.stop()


if not set(y_test.unique()).issubset({0, 1}):

    st.error(
        " The target column must contain only 0 and 1."
    )
    st.stop()


if set(y_test.unique()) != {0, 1}:

    st.error(
        " The uploaded test dataset must contain both target classes (0 and 1) "
        "to calculate AUC."
    )
    st.stop()


y_test = y_test.astype(int)

st.success("✓ Dataset validation passed!")


# --------------------------------------------------
# 9. LOAD AND EVALUATE ALL FIVE MODELS
# --------------------------------------------------

st.header("Model Comparison")

st.info(
    "🔄 Loading 5 pre-trained models and evaluating on your test data... "
    "(No retraining occurs)"
)

comparison_results = {}
model_objects = {}
model_predictions = {}
model_errors = {}

progress_bar = st.progress(0)
status_text = st.empty()

for idx, (model_name, model_path) in enumerate(MODEL_FILES.items()):

    status_text.text(f"Evaluating {model_name}...")
    progress_bar.progress((idx + 1) / len(MODEL_FILES))

    try:

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        model = load_model(model_path)

        expected_features = get_expected_features(model)

        missing_features = [
            feature for feature in expected_features
            if feature not in X_test.columns
        ]

        extra_features = [
            feature for feature in X_test.columns
            if feature not in expected_features
        ]

        if missing_features:

            raise ValueError(
                f"Missing features: {', '.join(missing_features)}"
            )

        if extra_features:

            raise ValueError(
                f"Unexpected features: {', '.join(extra_features)}"
            )

        # Arrange columns in the order expected by the model
        X_model = X_test[expected_features]

        # Evaluate model
        metrics, predictions = evaluate_model(
            model,
            X_model,
            y_test
        )

        comparison_results[model_name] = metrics

        model_objects[model_name] = model

        model_predictions[model_name] = predictions

    except Exception as error:

        model_errors[model_name] = str(error)

progress_bar.empty()
status_text.empty()


# --------------------------------------------------
# 10. DISPLAY MODEL ERRORS
# --------------------------------------------------

if model_errors:

    st.warning(
        " Some models could not be evaluated. "
        "Please check the saved model files and input features."
    )

    for model_name, error in model_errors.items():

        st.error(f"**{model_name}:** {error}")


if not comparison_results:

    st.error("No models could be evaluated. Please check your test data and models.")
    st.stop()


# --------------------------------------------------
# 11. DISPLAY COMPARISON TABLE
# --------------------------------------------------

st.subheader(" Performance Comparison - All Models")

comparison_df = pd.DataFrame.from_dict(
    comparison_results,
    orient="index"
)

comparison_df.index.name = "Model"

comparison_df = comparison_df.round(4)

st.dataframe(
    comparison_df,
    use_container_width=True
)

# Add summary statistics
col1, col2, col3 = st.columns(3)

best_accuracy_model = comparison_df["Accuracy"].idxmax()
best_accuracy = comparison_df["Accuracy"].max()
col1.metric(" Best Accuracy", f"{best_accuracy:.4f}", f"{best_accuracy_model}")

best_auc_model = comparison_df["AUC"].idxmax()
best_auc = comparison_df["AUC"].max()
col2.metric(" Best AUC", f"{best_auc:.4f}", f"{best_auc_model}")

best_f1_model = comparison_df["F1 Score"].idxmax()
best_f1 = comparison_df["F1 Score"].max()
col3.metric("Best F1 Score", f"{best_f1:.4f}", f"{best_f1_model}")

st.write(f"**Total models evaluated:** {len(comparison_results)}/5")


# --------------------------------------------------
# 12. SELECT MODEL FOR DETAILED EVALUATION
# --------------------------------------------------

st.header("Detailed Model Evaluation")

st.write("Select a model from the dropdown to view detailed evaluation metrics, confusion matrix, and classification report.")

selected_model_name = st.selectbox(
    "Choose a model to analyze:",
    options=list(comparison_results.keys()),
    help="Select a pre-trained model to view its detailed performance on the test data"
)


# --------------------------------------------------
# 13. DISPLAY SIX EVALUATION METRICS
# --------------------------------------------------

selected_metrics = comparison_results[selected_model_name]

st.subheader(f" {selected_model_name} - Evaluation Metrics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Accuracy",
    f"{selected_metrics['Accuracy']:.4f}",
    help="Proportion of correct predictions"
)

col2.metric(
    "AUC",
    f"{selected_metrics['AUC']:.4f}",
    help="Area Under the ROC Curve"
)

col3.metric(
    "Precision",
    f"{selected_metrics['Precision']:.4f}",
    help="True positives / (True positives + False positives)"
)

col4, col5, col6 = st.columns(3)

col4.metric(
    "Recall",
    f"{selected_metrics['Recall']:.4f}",
    help="True positives / (True positives + False negatives)"
)

col5.metric(
    "F1 Score",
    f"{selected_metrics['F1 Score']:.4f}",
    help="Harmonic mean of Precision and Recall"
)

col6.metric(
    "MCC",
    f"{selected_metrics['MCC']:.4f}",
    help="Matthews Correlation Coefficient (-1 to 1)"
)


# --------------------------------------------------
# 14. CONFUSION MATRIX
# --------------------------------------------------

st.subheader("Confusion Matrix")

st.write(
    "The confusion matrix shows True Positives (TP), True Negatives (TN), "
    "False Positives (FP), and False Negatives (FN):"
)

selected_predictions = model_predictions[
    selected_model_name
]

cm = confusion_matrix(
    y_test,
    selected_predictions,
    labels=[0, 1]
)

confusion_df = pd.DataFrame(
    cm,
    index=["Actual Benign (0)", "Actual Malignant (1)"],
    columns=["Predicted Benign (0)", "Predicted Malignant (1)"]
)

st.dataframe(
    confusion_df,
    use_container_width=True
)

# Add confusion matrix interpretation
tn, fp, fn, tp = cm.ravel()
st.write(f"""
**Confusion Matrix Breakdown:**
- **True Negatives (TN):** {tn} - Correctly identified as Benign
- **False Positives (FP):** {fp} - Incorrectly identified as Malignant
- **False Negatives (FN):** {fn} - Incorrectly identified as Benign (⚠️ Critical misses)
- **True Positives (TP):** {tp} - Correctly identified as Malignant
""")


# --------------------------------------------------
# 15. CLASSIFICATION REPORT
# --------------------------------------------------

st.subheader("📋 Classification Report")

st.write(
    "Detailed per-class metrics including Precision, Recall, F1-Score, and Support:"
)

report = classification_report(
    y_test,
    selected_predictions,
    labels=[0, 1],
    target_names=["Benign", "Malignant"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)

# Add metrics explanation
st.write("""
**Metrics Explanation:**
- **Precision:** Among predicted positives, how many were actually positive?
- **Recall:** Among actual positives, how many were correctly identified?
- **F1-Score:** Harmonic mean of Precision and Recall (overall measure)
- **Support:** Number of samples for each class in the test set
""")