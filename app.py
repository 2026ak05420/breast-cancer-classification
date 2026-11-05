
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


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="",
    layout="wide"
)


# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 {
        color: #1e293b;
        font-size: 30px !important;
        font-weight: 650 !important;
    }

    h2, h3 {
        color: #26364a;
        font-weight: 600 !important;
    }

    p, label {
        color: #475569;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed #cbd5e1;
        border-radius: 8px;
        padding: 12px;
        background: #fafbfc;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }

    div[data-testid="stSelectbox"] {
        max-width: 500px;
    }

    hr {
        border-color: #e2e8f0;
    }

    .footer {
        color: #64748b;
        text-align: center;
        font-size: 13px;
        padding-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. PAGE HEADER
# ==================================================

st.title("Breast Cancer Classification")

st.caption(
    "Machine Learning Model Evaluation | "
    "BITS Pilani M.Tech AIML"
)

st.write(
    "This application evaluates five trained machine learning "
    "classifiers using an uploaded test dataset."
)

st.caption("Target mapping: 0 = Benign | 1 = Malignant")

st.divider()


# ==================================================
# 4. MODEL FILE PATHS
# ==================================================

MODEL_DIR = Path("model")

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "K-Nearest Neighbors": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Support Vector Machine": MODEL_DIR / "support_vector_machine.joblib"
}

TARGET = "target"


# ==================================================
# 5. LOAD SAVED MODEL
# ==================================================

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# ==================================================
# 6. GET EXPECTED INPUT FEATURES
# ==================================================

def get_expected_features(model):

    if (
        hasattr(model, "named_steps")
        and "preprocess" in model.named_steps
    ):
        return list(
            model.named_steps["preprocess"].feature_names_in_
        )

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    raise ValueError(
        "Unable to identify the expected input features."
    )


# ==================================================
# 7. MODEL EVALUATION FUNCTION
# ==================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The model does not support probability predictions."
        )

    probabilities = model.predict_proba(X_test)

    if 1 not in model.classes_:
        raise ValueError(
            "The model does not contain the malignant class (1)."
        )

    positive_class_index = list(model.classes_).index(1)

    malignant_probabilities = probabilities[
        :, positive_class_index
    ]

    metrics = {
        "Accuracy": accuracy_score(
            y_test, predictions
        ),

        "AUC": roc_auc_score(
            y_test, malignant_probabilities
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

        "F1-score": f1_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_test, predictions
        )
    }

    return metrics, predictions


# ==================================================
# 8. UPLOAD TEST DATASET
# ==================================================

st.header("Upload Test Dataset")

st.write(
    "Upload the test CSV file containing all 30 input features "
    "and the actual target column."
)

uploaded_file = st.file_uploader(
    "Choose test_data.csv",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Upload your test dataset to start model evaluation."
    )

    st.stop()


# ==================================================
# 9. READ TEST DATASET
# ==================================================

try:

    test_data = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(f"Unable to read the uploaded CSV: {error}")
    st.stop()


if test_data.empty:

    st.error("The uploaded dataset is empty.")
    st.stop()


# ==================================================
# 10. DATASET OVERVIEW
# ==================================================

st.divider()

st.header("Dataset Overview")

col1, col2 = st.columns(2)

col1.metric(
    "Total Observations",
    test_data.shape[0]
)

col2.metric(
    "Total Columns",
    test_data.shape[1]
)

with st.expander("Preview uploaded dataset", expanded=True):

    st.dataframe(
        test_data.head(),
        use_container_width=True
    )


# ==================================================
# 11. VALIDATE TEST DATA
# ==================================================

if TARGET not in test_data.columns:

    st.error(
        "The uploaded CSV must contain a column named 'target'."
    )

    st.stop()


if test_data.columns.duplicated().any():

    st.error("The uploaded CSV contains duplicate column names.")
    st.stop()


if test_data[TARGET].isnull().any():

    st.error("The target column contains missing values.")
    st.stop()


y_test = test_data[TARGET]

X_test = test_data.drop(columns=[TARGET])


if not set(y_test.unique()).issubset({0, 1}):

    st.error(
        "The target column must contain only 0 and 1."
    )

    st.stop()


if set(y_test.unique()) != {0, 1}:

    st.error(
        "Both target classes must be present to calculate AUC."
    )

    st.stop()


y_test = y_test.astype(int)


# ==================================================
# 12. LOAD AND EVALUATE ALL FIVE MODELS
# ==================================================

comparison_results = {}
model_objects = {}
model_predictions = {}
model_errors = {}

for model_name, model_path in MODEL_FILES.items():

    try:

        if not model_path.exists():

            raise FileNotFoundError(
                f"Saved model not found: {model_path}"
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
                f"Missing input features: {missing_features}"
            )

        if extra_features:

            raise ValueError(
                f"Unexpected input features: {extra_features}"
            )

        X_model = X_test[expected_features]

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


# ==================================================
# 13. HANDLE MODEL ERRORS
# ==================================================

if model_errors:

    st.error(
        "Some models could not be evaluated. "
        "Please check the model files and dataset."
    )

    for model_name, error in model_errors.items():

        st.write(f"**{model_name}:** {error}")


if not comparison_results:

    st.error("No models could be evaluated.")
    st.stop()


# ==================================================
# 14. MODEL COMPARISON
# ==================================================

st.divider()

st.header("Model Performance Comparison")

st.write(
    "The following table compares the performance of all "
    "available trained models on the uploaded test dataset."
)

comparison_df = pd.DataFrame.from_dict(
    comparison_results,
    orient="index"
)

comparison_df.index.name = "ML Model"

comparison_df = comparison_df.round(4)

st.dataframe(
    comparison_df.style.format("{:.4f}"),
    use_container_width=True
)


# ==================================================
# 15. SELECT MODEL
# ==================================================

st.divider()

st.header("Individual Model Evaluation")

selected_model_name = st.selectbox(
    "Select a machine learning model",
    options=list(comparison_results.keys())
)

selected_metrics = comparison_results[
    selected_model_name
]


# ==================================================
# 16. DISPLAY EVALUATION METRICS
# ==================================================

st.subheader("Evaluation Metrics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Accuracy",
    f"{selected_metrics['Accuracy']:.4f}"
)

col2.metric(
    "AUC",
    f"{selected_metrics['AUC']:.4f}"
)

col3.metric(
    "Precision",
    f"{selected_metrics['Precision']:.4f}"
)

col4, col5, col6 = st.columns(3)

col4.metric(
    "Recall",
    f"{selected_metrics['Recall']:.4f}"
)

col5.metric(
    "F1-score",
    f"{selected_metrics['F1-score']:.4f}"
)

col6.metric(
    "MCC",
    f"{selected_metrics['MCC']:.4f}"
)


# ==================================================
# 17. CONFUSION MATRIX
# ==================================================

st.divider()

st.subheader("Confusion Matrix")

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


# ==================================================
# 18. CLASSIFICATION REPORT
# ==================================================

st.subheader("Classification Report")

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


# ==================================================
# 19. FOOTER
# ==================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        BITS Pilani | M.Tech Artificial Intelligence and Machine Learning<br>
        Machine Learning Assignment 2
    </div>
    """,
    unsafe_allow_html=True
)
