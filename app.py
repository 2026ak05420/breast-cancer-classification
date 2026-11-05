
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

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="centered"
)

# Project title
st.title("Breast Cancer Classification")


st.write(
    "This application evaluates five machine learning models "
    "using an uploaded test dataset."
)

st.write("Target: 0 - Benign | 1 - Malignant")

st.divider()


# Model paths
MODEL_DIR = Path("model")

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "KNN": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "SVM": MODEL_DIR / "support_vector_machine.joblib"
}

TARGET = "target"


# Load saved model
@st.cache_resource
def load_model(path):
    return joblib.load(path)


# Get model input features
def get_features(model):

    if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
        return list(model.named_steps["preprocess"].feature_names_in_)

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    raise ValueError("Could not identify model features.")


# Evaluate model
def evaluate_model(model, X, y):

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    positive_index = list(model.classes_).index(1)
    malignant_probabilities = probabilities[:, positive_index]

    results = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, malignant_probabilities),
        "Precision": precision_score(y, predictions, pos_label=1, zero_division=0),
        "Recall": recall_score(y, predictions, pos_label=1, zero_division=0),
        "F1-score": f1_score(y, predictions, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions)
    }

    return results, predictions


# Upload test dataset
st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload the test dataset to continue.")
    st.stop()


# Read dataset
try:
    data = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Error reading CSV file: {error}")
    st.stop()


# Dataset information
# Dataset information
st.subheader("Dataset Information")

st.write("Number of rows:", data.shape[0])
st.write("Number of columns:", data.shape[1])

st.dataframe(data.head(), use_container_width=True)

with st.expander("View dataset"):
    st.dataframe(data.head(), use_container_width=True)


# Validate dataset
if TARGET not in data.columns:
    st.error("The CSV file must contain a target column.")
    st.stop()

if data.columns.duplicated().any():
    st.error("Duplicate column names found in the dataset.")
    st.stop()

if data[TARGET].isnull().any():
    st.error("Missing values found in the target column.")
    st.stop()

y_test = data[TARGET]

if not set(y_test.unique()).issubset({0, 1}):
    st.error("Target values must be 0 or 1.")
    st.stop()

if set(y_test.unique()) != {0, 1}:
    st.error("Both target classes must be present for AUC calculation.")
    st.stop()

y_test = y_test.astype(int)
X_test = data.drop(columns=[TARGET])


# Evaluate all models
st.divider()
st.header("Model Comparison")

comparison = {}
predictions_dict = {}
models = {}
errors = {}

for name, path in MODEL_FILES.items():

    try:
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model = load_model(path)
        expected_features = get_features(model)

        missing = [f for f in expected_features if f not in X_test.columns]
        extra = [f for f in X_test.columns if f not in expected_features]

        if missing:
            raise ValueError(f"Missing features: {missing}")

        if extra:
            raise ValueError(f"Unexpected features: {extra}")

        X_model = X_test[expected_features]

        results, predictions = evaluate_model(
            model, X_model, y_test
        )

        comparison[name] = results
        predictions_dict[name] = predictions
        models[name] = model

    except Exception as error:
        errors[name] = str(error)


if errors:
    st.warning("Some models could not be evaluated.")

    for name, error in errors.items():
        st.write(f"{name}: {error}")


if not comparison:
    st.error("No models could be evaluated.")
    st.stop()


# Display comparison table
comparison_df = pd.DataFrame.from_dict(
    comparison,
    orient="index"
)

comparison_df.index.name = "Model"

st.dataframe(
    comparison_df.round(4),
    use_container_width=True
)


# Select model
st.divider()
st.header("Individual Model Evaluation")

selected_model = st.selectbox(
    "Select Model",
    list(comparison.keys())
)

metrics = comparison[selected_model]
predictions = predictions_dict[selected_model]


# Display metrics
st.subheader("Evaluation Metrics")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    st.metric("Precision", f"{metrics['Precision']:.4f}")
    st.metric("F1-score", f"{metrics['F1-score']:.4f}")

with col2:
    st.metric("AUC", f"{metrics['AUC']:.4f}")
    st.metric("Recall", f"{metrics['Recall']:.4f}")
    st.metric("MCC", f"{metrics['MCC']:.4f}")


# Confusion matrix
st.divider()
st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Benign", "Actual Malignant"],
    columns=["Predicted Benign", "Predicted Malignant"]
)

st.dataframe(cm_df, use_container_width=True)


# Classification report
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
    report_df.round(4),
    use_container_width=True
)


