import streamlit as st
import pandas as pd
import joblib
import numpy as np
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

# Page config
st.set_page_config(
    page_title="ML Model Evaluation",
    layout="wide"
)

# Title
st.title("Breast Cancer Classification - Model Evaluation")
st.write("Upload test data and evaluate multiple ML classification models")

# Model paths
MODEL_DIR = Path("model")
MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "K-Nearest Neighbors": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Support Vector Machine": MODEL_DIR / "support_vector_machine.joblib"
}
TARGET = "target"

# Helper functions
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

def get_expected_features(model):
    return model.named_steps['preprocessor'].feature_names_in_.tolist()

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    mcc = matthews_corrcoef(y_test, predictions)
    
    metrics = {
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MCC': mcc
    }
    
    return metrics, predictions

# Upload file
st.header("1. Upload Test Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is None:
    st.info("Please upload a CSV file to continue")
    st.stop()

# Load and validate data
try:
    test_data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.write(f"Dataset shape: {test_data.shape[0]} rows, {test_data.shape[1]} columns")

# Validate data
if TARGET not in test_data.columns:
    st.error(f"Target column '{TARGET}' not found in dataset")
    st.stop()

if test_data.empty:
    st.error("Dataset is empty")
    st.stop()

y_test = test_data[TARGET]
X_test = test_data.drop(columns=[TARGET])

if y_test.isnull().any():
    st.error("Target column contains missing values")
    st.stop()

if not set(y_test.unique()).issubset({0, 1}):
    st.error("Target values must be 0 or 1")
    st.stop()

if set(y_test.unique()) != {0, 1}:
    st.error("Dataset must contain both classes (0 and 1)")
    st.stop()

y_test = y_test.astype(int)
st.success("Data validation passed")

# Evaluate all models
st.header("2. Model Evaluation")
st.write("Evaluating models on test data...")

comparison_results = {}
model_objects = {}
model_predictions = {}

progress_bar = st.progress(0)
status = st.empty()

for idx, (model_name, model_path) in enumerate(MODEL_FILES.items()):
    status.text(f"Evaluating {model_name}...")
    progress_bar.progress((idx + 1) / len(MODEL_FILES))
    
    try:
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            continue
        
        model = load_model(model_path)
        expected_features = get_expected_features(model)
        
        missing = [f for f in expected_features if f not in X_test.columns]
        extra = [f for f in X_test.columns if f not in expected_features]
        
        if missing or extra:
            st.error(f"{model_name}: Feature mismatch")
            continue
        
        X_model = X_test[expected_features]
        metrics, predictions = evaluate_model(model, X_model, y_test)
        
        comparison_results[model_name] = metrics
        model_objects[model_name] = model
        model_predictions[model_name] = predictions
        
    except Exception as e:
        st.error(f"Error with {model_name}: {str(e)}")

progress_bar.empty()
status.empty()

if not comparison_results:
    st.error("No models could be evaluated")
    st.stop()

# Display results table
st.header("3. Results Comparison")
results_df = pd.DataFrame(comparison_results).T
results_df = results_df.round(4)
st.dataframe(results_df)

# Model selection for detailed view
st.header("4. Detailed Analysis")
selected_model = st.selectbox("Select a model to analyze:", list(comparison_results.keys()))

if selected_model:
    col1, col2, col3 = st.columns(3)
    
    metrics = comparison_results[selected_model]
    
    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    col2.metric("AUC", f"{metrics['AUC']:.4f}")
    col3.metric("Precision", f"{metrics['Precision']:.4f}")
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Recall", f"{metrics['Recall']:.4f}")
    col2.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    col3.metric("MCC", f"{metrics['MCC']:.4f}")
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    predictions = model_predictions[selected_model]
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    
    cm_df = pd.DataFrame(
        cm,
        index=['Actual Benign', 'Actual Malignant'],
        columns=['Predicted Benign', 'Predicted Malignant']
    )
    st.dataframe(cm_df)
    
    # Classification report
    st.subheader("Classification Report")
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=['Benign', 'Malignant'],
        output_dict=True,
        zero_division=0
    )
    
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4))
