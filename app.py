
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 10px;
        margin-bottom: 30px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .section-header {
        color: #667eea;
        font-size: 24px;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 15px;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 style="color: #667eea; margin: 0;">🩺 Breast Cancer Classification</h1>', unsafe_allow_html=True)
with col2:
    st.markdown('<h3 style="color: #764ba2; text-align: right; margin-top: 10px;">ML Classifier Evaluation</h3>', unsafe_allow_html=True)

st.markdown('<hr style="border: 2px solid #667eea;">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🎯 Purpose:** Evaluate ML classifiers")
with col2:
    st.markdown("**📊 Models:** 5 trained classifiers")
with col3:
    st.markdown("**🎲 Classes:** Benign (0) | Malignant (1)")

st.markdown("")


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

st.markdown('<div class="section-header">📤 Upload Test Dataset</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload your test_data.csv file",
        type=["csv"],
        help="CSV file with 30 features and a 'target' column"
    )

with col2:
    st.info("💡 Tip: Use the included test_data.csv for testing")

if uploaded_file is None:
    st.warning("⚠️ Please upload test_data.csv to evaluate the trained models.")
    st.stop()


# --------------------------------------------------
# 7. READ AND DISPLAY DATASET
# --------------------------------------------------

try:
    test_data = pd.read_csv(uploaded_file)

except Exception as error:
    st.error(f"❌ Unable to read the uploaded CSV: {error}")
    st.stop()

# Display dataset info
st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Samples", f"{test_data.shape[0]}", help="Number of test observations")

with col2:
    st.metric("🔢 Features", f"{test_data.shape[1]}", help="Total columns including target")

with col3:
    st.metric("📈 Input Features", f"{test_data.shape[1]-1}", help="Predictor variables")

with col4:
    st.metric("🎯 Target Column", "Present ✓", help="Classification target")

st.write("")

with st.expander("👀 View First 5 Rows", expanded=False):
    st.dataframe(test_data.head(), use_container_width=True)


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

st.markdown('<div class="section-header">🚀 Model Evaluation in Progress</div>', unsafe_allow_html=True)

with st.spinner('⏳ Loading and evaluating 5 pre-trained models...'):
    comparison_results = {}
    model_objects = {}
    model_predictions = {}
    model_errors = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (model_name, model_path) in enumerate(MODEL_FILES.items()):

        status_text.text(f"⚙️ Evaluating {model_name}... ({idx+1}/5)")
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

st.success("✅ Model evaluation completed successfully!")


# --------------------------------------------------
# 10. DISPLAY MODEL ERRORS
# --------------------------------------------------

if model_errors:

    st.markdown('<div class="section-header">⚠️ Model Loading Errors</div>', unsafe_allow_html=True)

    for model_name, error in model_errors.items():
        st.error(f"🚨 **{model_name}:** {error}")

    st.warning("Please check your saved model files and input features.")


if not comparison_results:

    st.error("❌ No models could be evaluated. Please check your test data and model files.")
    st.stop()


# --------------------------------------------------
# 11. DISPLAY COMPARISON TABLE
# --------------------------------------------------

st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

comparison_df = pd.DataFrame.from_dict(
    comparison_results,
    orient="index"
)

comparison_df.index.name = "Model"

comparison_df = comparison_df.round(4)

st.dataframe(
    comparison_df.style.highlight_max(axis=0, color='#90EE90').format("{:.4f}"),
    use_container_width=True
)

st.write("")

# Best model highlights
best_accuracy_model = comparison_df["Accuracy"].idxmax()
best_accuracy = comparison_df["Accuracy"].max()

best_auc_model = comparison_df["AUC"].idxmax()
best_auc = comparison_df["AUC"].max()

best_f1_model = comparison_df["F1 Score"].idxmax()
best_f1 = comparison_df["F1 Score"].max()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 20px; border-radius: 10px; text-align: center;'>
        <h3>🏆 Best F1 Score</h3>
        <h2>{best_f1:.4f}</h2>
        <p>{best_f1_model}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                color: white; padding: 20px; border-radius: 10px; text-align: center;'>
        <h3>🎯 Best Accuracy</h3>
        <h2>{best_accuracy:.4f}</h2>
        <p>{best_accuracy_model}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                color: white; padding: 20px; border-radius: 10px; text-align: center;'>
        <h3>📈 Best AUC</h3>
        <h2>{best_auc:.4f}</h2>
        <p>{best_auc_model}</p>
    </div>
    """, unsafe_allow_html=True)

st.write(f"**Models Evaluated:** {len(comparison_results)}/5 ✓")


# --------------------------------------------------
# 12. SELECT MODEL FOR DETAILED EVALUATION
# --------------------------------------------------

st.markdown('<div class="section-header">🔍 Detailed Model Evaluation</div>', unsafe_allow_html=True)

st.write("Select a model to view detailed metrics, confusion matrix, and classification report.")

col1, col2 = st.columns([2, 1])

with col1:
    selected_model_name = st.selectbox(
        "📌 Choose a model to analyze:",
        options=list(comparison_results.keys()),
        help="Select a pre-trained model to view its detailed performance"
    )

with col2:
    st.write("")
    model_score = comparison_results[selected_model_name]['F1 Score']
    score_pct = int(model_score * 100)
    st.metric("Selected Model Score", f"{score_pct}%")

st.write("")

# --------------------------------------------------
# 13. DISPLAY SIX EVALUATION METRICS
# --------------------------------------------------

selected_metrics = comparison_results[selected_model_name]

st.markdown(f'<h3 style="color: #667eea;">📊 {selected_model_name} - Performance Metrics</h3>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🎯 Accuracy",
        f"{selected_metrics['Accuracy']:.4f}",
        help="Proportion of correct predictions"
    )

with col2:
    st.metric(
        "📈 AUC",
        f"{selected_metrics['AUC']:.4f}",
        help="Area Under the ROC Curve"
    )

with col3:
    st.metric(
        "✓ Precision",
        f"{selected_metrics['Precision']:.4f}",
        help="True positives / (True positives + False positives)"
    )

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🔍 Recall",
        f"{selected_metrics['Recall']:.4f}",
        help="True positives / (True positives + False negatives)"
    )

with col2:
    st.metric(
        "⚖️ F1 Score",
        f"{selected_metrics['F1 Score']:.4f}",
        help="Harmonic mean of precision and recall"
    )

with col3:
    st.metric(
        "📊 MCC",
        f"{selected_metrics['MCC']:.4f}",
        help="Matthews Correlation Coefficient"
    )

st.write("")

# --------------------------------------------------
# 14. CONFUSION MATRIX
# --------------------------------------------------

st.markdown('<div class="section-header">📐 Confusion Matrix</div>', unsafe_allow_html=True)

st.info(
    "📌 **True Positives (TP):** Correctly identified malignant cases  |  "
    "**False Negatives (FN):** Missed malignant cases (most critical)  |  "
    "**False Positives (FP):** Benign cases incorrectly flagged"
)

selected_predictions = model_predictions[selected_model_name]

cm = confusion_matrix(y_test, selected_predictions, labels=[0, 1])

confusion_df = pd.DataFrame(
    cm,
    index=["Actual Benign (0)", "Actual Malignant (1)"],
    columns=["Predicted Benign (0)", "Predicted Malignant (1)"]
)

st.dataframe(
    confusion_df.style.highlight_max(color='#90EE90'),
    use_container_width=True
)

# Display confusion matrix metrics
col1, col2, col3, col4 = st.columns(4)

tn, fp, fn, tp = cm.ravel()

with col1:
    st.markdown(f"""
    <div style='background: #E8F5E9; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50;'>
        <h4 style='color: #2E7D32; margin: 0;'>✓ True Negatives</h4>
        <h2 style='color: #1B5E20; margin: 10px 0 0 0;'>{tn}</h2>
        <p style='color: #558B2F; margin: 5px 0 0 0; font-size: 12px;'>Correctly identified benign</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='background: #FCE4EC; padding: 15px; border-radius: 8px; border-left: 4px solid #E91E63;'>
        <h4 style='color: #880E4F; margin: 0;'>⚠️ False Positives</h4>
        <h2 style='color: #C2185B; margin: 10px 0 0 0;'>{fp}</h2>
        <p style='color: #D81B60; margin: 5px 0 0 0; font-size: 12px;'>Benign flagged as malignant</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style='background: #FFEBEE; padding: 15px; border-radius: 8px; border-left: 4px solid #F44336;'>
        <h4 style='color: #C62828; margin: 0;'>🚨 False Negatives</h4>
        <h2 style='color: #B71C1C; margin: 10px 0 0 0;'>{fn}</h2>
        <p style='color: #D32F2F; margin: 5px 0 0 0; font-size: 12px;'>Malignant cases missed</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div style='background: #E3F2FD; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3;'>
        <h4 style='color: #1565C0; margin: 0;'>✅ True Positives</h4>
        <h2 style='color: #0D47A1; margin: 10px 0 0 0;'>{tp}</h2>
        <p style='color: #1976D2; margin: 5px 0 0 0; font-size: 12px;'>Correctly identified malignant</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

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

st.markdown('<div class="section-header">📋 Classification Report</div>', unsafe_allow_html=True)

st.info(
    "📊 **Per-class metrics:** Precision (accuracy of predictions), Recall (completeness of detection), "
    "F1-Score (balanced measure), Support (sample count)"
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
    report_df.round(4).style.highlight_max(color='#90EE90'),
    use_container_width=True
)

st.write("")

# Detailed interpretation
col1, col2 = st.columns(2)

with col1:
    benign_metrics = report["Benign"]
    st.markdown(f"""
    <div style='background: #E8F5E9; padding: 15px; border-radius: 8px;'>
        <h3 style='color: #2E7D32;'>🔵 Benign Class (0)</h3>
        <ul style='color: #1B5E20; line-height: 1.8;'>
            <li><b>Precision:</b> {benign_metrics['precision']:.4f} (of predicted benign, % correct)</li>
            <li><b>Recall:</b> {benign_metrics['recall']:.4f} (of actual benign, % detected)</li>
            <li><b>F1-Score:</b> {benign_metrics['f1-score']:.4f} (overall performance)</li>
            <li><b>Support:</b> {int(benign_metrics['support'])} (samples in test set)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    malignant_metrics = report["Malignant"]
    st.markdown(f"""
    <div style='background: #FFEBEE; padding: 15px; border-radius: 8px;'>
        <h3 style='color: #C62828;'>🔴 Malignant Class (1)</h3>
        <ul style='color: #B71C1C; line-height: 1.8;'>
            <li><b>Precision:</b> {malignant_metrics['precision']:.4f} (of predicted malignant, % correct)</li>
            <li><b>Recall:</b> {malignant_metrics['recall']:.4f} (of actual malignant, % detected)</li>
            <li><b>F1-Score:</b> {malignant_metrics['f1-score']:.4f} (overall performance)</li>
            <li><b>Support:</b> {int(malignant_metrics['support'])} (samples in test set)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)