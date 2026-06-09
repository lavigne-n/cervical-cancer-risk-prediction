import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ======================================================
# Cervical Cancer Risk Prediction App
# Deployment-ready and compatible with improved training
# ======================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "cervical_cancer_model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"
FEATURES_PATH = BASE_DIR / "model_features.json"
COMPARISON_PATH = BASE_DIR / "model_comparison_results.csv"
IMPORTANCE_PATH = BASE_DIR / "feature_importance.csv"

DEFAULT_FEATURES = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes",
    "Hormonal Contraceptives",
    "STDs",
    "Dx:HPV",
]

NUMERIC_DEFAULTS = {
    "Age": 30,
    "Number of sexual partners": 1,
    "First sexual intercourse": 18,
    "Num of pregnancies": 1,
    "Smokes (years)": 0.0,
    "Smokes (packs/year)": 0.0,
    "Hormonal Contraceptives (years)": 0.0,
    "IUD (years)": 0.0,
    "STDs (number)": 0,
    "STDs: Number of diagnosis": 0,
}

YES_NO_FEATURES = {
    "Smokes",
    "Hormonal Contraceptives",
    "IUD",
    "STDs",
    "STDs:condylomatosis",
    "STDs:cervical condylomatosis",
    "STDs:vaginal condylomatosis",
    "STDs:vulvo-perineal condylomatosis",
    "STDs:syphilis",
    "STDs:pelvic inflammatory disease",
    "STDs:genital herpes",
    "STDs:molluscum contagiosum",
    "STDs:AIDS",
    "STDs:HIV",
    "STDs:Hepatitis B",
    "STDs:HPV",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "Dx",
}

BASIC_FEATURES = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes",
    "Hormonal Contraceptives",
    "STDs",
    "Dx:HPV",
]

# Page settings must come before any other Streamlit display command
st.set_page_config(
    page_title="Cervical Cancer Risk Prediction",
    page_icon="🩺",
    layout="wide",
)

STYLE = """
<style>
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

.hero-card {
    padding: 1.4rem 1.6rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #eaf4ff 0%, #ddebff 100%);
    border: 1px solid #c6d8ff;
    margin-bottom: 1rem;
}

.hero-title {
    color: #102A43 !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    color: #334E68 !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}

.risk-card {
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid #e6e6e6;
    background: #fafafa;
}

.small-muted {
    color: #666;
    font-size: 0.92rem;
}
</style>
"""

HERO = """
<div class="hero-card">
    <div class="hero-title">🩺 Cervical Cancer Risk Prediction System</div>
    <div class="hero-subtitle">
        A machine learning decision-support prototype for estimating cervical cancer risk
        using selected clinical, demographic, and behavioral factors.
    </div>
</div>
"""


def load_json(path, default=None):
    if not Path(path).exists():
        return default
    with open(path, "r") as file:
        return json.load(file)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_csv(path):
    if not Path(path).exists():
        return None
    return pd.read_csv(path)


def get_feature_list(metrics):
    features = load_json(FEATURES_PATH, default=None)
    if features:
        return features
    if metrics and "Features Used" in metrics:
        return metrics["Features Used"]
    return DEFAULT_FEATURES


def yes_no_input(label, key, default="No"):
    answer = st.selectbox(
        label,
        ["No", "Yes"],
        index=0 if default == "No" else 1,
        key=key,
    )
    return 1 if answer == "Yes" else 0


def numeric_input(feature, key):
    default = NUMERIC_DEFAULTS.get(feature, 0.0)

    if feature == "Age":
        return st.number_input("Age", min_value=10, max_value=100, value=int(default), step=1, key=key)
    if feature == "Number of sexual partners":
        return st.number_input("Number of sexual partners", min_value=0, max_value=50, value=int(default), step=1, key=key)
    if feature == "First sexual intercourse":
        return st.number_input("Age at first sexual intercourse", min_value=0, max_value=60, value=int(default), step=1, key=key)
    if feature == "Num of pregnancies":
        return st.number_input("Number of pregnancies", min_value=0, max_value=20, value=int(default), step=1, key=key)
    if feature == "Smokes (years)":
        return st.number_input("Smoking duration in years", min_value=0.0, max_value=80.0, value=float(default), step=0.5, key=key)
    if feature == "Smokes (packs/year)":
        return st.number_input("Smoking packs per year", min_value=0.0, max_value=100.0, value=float(default), step=0.1, key=key)
    if feature == "Hormonal Contraceptives (years)":
        return st.number_input("Hormonal contraceptive use in years", min_value=0.0, max_value=50.0, value=float(default), step=0.5, key=key)
    if feature == "IUD (years)":
        return st.number_input("IUD use in years", min_value=0.0, max_value=50.0, value=float(default), step=0.5, key=key)
    if feature in ["STDs (number)", "STDs: Number of diagnosis"]:
        return st.number_input(feature, min_value=0, max_value=30, value=int(default), step=1, key=key)

    return st.number_input(feature, value=float(default), step=0.1, key=key)


def feature_label(feature):
    labels = {
        "Smokes": "Does the patient smoke?",
        "Hormonal Contraceptives": "Uses hormonal contraceptives?",
        "IUD": "Uses IUD?",
        "STDs": "History of STDs?",
        "Dx:HPV": "HPV diagnosis?",
        "Dx:Cancer": "Previous cancer diagnosis?",
        "Dx:CIN": "CIN diagnosis?",
        "Dx": "Any previous diagnosis recorded?",
        "STDs:HIV": "HIV diagnosis?",
        "STDs:HPV": "STD-related HPV diagnosis?",
        "STDs:syphilis": "History of syphilis?",
        "STDs:Hepatitis B": "History of Hepatitis B?",
        "STDs:genital herpes": "History of genital herpes?",
        "STDs:condylomatosis": "History of condylomatosis?",
        "STDs:cervical condylomatosis": "History of cervical condylomatosis?",
        "STDs:vaginal condylomatosis": "History of vaginal condylomatosis?",
        "STDs:vulvo-perineal condylomatosis": "History of vulvo-perineal condylomatosis?",
        "STDs:pelvic inflammatory disease": "History of pelvic inflammatory disease?",
        "STDs:molluscum contagiosum": "History of molluscum contagiosum?",
        "STDs:AIDS": "AIDS diagnosis?",
    }
    return labels.get(feature, feature)


def build_input_form(features):
    values = {}

    st.subheader("Patient Information")
    st.caption("Enter the patient's known risk factors. Unknown advanced factors can be left at the default value.")

    with st.form("prediction_form"):
        basic_cols = st.columns(2)
        for index, feature in enumerate([f for f in BASIC_FEATURES if f in features]):
            with basic_cols[index % 2]:
                if feature in YES_NO_FEATURES:
                    values[feature] = yes_no_input(feature_label(feature), key=f"basic_{feature}")
                else:
                    values[feature] = numeric_input(feature, key=f"basic_{feature}")

        advanced_features = [f for f in features if f not in BASIC_FEATURES]
        if advanced_features:
            with st.expander("Additional Clinical Risk Factors", expanded=False):
                advanced_cols = st.columns(2)
                for index, feature in enumerate(advanced_features):
                    with advanced_cols[index % 2]:
                        if feature in YES_NO_FEATURES:
                            values[feature] = yes_no_input(feature_label(feature), key=f"advanced_{feature}")
                        else:
                            values[feature] = numeric_input(feature, key=f"advanced_{feature}")

        threshold = st.slider(
            "Risk threshold for classifying a patient as at-risk",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.05,
            help="Lower thresholds identify more possible high-risk patients but may increase false positives.",
        )

        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    for feature in features:
        values.setdefault(feature, 0)

    input_data = pd.DataFrame([[values[feature] for feature in features]], columns=features)
    return submitted, input_data, threshold


def get_probability(model, input_data):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(input_data)[0][1])

    prediction = int(model.predict(input_data)[0])
    return float(prediction)


def get_risk_details(probability, threshold):
    if probability < threshold:
        return (
            "Low Risk",
            "The patient is below the selected risk threshold. Regular cervical cancer screening and preventive healthcare are still encouraged.",
            "success",
        )

    if probability < 0.60:
        return (
            "Moderate Risk",
            "The patient is above the selected threshold and may require closer monitoring, routine screening, and review by a healthcare professional.",
            "warning",
        )

    return (
        "High Risk",
        "The patient has a high predicted risk and may require further cervical cancer screening, clinical follow-up, or consultation with a qualified healthcare professional.",
        "error",
    )


def show_alert(level, message):
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.error(message)


# =========================
# App layout
# =========================

st.markdown(STYLE, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)

model = load_model()
metrics = load_json(METRICS_PATH, default={})
features = get_feature_list(metrics)
comparison_df = load_csv(COMPARISON_PATH)
importance_df = load_csv(IMPORTANCE_PATH)

st.warning(
    "This system is for academic and decision-support purposes only. It must not be used as a final medical diagnosis."
)

if model is None:
    st.error(
        "Model file not found. Run `python train_model_improved.py` first so that `cervical_cancer_model.pkl` is created."
    )
    st.stop()

prediction_tab, evaluation_tab, about_tab = st.tabs([
    "🔍 Prediction",
    "📊 Model Evaluation",
    "ℹ️ About Project",
])

with prediction_tab:
    left_col, right_col = st.columns([1.3, 1])

    with left_col:
        submitted, input_data, threshold = build_input_form(features)

    with right_col:
        st.subheader("Input Summary")
        st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

        st.info(
            "The app uses the same feature list saved during model training, which helps prevent mismatches between training and prediction."
        )

    if submitted:
        probability = get_probability(model, input_data)
        risk_category, recommendation, level = get_risk_details(probability, threshold)

        st.divider()
        st.subheader("Prediction Result")

        result_cols = st.columns(3)
        result_cols[0].metric("Predicted Risk Probability", f"{probability:.2%}")
        result_cols[1].metric("Selected Threshold", f"{threshold:.0%}")
        result_cols[2].metric("Risk Category", risk_category)

        st.progress(min(max(probability, 0.0), 1.0))
        show_alert(level, f"Risk Category: {risk_category}")

        st.markdown("**Recommendation:**")
        st.write(recommendation)

        result_df = input_data.copy()
        result_df["Risk Probability"] = round(probability, 4)
        result_df["Selected Threshold"] = threshold
        result_df["Risk Category"] = risk_category
        result_df["Recommendation"] = recommendation

        st.download_button(
            label="Download Prediction Result",
            data=result_df.to_csv(index=False),
            file_name="cervical_cancer_prediction_result.csv",
            mime="text/csv",
            use_container_width=True,
        )

with evaluation_tab:
    st.header("Model Evaluation and Performance")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Best Model", metrics.get("Best Model", "Not available"))
    metric_cols[1].metric("Accuracy", metrics.get("Accuracy", "N/A"))
    metric_cols[2].metric("Precision", metrics.get("Precision", "N/A"))
    metric_cols[3].metric("Recall", metrics.get("Recall", "N/A"))
    metric_cols[4].metric("F1 Score", metrics.get("F1 Score", "N/A"))

    st.caption(
        "Recall is prioritized because the project focuses on reducing missed high-risk cases."
    )

    if comparison_df is not None:
        st.subheader("Model Comparison Results")
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.info("Model comparison file not found. Run the improved `train_model_improved.py` to generate it.")

    st.subheader("Feature Importance")
    if importance_df is not None and "Importance" in importance_df.columns:
        clean_importance = importance_df.dropna(subset=["Importance"])
        if not clean_importance.empty:
            st.bar_chart(clean_importance.set_index("Feature")["Importance"])
        else:
            st.info("Feature importance is not available for the selected model type.")
    else:
        st.info("Feature importance file not found. Run `python train_model_improved.py` again.")

    st.subheader("Confusion Matrix")
    conf_matrix = metrics.get("Confusion Matrix")
    if conf_matrix:
        confusion_df = pd.DataFrame(
            conf_matrix,
            index=["Actual Low Risk", "Actual High Risk"],
            columns=["Predicted Low Risk", "Predicted High Risk"],
        )
        st.dataframe(confusion_df, use_container_width=True)
    else:
        st.info("Confusion matrix is not available yet.")

    with st.expander("Features used by the current model"):
        st.write(features)

with about_tab:
    st.header("About the Project")

    st.write(
        """
        This project develops a machine learning-based cervical cancer risk prediction system using selected clinical,
        demographic, and behavioral risk factors. The aim is to support early identification of individuals who may
        require further screening or medical follow-up.
        """
    )

    st.subheader("Project Objective")
    st.write(
        """
        The main objective is to build a decision-support prototype that predicts cervical cancer risk and assists in
        screening prioritization. The system compares multiple machine learning models and selects the best model based
        mainly on recall and F1-score.
        """
    )

    st.subheader("Why the App Was Improved")
    st.write(
        """
        The earlier app used only a small number of input features and did not fully connect the threshold slider to the
        displayed risk category. This version loads the exact feature list used during training and applies the selected
        threshold directly during classification.
        """
    )

    st.subheader("Limitations")
    st.write(
        """
        1. The dataset is relatively small and may not fully represent all populations.  
        2. The system is trained on secondary publicly available data.  
        3. The prediction should not be treated as a medical diagnosis.  
        4. Missing values and class imbalance may affect model performance.  
        5. The model should be validated using larger and more diverse clinical datasets before real-world use.  
        """
    )

    st.info(
        "Final medical decisions should always be made by qualified healthcare professionals."
    )
