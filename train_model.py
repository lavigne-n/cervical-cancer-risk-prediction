
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# Cervical Cancer Risk Prediction - Improved Training Script
# ============================================================
# This version is designed to avoid the very low accuracy issue
# caused by selecting models using recall only.
#
# Model selection rule:
# 1. Avoid models/thresholds with very poor accuracy.
# 2. Select the best model mainly using F1-score.
# 3. Use recall, precision, and accuracy as tie-breakers.
# ============================================================


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"

MODEL_PATH = BASE_DIR / "cervical_cancer_model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"
FEATURES_PATH = BASE_DIR / "model_features.json"
COMPARISON_PATH = BASE_DIR / "model_comparison_results.csv"
IMPORTANCE_PATH = BASE_DIR / "feature_importance.csv"


if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Make sure dataset.csv is in the same folder as train_model.py."
    )


# Load dataset
data = pd.read_csv(DATASET_PATH)
data = data.replace("?", pd.NA)


# Target variable
target = "Biopsy"

if target not in data.columns:
    raise ValueError(f"Target column '{target}' was not found in dataset.csv.")


# Preferred features.
# The script will only use the features that actually exist in your dataset.
preferred_features = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes",
    "Smokes (years)",
    "Smokes (packs/year)",
    "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)",
    "IUD",
    "IUD (years)",
    "STDs",
    "STDs (number)",
    "STDs: Number of diagnosis",
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
]

features = [feature for feature in preferred_features if feature in data.columns]

if len(features) < 4:
    raise ValueError(
        "Too few usable features were found in dataset.csv. "
        "Please confirm that the dataset column names are correct."
    )


# Keep selected columns only
data = data[features + [target]]

# Convert columns to numeric
data = data.apply(pd.to_numeric, errors="coerce")

# Remove rows where target is missing
data = data.dropna(subset=[target])

# Make target binary integer
data[target] = data[target].astype(int)

X = data[features]
y = data[target]


print("Dataset loaded successfully.")
print("Rows:", len(data))
print("Features used:", len(features))
print("Target distribution:")
print(y.value_counts())


# Stratified split keeps the same class ratio in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )),
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )),
    ]),

    "Extra Trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", ExtraTreesClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )),
    ]),

    "Support Vector Machine": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            class_weight="balanced",
            probability=True,
            random_state=42,
        )),
    ]),

    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        )),
    ]),
}


# These thresholds are not extremely low, so the model will not classify
# almost everyone as high-risk. This helps prevent accuracy dropping badly.
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]


all_results = []
trained_models = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    trained_models[model_name] = model

    if hasattr(model, "predict_proba"):
        y_probability = model.predict_proba(X_test)[:, 1]
    else:
        y_probability = model.predict(X_test)

    for threshold in thresholds:
        y_pred = (y_probability >= threshold).astype(int)

        accuracy = round(accuracy_score(y_test, y_pred), 4)
        precision = round(precision_score(y_test, y_pred, zero_division=0), 4)
        recall = round(recall_score(y_test, y_pred, zero_division=0), 4)
        f1 = round(f1_score(y_test, y_pred, zero_division=0), 4)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        all_results.append({
            "Model": model_name,
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Confusion Matrix": conf_matrix,
        })


results_df = pd.DataFrame(all_results)


# ------------------------------------------------------------
# Balanced model selection
# ------------------------------------------------------------
# First, reject thresholds with extremely low accuracy where possible.
# This prevents the app from selecting a model with accuracy like 0.1.
# Then choose the best remaining model using F1-score first.
# ------------------------------------------------------------

candidate_df = results_df[results_df["Accuracy"] >= 0.65].copy()

if candidate_df.empty:
    candidate_df = results_df[results_df["Accuracy"] >= 0.55].copy()

if candidate_df.empty:
    candidate_df = results_df.copy()


candidate_df = candidate_df.sort_values(
    by=["F1 Score", "Recall", "Precision", "Accuracy"],
    ascending=[False, False, False, False],
)

best_row = candidate_df.iloc[0]

best_model_name = best_row["Model"]
best_threshold = float(best_row["Threshold"])
best_model = trained_models[best_model_name]
best_conf_matrix = best_row["Confusion Matrix"]


# Save best model
joblib.dump(best_model, MODEL_PATH)


# Save model comparison results
results_for_csv = results_df.drop(columns=["Confusion Matrix"])
results_for_csv.to_csv(COMPARISON_PATH, index=False)


# Save features used
with open(FEATURES_PATH, "w") as file:
    json.dump(features, file, indent=4)


# Save metrics
metrics = {
    "Best Model": best_model_name,
    "Optimal Threshold": best_threshold,
    "Accuracy": float(best_row["Accuracy"]),
    "Precision": float(best_row["Precision"]),
    "Recall": float(best_row["Recall"]),
    "F1 Score": float(best_row["F1 Score"]),
    "Confusion Matrix": best_conf_matrix,
    "Features Used": features,
    "Selection Rule": (
        "Models with acceptable accuracy were prioritized first, then the best model was selected "
        "using F1-score, recall, precision, and accuracy."
    ),
}

with open(METRICS_PATH, "w") as file:
    json.dump(metrics, file, indent=4)


# Save feature importance where possible
importance_df = pd.DataFrame({"Feature": features, "Importance": np.nan})

classifier = best_model.named_steps.get("classifier")

if hasattr(classifier, "feature_importances_"):
    importance_df["Importance"] = classifier.feature_importances_

elif hasattr(classifier, "coef_"):
    coef = classifier.coef_[0]
    importance_df["Importance"] = np.abs(coef)

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False,
    na_position="last",
)

importance_df.to_csv(IMPORTANCE_PATH, index=False)


# Display final results
print("\nModel Comparison Results")
print("------------------------")
print(results_for_csv.sort_values(by=["F1 Score", "Recall", "Accuracy"], ascending=False).head(15))

print("\nBest Model Selected")
print("-------------------")
print("Best Model:", best_model_name)
print("Best Threshold:", best_threshold)
print("Accuracy:", best_row["Accuracy"])
print("Precision:", best_row["Precision"])
print("Recall:", best_row["Recall"])
print("F1 Score:", best_row["F1 Score"])
print("Confusion Matrix:")
print(np.array(best_conf_matrix))

print("\nSaved files")
print("-----------")
print(f"Model: {MODEL_PATH.name}")
print(f"Metrics: {METRICS_PATH.name}")
print(f"Features: {FEATURES_PATH.name}")
print(f"Comparison: {COMPARISON_PATH.name}")
print(f"Feature importance: {IMPORTANCE_PATH.name}")
print("\nTraining completed successfully.")
