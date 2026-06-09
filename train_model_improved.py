import json
import joblib
import pandas as pd


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ======================================================
# Cervical Cancer Risk Prediction - Improved Training
# ======================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
TARGET = "Biopsy"
RANDOM_STATE = 42

# Load dataset
data = pd.read_csv(DATASET_PATH)
data = data.replace("?", pd.NA)

# Stronger feature list. The script will only use columns that exist in your dataset.
candidate_features = [
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
    "STDs: Number of diagnosis",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "Dx"
]

features = [col for col in candidate_features if col in data.columns]

if TARGET not in data.columns:
    raise ValueError(f"Target column '{TARGET}' was not found in the dataset.")

if len(features) == 0:
    raise ValueError("None of the selected feature columns were found in the dataset.")

# Keep selected columns only
data = data[features + [TARGET]]
data = data.apply(pd.to_numeric, errors="coerce")

# Drop rows where target is missing
data = data.dropna(subset=[TARGET])

X = data[features]
y = data[TARGET].astype(int)

# Train/test split with stratification to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# Define candidate models
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "Extra Trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "Support Vector Machine": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ))
    ])
}

results = []
best_model = None
best_model_name = None
best_score = -1
best_conf_matrix = None

for model_name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred, zero_division=0), 4)
    recall = round(recall_score(y_test, y_pred, zero_division=0), 4)
    f1 = round(f1_score(y_test, y_pred, zero_division=0), 4)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Healthcare priority: recall first, then F1 score.
    # This reduces the chance of missing actual high-risk patients.
    selection_score = (recall * 0.7) + (f1 * 0.3)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Selection Score": round(selection_score, 4)
    })

    if selection_score > best_score:
        best_score = selection_score
        best_model = model
        best_model_name = model_name
        best_conf_matrix = conf_matrix

results_df = pd.DataFrame(results).sort_values(by="Selection Score", ascending=False)

print("\nModel Comparison Results")
print("------------------------")
print(results_df)
print("\nBest Model Selected:", best_model_name)

# Save best model and supporting files
joblib.dump(best_model, "cervical_cancer_model.pkl")
results_df.to_csv("model_comparison_results.csv", index=False)

best_metrics = results_df[results_df["Model"] == best_model_name].iloc[0].to_dict()

metrics = {
    "Best Model": best_model_name,
    "Accuracy": best_metrics["Accuracy"],
    "Precision": best_metrics["Precision"],
    "Recall": best_metrics["Recall"],
    "F1 Score": best_metrics["F1 Score"],
    "Selection Score": best_metrics["Selection Score"],
    "Confusion Matrix": best_conf_matrix.tolist(),
    "Features Used": features
}

with open("model_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

with open("model_features.json", "w") as file:
    json.dump(features, file, indent=4)

# Save feature importance where supported
classifier = best_model.named_steps["classifier"]
if hasattr(classifier, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": classifier.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    importance_df.to_csv("feature_importance.csv", index=False)
else:
    pd.DataFrame({"Feature": features, "Importance": [None] * len(features)}).to_csv(
        "feature_importance.csv", index=False
    )

print("\nTraining completed successfully.")
print("Saved: cervical_cancer_model.pkl")
print("Saved: model_metrics.json")
print("Saved: model_comparison_results.csv")
print("Saved: model_features.json")
print("Saved: feature_importance.csv")
