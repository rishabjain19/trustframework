from __future__ import annotations
import os
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report
import shap

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
UCI_PATH = DATA_DIR / "student-mat.csv"
FALLBACK_PATH = DATA_DIR / "students.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"
EXPLANATIONS_PATH = OUTPUT_DIR / "shap_explanations.csv"
MODEL_PATH = OUTPUT_DIR / "model.pkl"

CONTRACT_COLUMNS = [
    "student_id",
    "risk_prob",
    "confidence",
    "top_factor_1",
    "top_factor_2",
    "top_factor_3",
]


def load_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    target_file = None
    if UCI_PATH.exists():
        target_file = UCI_PATH
    elif FALLBACK_PATH.exists():
        target_file = FALLBACK_PATH

    if target_file:
        with open(target_file, "r", encoding="utf-8") as f:
            first_line = f.readline()
            sep = ";" if ";" in first_line else ","
        df = pd.read_csv(target_file, sep=sep)
        print(f"Loaded dataset from: {target_file} (delimiter: '{sep}')")
    else:
        print("No student dataset found; generating representative demo cohort.")
        rng = np.random.default_rng(42)
        n = 395
        df = pd.DataFrame({
            "age": rng.integers(15, 22, size=n),
            "studytime": rng.choice([1, 2, 3, 4], size=n, p=[0.25, 0.50, 0.15, 0.10]),
            "failures": rng.choice([0, 1, 2, 3], size=n, p=[0.78, 0.13, 0.05, 0.04]),
            "absences": rng.poisson(5, size=n),
            "freetime": rng.integers(1, 6, size=n),
            "goout": rng.integers(1, 6, size=n),
            "Dalc": rng.integers(1, 6, size=n),
            "Walc": rng.integers(1, 6, size=n),
            "health": rng.integers(1, 6, size=n),
            "higher": rng.choice(["yes", "no"], size=n, p=[0.90, 0.10]),
            "G1": rng.integers(3, 19, size=n),
            "G2": rng.integers(3, 19, size=n),
            "G3": rng.integers(0, 20, size=n)
        })
        df.to_csv(FALLBACK_PATH, index=False)

    df.columns = df.columns.str.strip()
    df["at_risk"] = (df["G3"] < 10).astype(int)
    drop_cols = [c for c in ["G1", "G2", "G3", "at_risk"] if c in df.columns]
    X_raw = df.drop(columns=drop_cols)
    y = df["at_risk"]
    X = pd.get_dummies(X_raw, drop_first=True)
    return X, y, df


def train_and_explain():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, df_raw = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print("\n--- MODEL PERFORMANCE ---")
    print(f"Accuracy: {acc:.3f} | Recall: {rec:.3f} | ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    all_prob = model.predict_proba(X)[:, 1]
    confidence = np.maximum(all_prob, 1 - all_prob)

    explainer = shap.LinearExplainer(model, X)
    shap_vals = explainer.shap_values(X)
    feature_names = np.array(X.columns)

    top_1, top_2, top_3 = [], [], []
    for i in range(len(X)):
        ranked = np.argsort(-shap_vals[i])
        top_1.append(feature_names[ranked[0]])
        top_2.append(feature_names[ranked[1]])
        top_3.append(feature_names[ranked[2]])

    student_ids = [f"STU_{i+1:03d}" for i in range(len(X))]

    predictions_df = pd.DataFrame({
        "student_id": student_ids,
        "risk_prob": np.round(all_prob, 4),
        "confidence": np.round(confidence, 4),
        "top_factor_1": top_1,
        "top_factor_2": top_2,
        "top_factor_3": top_3,
    })

    predictions_df[CONTRACT_COLUMNS].to_csv(PREDICTIONS_PATH, index=False)
    print(f"Wrote {PREDICTIONS_PATH}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    shap_df.insert(0, "student_id", student_ids)
    shap_df.to_csv(EXPLANATIONS_PATH, index=False)
    print(f"Wrote {EXPLANATIONS_PATH}")

    metrics = {"accuracy": float(acc), "recall": float(rec), "auc": float(auc)}
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    train_and_explain()