"""Train the student-risk model and publish the Person A prediction contract."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT: Final = Path(__file__).resolve().parent
DATA_DIR: Final = ROOT / "data"
OUTPUT_DIR: Final = ROOT / "outputs"
DATA_PATH: Final = DATA_DIR / "students.csv"
PREDICTIONS_PATH: Final = OUTPUT_DIR / "predictions.csv"
EXPLANATIONS_PATH: Final = OUTPUT_DIR / "shap_explanations.csv"
MODEL_PATH: Final = OUTPUT_DIR / "model.pkl"
FEATURES: Final = ["attendance_rate", "assignment_score", "prior_gpa", "support_requests"]
CONTRACT_COLUMNS: Final = [
    "student_id",
    "risk_prob",
    "confidence",
    "top_factor_1",
    "top_factor_2",
    "top_factor_3",
]


def _demo_data() -> pd.DataFrame:
    """Create reproducible data for development before real data is available."""
    rng = np.random.default_rng(42)
    rows = 120
    attendance = rng.uniform(0.45, 1.0, rows)
    assignments = rng.uniform(35, 100, rows)
    gpa = rng.uniform(2.0, 4.0, rows)
    support = rng.poisson(1.4, rows).clip(0, 6)
    risk_score = (
        2.2 * (1 - attendance)
        + 0.018 * (70 - assignments).clip(min=0)
        + 0.9 * (3.0 - gpa).clip(min=0)
        + 0.12 * support
    )
    at_risk = (risk_score > np.quantile(risk_score, 0.62)).astype(int)
    return pd.DataFrame(
        {
            "student_id": [f"S{i:04d}" for i in range(1, rows + 1)],
            "attendance_rate": attendance.round(4),
            "assignment_score": assignments.round(2),
            "prior_gpa": gpa.round(2),
            "support_requests": support,
            "at_risk": at_risk,
        }
    )


def load_data() -> pd.DataFrame:
    """Load real data when present, otherwise create a clearly labelled demo set."""
    if DATA_PATH.exists():
        frame = pd.read_csv(DATA_PATH)
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        frame = _demo_data()
        frame.to_csv(DATA_DIR / "students_demo.csv", index=False)
        print(f"No {DATA_PATH.name} found; using deterministic demo data.")
    missing = set(["student_id", *FEATURES]) - set(frame.columns)
    if missing:
        raise ValueError(f"{DATA_PATH} is missing required columns: {sorted(missing)}")
    if "at_risk" not in frame.columns:
        frame = frame.copy()
        frame["at_risk"] = (
            (frame["attendance_rate"] < 0.75)
            | (frame["assignment_score"] < 55)
            | (frame["prior_gpa"] < 2.5)
        ).astype(int)
    if frame["at_risk"].nunique() < 2:
        raise ValueError("The at_risk label must contain both 0 and 1 classes.")
    return frame


def _top_factors(model: Pipeline, row: pd.DataFrame) -> list[str]:
    """Return three factors with the largest absolute local linear contribution."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed = preprocessor.transform(row)
    values = np.asarray(transformed).ravel()
    coefficients = classifier.coef_[0]
    names = preprocessor.get_feature_names_out()
    contributions = np.abs(values * coefficients)
    ranked = np.argsort(contributions)[::-1]
    selected: list[str] = []
    for index in ranked:
        name = names[index].replace("num__", "").replace("cat__", "")
        if name not in selected:
            selected.append(name)
        if len(selected) == 3:
            break
    return selected + [""] * (3 - len(selected))


def train_and_publish() -> dict[str, float]:
    """Train, evaluate, explain, and publish predictions using the fixed contract."""
    frame = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame["at_risk"], test_size=0.25, random_state=42, stratify=frame["at_risk"]
    )
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), FEATURES)],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    test_prob = model.predict_proba(x_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
    }
    print(classification_report(y_test, test_pred, zero_division=0))

    all_prob = model.predict_proba(frame[FEATURES])[:, 1]
    confidence = np.maximum(all_prob, 1 - all_prob)
    factors = [_top_factors(model, frame.iloc[[index]][FEATURES]) for index in range(len(frame))]
    predictions = pd.DataFrame(
        {
            "student_id": frame["student_id"].astype(str),
            "risk_prob": np.round(all_prob, 6),
            "confidence": np.round(confidence, 6),
            "top_factor_1": [factor[0] for factor in factors],
            "top_factor_2": [factor[1] for factor in factors],
            "top_factor_3": [factor[2] for factor in factors],
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions[CONTRACT_COLUMNS].to_csv(PREDICTIONS_PATH, index=False)
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(model, handle)
    _write_explanations(model, frame)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {PREDICTIONS_PATH}")
    return metrics


def _write_explanations(model: Pipeline, frame: pd.DataFrame) -> None:
    """Persist SHAP values when available, otherwise persist equivalent linear values."""
    values = model.named_steps["preprocessor"].transform(frame[FEATURES])
    names = model.named_steps["preprocessor"].get_feature_names_out()
    classifier = model.named_steps["classifier"]
    method = "model_coefficient_contribution"
    shap_values = np.asarray(values) * classifier.coef_[0]
    try:
        import shap

        explainer = shap.LinearExplainer(model.named_steps["classifier"], values)
        shap_values = np.asarray(explainer.shap_values(values))
        method = "shap.LinearExplainer"
    except ImportError:
        print("SHAP is not installed; saved coefficient contributions with an explicit method label.")
    explanation = pd.DataFrame(shap_values, columns=names)
    explanation.insert(0, "student_id", frame["student_id"].astype(str).to_numpy())
    explanation.insert(1, "method", method)
    explanation.to_csv(EXPLANATIONS_PATH, index=False)


if __name__ == "__main__":
    train_and_publish()
