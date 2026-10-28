"""TRUST scoring for dashboard educator actions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PREDICTIONS_PATH = ROOT / "outputs" / "predictions.csv"
ACTIONS_PATH = ROOT / "logs" / "actions.csv"
ACTION_COLUMNS = ["student_id", "ai_flag", "educator_action", "timestamp"]


def read_actions(path: Path = ACTIONS_PATH) -> pd.DataFrame:
    """Read the dashboard contract, returning an empty typed frame before first use."""
    if not path.exists():
        return pd.DataFrame(columns=ACTION_COLUMNS)
    actions = pd.read_csv(path)
    missing = set(ACTION_COLUMNS) - set(actions.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return actions[ACTION_COLUMNS].copy()


def score_trust(predictions: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-action trust from agreement, confidence, and risk separation."""
    merged = actions.merge(predictions, on="student_id", how="inner", validate="many_to_one")
    if merged.empty:
        return pd.DataFrame(columns=["student_id", "trust_score"])
    ai_flag = merged["ai_flag"].astype(int)
    model_flag = (merged["risk_prob"] >= 0.5).astype(int)
    agreement = (ai_flag == model_flag).astype(float)
    separation = (merged["risk_prob"] - 0.5).abs() * 2
    trust_score = 0.5 * agreement + 0.3 * merged["confidence"] + 0.2 * separation
    return pd.DataFrame({"student_id": merged["student_id"], "trust_score": trust_score.round(6)})


def aggregate_trust(predictions_path: Path = PREDICTIONS_PATH, actions_path: Path = ACTIONS_PATH) -> float:
    """Return the mean TRUST score, or zero when no dashboard actions exist yet."""
    if not predictions_path.exists():
        raise FileNotFoundError(f"Run model.py before TRUST scoring: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    actions = read_actions(actions_path)
    scored = score_trust(predictions, actions)
    return float(scored["trust_score"].mean()) if not scored.empty else 0.0


if __name__ == "__main__":
    print(f"TRUST score: {aggregate_trust():.3f}")
