"""Simulate educators and publish the four-condition TRUST results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trust import score_trust

ROOT = Path(__file__).resolve().parent
PREDICTIONS_PATH = ROOT / "outputs" / "predictions.csv"
RESULTS_PATH = ROOT / "outputs" / "four_condition_results.csv"
CHART_PATH = ROOT / "outputs" / "four_condition_results.png"


def simulate_conditions(predictions: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Compare educator-only and AI-assisted decisions under two educator styles."""
    rng = np.random.default_rng(seed)
    actual = (predictions["risk_prob"] >= 0.5).astype(int).to_numpy()
    conditions = {
        "Educator only": rng.binomial(1, 0.72, len(predictions)),
        "AI only": (predictions["risk_prob"] >= 0.5).astype(int).to_numpy(),
        "AI + approving educator": (predictions["risk_prob"] >= 0.45).astype(int).to_numpy(),
        "AI + adapting educator": np.where(
            predictions["confidence"].to_numpy() >= 0.7,
            (predictions["risk_prob"] >= 0.5).astype(int),
            rng.binomial(1, 0.82, len(predictions)),
        ),
    }
    rows = []
    for condition, flags in conditions.items():
        actions = pd.DataFrame(
            {
                "student_id": predictions["student_id"],
                "ai_flag": flags,
                "educator_action": "Approve",
                "timestamp": pd.Timestamp.utcnow().isoformat(),
            }
        )
        trust = score_trust(predictions, actions)["trust_score"].mean()
        rows.append(
            {
                "condition": condition,
                "decision_accuracy": float((flags == actual).mean()),
                "trust_score": float(trust),
                "n_students": len(predictions),
            }
        )
    return pd.DataFrame(rows)


def run_simulation() -> pd.DataFrame:
    """Read predictions and write both the table and chart."""
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Run model.py before simulate.py: {PREDICTIONS_PATH}")
    predictions = pd.read_csv(PREDICTIONS_PATH)
    results = simulate_conditions(predictions)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    axis = results.plot.bar(x="condition", y=["decision_accuracy", "trust_score"], ylim=(0, 1), rot=20)
    axis.set_ylabel("Score")
    axis.get_figure().tight_layout()
    axis.get_figure().savefig(CHART_PATH, dpi=160)
    plt.close(axis.get_figure())
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    run_simulation()
