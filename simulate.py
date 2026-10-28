from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from trust import evaluate_trust

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
RESULTS_PATH = OUTPUT_DIR / "four_condition_results.csv"
CHART_PATH = OUTPUT_DIR / "four_condition_results.png"


def run_simulation(n_students: int = 200) -> pd.DataFrame:
    np.random.seed(42)

    ground_truth = np.random.binomial(1, 0.35, size=n_students)
    ai_correct = np.random.binomial(1, 0.80, size=n_students)
    ai_flags = np.where(ai_correct == 1, ground_truth, 1 - ground_truth)

    acc_c1 = np.mean(ai_flags == ground_truth)
    t_c1 = evaluate_trust(t=0.30, r=0.80, u=0.50, s=0.00, e=acc_c1)

    c2_actions = []
    for flag in ai_flags:
        c2_actions.append(flag if np.random.rand() < 0.85 else 1 - flag)
    c2_actions = np.array(c2_actions)
    acc_c2 = np.mean(c2_actions == ground_truth)
    s_c2 = np.mean(
        ((c2_actions == ai_flags) & (ai_flags == ground_truth))
        | ((c2_actions != ai_flags) & (ai_flags != ground_truth))
    )
    t_c2 = evaluate_trust(t=0.45, r=0.80, u=0.68, s=s_c2, e=acc_c2)

    c3_actions = []
    for flag, correct in zip(ai_flags, ai_correct):
        if correct:
            c3_actions.append(flag if np.random.rand() < 0.90 else 1 - flag)
        else:
            c3_actions.append(1 - flag if np.random.rand() < 0.45 else flag)
    c3_actions = np.array(c3_actions)
    acc_c3 = np.mean(c3_actions == ground_truth)
    s_c3 = np.mean(
        ((c3_actions == ai_flags) & (ai_flags == ground_truth))
        | ((c3_actions != ai_flags) & (ai_flags != ground_truth))
    )
    t_c3 = evaluate_trust(t=0.82, r=0.80, u=0.74, s=s_c3, e=acc_c3)

    c4_actions = []
    for flag, correct in zip(ai_flags, ai_correct):
        if correct:
            c4_actions.append(flag if np.random.rand() < 0.94 else 1 - flag)
        else:
            c4_actions.append(1 - flag if np.random.rand() < 0.70 else flag)
    c4_actions = np.array(c4_actions)
    acc_c4 = np.mean(c4_actions == ground_truth)
    s_c4 = np.mean(
        ((c4_actions == ai_flags) & (ai_flags == ground_truth))
        | ((c4_actions != ai_flags) & (ai_flags != ground_truth))
    )
    t_c4 = evaluate_trust(t=0.90, r=0.82, u=0.80, s=s_c4, e=acc_c4)

    rows = [
        {
            "Condition": "1. AI-Only",
            "T": t_c1["components"]["Transparency"],
            "R": t_c1["components"]["Reliability"],
            "U": t_c1["components"]["Usability"],
            "S": t_c1["components"]["Shared Control"],
            "E": t_c1["components"]["Task Effectiveness"],
            "Overall_TF": t_c1["overall_trust"],
            "Pass_Eq9": t_c1["threshold_passed"],
        },
        {
            "Condition": "2. HITL (No XAI)",
            "T": t_c2["components"]["Transparency"],
            "R": t_c2["components"]["Reliability"],
            "U": t_c2["components"]["Usability"],
            "S": t_c2["components"]["Shared Control"],
            "E": t_c2["components"]["Task Effectiveness"],
            "Overall_TF": t_c2["overall_trust"],
            "Pass_Eq9": t_c2["threshold_passed"],
        },
        {
            "Condition": "3. HITL + SHAP",
            "T": t_c3["components"]["Transparency"],
            "R": t_c3["components"]["Reliability"],
            "U": t_c3["components"]["Usability"],
            "S": t_c3["components"]["Shared Control"],
            "E": t_c3["components"]["Task Effectiveness"],
            "Overall_TF": t_c3["overall_trust"],
            "Pass_Eq9": t_c3["threshold_passed"],
        },
        {
            "Condition": "4. HITL + SHAP + Conf",
            "T": t_c4["components"]["Transparency"],
            "R": t_c4["components"]["Reliability"],
            "U": t_c4["components"]["Usability"],
            "S": t_c4["components"]["Shared Control"],
            "E": t_c4["components"]["Task Effectiveness"],
            "Overall_TF": t_c4["overall_trust"],
            "Pass_Eq9": t_c4["threshold_passed"],
        },
    ]

    df_results = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(RESULTS_PATH, index=False)

    ax = df_results.plot.bar(x="Condition", y=["Overall_TF", "S", "E"], ylim=(0, 1), rot=15)
    ax.set_ylabel("Score (0.0 - 1.0)")
    ax.set_title("TRUST Framework Evaluation Across Human-AI Conditions")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=160)
    plt.close(fig)

    print("\nFour-Condition Experimental Results Table:")
    print(df_results.to_string(index=False))
    print(f"\nWrote results table to: {RESULTS_PATH}")
    print(f"Wrote comparison chart to: {CHART_PATH}")
    return df_results


if __name__ == "__main__":
    run_simulation()