# TRUST Framework

This repository contains the two-person student-risk and educator-trust project.

## Team boundary

- **Person A (ML/TRUST):** `model.py`, `trust.py`, `simulate.py`, `data/`, and `outputs/`.
- **Person B (dashboard/report):** `app.py`, `report/`, `demo/`, and `logs/actions.csv`.

Both people may read the other person's files, but should only edit their own
files after agreeing on the interfaces below.

## Shared file contract

### `outputs/predictions.csv` (written by Person A)

The file must contain these columns, in this order:

```text
student_id,risk_prob,confidence,top_factor_1,top_factor_2,top_factor_3
```

- `student_id`: stable string identifier.
- `risk_prob`: floating-point probability in `[0, 1]` that the student is at risk.
- `confidence`: floating-point model confidence in `[0, 1]`.
- `top_factor_1..3`: human-readable explanation names.

### `logs/actions.csv` (written by Person B, read by Person A)

The file must contain these columns, in this order:

```text
student_id,ai_flag,educator_action,timestamp
```

- `ai_flag`: `1` when the dashboard displayed an AI risk flag, otherwise `0`.
- `educator_action`: one of `Approve`, `Adapt`, or `Override`.
- `timestamp`: ISO-8601 timestamp.

Person A's TRUST scorer tolerates a missing or empty actions file so the ML
pipeline can run before the dashboard exists.

## Setup

```powershell
cd .\trustframework
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python model.py
python simulate.py
python trust.py
```

If `data/students.csv` is absent, `model.py` creates a deterministic demo
dataset and reports that fact. Replace it with real data before final results.

## Execution plan

1. Freeze the two CSV contracts above.
2. Train and evaluate the at-risk model and write `outputs/predictions.csv`.
3. Generate `outputs/shap_explanations.csv` (SHAP when installed, with an
   explicit model-coefficient fallback for environments where SHAP is absent).
4. Score dashboard actions with the TRUST functions in `trust.py`.
5. Run the four simulated educator conditions and write the results table and
   chart.
6. Person B connects the Streamlit dashboard to the same files and writes
   `logs/actions.csv`.
7. Run the end-to-end validation before the report/demo.

The current branch is intentionally uncommitted because the remote repository
was empty when it was cloned.
