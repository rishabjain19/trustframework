# TRUST Framework: ML & Engine Viva Defense Notes (Person A)

## 1. Core Problem & Dataset Setup

- **Dataset**: UCI Student Performance (`student-mat.csv`, 395 records, 30 features).
- **At-Risk Target**: $G3 < 10$ (failing final grade binarized as class 1, pass as 0).
- **Data Leakage Prevention**: We dropped `G1` and `G2` (period grades). If included, the model easily memorizes prior grades rather than learning behavioral risk indicators like `absences`, `failures`, and `studytime`.
- **Train/Test Split**: 80/20 stratified holdout split so the model is evaluated strictly on unseen records.

## 2. Why Logistic Regression & Not a Complex Neural Net?

- Aligns with Cynthia Rudin's principle: for tabular risk assessment in high-stakes domains (education), simple models with explainability layers prevent unnecessary complexity, avoid severe overfitting on small cohorts (395 samples), and provide auditable linear coefficients.

## 3. Accuracy vs. Recall

- Our model achieves ~70% accuracy and 0.718 ROC-AUC.
- However, baseline recall at threshold 0.50 is ~31%.
- **Viva Defense**: Accuracy alone is insufficient. This gap is precisely why human educators are in the loop. The model outputs calibrated continuous probabilities so educators can review borderline students rather than relying on a hard binary decision.

## 4. SHAP (SHapley Additive exPlanations)

- Based on cooperative game theory.
- For every individual student, SHAP calculates the exact positive or negative contribution (log-odds) each feature made toward the risk prediction.
- Top drivers extracted: past course `failures`, high `absences`, low `studytime`.

## 5. Mathematical Formulations in `trust.py`

- **Corrected Eq. 1 (Overall Score)**:
  $$\Gamma F = w_T \cdot T + w_R \cdot R + w_U \cdot U + w_S \cdot S + w_E \cdot E$$
  (Equal weighting: $w_i = 0.20$).
- **Eq. 9 (Safety Guardrail)**:
  Any component dropping below $\theta = 0.60$ raises a critical flag.
- **Table 2 Verification Checkpoint**:
  Inputs ($T=0.86, R=0.86, U=0.78, S=0.92, E=0.96$) yield exactly **0.876**.

## 6. Four-Condition Simulation Results (`simulate.py`)

1. **AI-Only**: Overall $\Gamma F = 0.481$. Fails Eq. 9 because Shared Control $S = 0.00$.
2. **HITL without XAI**: Overall $\Gamma F = 0.654$. Fails Eq. 9 because Transparency $T = 0.45 < 0.60$ (educator shows uncritical complacency).
3. **HITL + SHAP**: Overall $\Gamma F = 0.808$. Passes Eq. 9 (educators override incorrect flags when features do not make pedagogical sense).
4. **HITL + SHAP + Confidence**: Overall $\Gamma F = 0.876$. Passes Eq. 9 with optimal calibrated reliance.
