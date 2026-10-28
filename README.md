# TRUST Framework: Explainable AI & Human-in-the-Loop At-Risk Learner System

An empirical implementation and evaluation platform for the **TRUST Framework** in educational AI. This project operationalizes an **Early-Warning Student Risk Prediction System** paired with **Local Explainability (SHAP)**, human educator action controls (**Approve / Adapt / Override**), and an automated **TRUST Mathematical Scoring Engine** to evaluate human-AI collaborative decision-making.

---

##  Project Overview

This project provides a full end-to-end implementation of an at-risk learner dashboard evaluated against the TRUST framework metrics:

1. **At-Risk Learner Prediction**:
   - Uses the **UCI Student Performance Dataset** (`data/student-mat.csv`) to predict student academic failure risk ($G3 < 10$).
   - Explicitly drops interim period grades ($G1, G2$) to prevent data leakage, learning behavioral and academic risk factors.
   - Fits an interpretable Logistic Regression model producing continuous risk probabilities and calibrated confidence scores.

2. **Explainable AI (XAI) Layer**:
   - Implements **SHAP (SHapley Additive exPlanations)** via `shap.LinearExplainer`.
   - Extracts and ranks the top 3 contributing risk factors for each individual student (e.g., past failures, high absences, study time) to explain _why_ an alert was generated.

3. **Human-in-the-Loop (HITL) Dashboard**:
   - A Streamlit web dashboard where educators review predicted student risk, examine confidence and local SHAP explanations, and submit contextual intervention actions (`Approve`, `Adapt`, `Override`).
   - Every educator interaction is recorded into `logs/actions.csv` for auditability and trust scoring.

4. **TRUST Evaluation Engine**:
   - Implements the mathematical formulations for the 5 TRUST pillars: **Transparency ($T$)**, **Reliability ($R$)**, **Usability ($U$)**, **Shared Control ($S$)**, and **Task Effectiveness ($E$)**.
   - Aggregates scores into an overall composite index ($\Gamma F$):
     $$\Gamma F = w_T \cdot T + w_R \cdot R + w_U \cdot U + w_S \cdot S + w_E \cdot E$$
   - Enforces the **Eq. 9 safety guardrail** ($\theta = 0.60$), flagging any component failing to meet minimum operational safety standards.
   - Evaluates system reliance across four experimental conditions via simulated educator decision-making.

---

## Project Architecture

```text
trustframework/
├── data/
│   └── student-mat.csv          
├── logs/
│   ├── actions.csv             
│   └── predictions.csv          
├── outputs/
│   ├── four_condition_results.csv 
│   ├── four_condition_results.png 
│   ├── metrics.json            
│   ├── model.pkl               
│   ├── predictions.csv          
│   └── shap_explanations.csv    
├── model.py                    
├── trust.py                    
├── simulate.py                  
├── app.py                      
├── notes.md                     
└── requirements.txt             
![Project Screenshot](https://github.com/user-attachments/assets/ac5fa518-b302-486b-a17b-9578c68b188d)

