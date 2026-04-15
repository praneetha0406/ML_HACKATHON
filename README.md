# ⚡ Adaptive Energy Prediction System (AEPS)

![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=flat-square)
![Accuracy](https://img.shields.io/badge/R%C2%B2_Accuracy-99.17%25-blue?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-OpenML_ELEC2-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=flat-square)

An enterprise-grade, localized short-term electricity demand forecasting pipeline optimized for the highly volatile Australian National Electricity Market (NEM). By predicting load distributions 30 minutes seamlessly ahead of time, AEPS allows municipal grids to spin-up generator arrays perfectly in sync with real-time human consumption, eliminating blackouts and preventing millions of dollars in mechanical waste.

## 🚀 Key Features

*   **Zero-Leakage Architecture:** Mathematically sound autoregressive pipeline relying solely on temporal momentum (`lag` features) and cyclic sinusoidal encodings. Strictly isolated from absolute future inputs or concurrent market variables.
*   **Ensemble Engine (Task 2):** Evaluates Linear Regression, XGBoost, an Artificial Neural Network (ANN), and Random Forest. Our hyper-tuned Random Forest algorithm dominated the leaderboard (99.17% R², 0.010 MAE).
*   **SHAP Explainable Machine Learning (Task 3):** No "Black Boxes". Incorporates enterprise-ready `TreeExplainer` mechanisms from the SHAP library to visualize and audit exact feature importance decisions in real-time.
*   **Aesthetic Command Dashboard:** Features a native, dark-themed Streamlit UI equipped with a scrollable pseudo-terminal log, modular architectural layouts, and responsive visualization tabs.

## 📂 Project Architecture

```text
├── Documentation/                         # Technical audit logs and methodology
├── Task_1_Data_Preprocessing/             # Ingestion, Feature Engineering, StandardScaler logic
├── Task_2_Model_Development/              # 4x ML Model Training & Generation Pipeline
├── Task_3_Explainable_Selection/          # Streamlit UI & XAI Visualizations
├── Execute_Pipeline.py                    # Master orchestrator script (End-to-End synchronization)
├── Rigorous_Audit_Test.py                 # Automated anti-leakage boundary checker
├── requirements.txt                       # Project dependencies
```

## 🧠 Methodology Overview

The primary challenge in forecasting time-series electricity demand is managing the inertia of cyclic human behavior. 
To conquer this without falling trap to "Data Leakage" (cheating via future concurrency), our team implemented a **Shifted-Lag Framework**. 

Rather than feeding the algorithms raw concurrent market snapshots, AEPS captures the exact mathematical momentum of the grid utilizing 30, 60, and 90-minute historical intervals (`lag_1`, `lag_2`, `lag_3`). Coupled with geometric cyclic time components (`period_sin`, `period_cos`), the algorithm maps a continuous circle of time, recognizing that 11:30 PM is physically adjacent to Midnight. 

The models were evaluated strictly using a chronological 80/20 train/test split. No random data shuffling was permitted.

## ⚙️ Installation & Usage

**1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/YourUsername/AEPS-Project.git
cd AEPS-Project
pip install -r requirements.txt
```

**2. Option A: Run the End-To-End Master Pipeline (Training & Audit):**
To systematically fetch the dataset, engineer features, train all 4 models, generate validation plots, and run a strict data-leakage evaluation step-by-step:
```bash
python Execute_Pipeline.py
```

**3. Option B: Launch the Enterprise Dashboard:**
To view the front-end dashboard, complete with SHAP reasoning and accuracy benchmarking:
```bash
cd Task_3_Explainable_Selection
streamlit run dashboard.py
```

---
*Built purposefully for the Fall Machine Learning Hackathon.*
