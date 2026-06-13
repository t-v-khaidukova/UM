# Developer Burnout Predictor — Deployment Guide

WQD7003 Group 7 · XGBoost Classification Model

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501** by default.

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `requirements.txt` | Python dependencies |
| `burnout_xgb_model.pkl` | *(optional)* Saved XGBoost model from Colab |

---

## Deploy to Streamlit Cloud

1. Push `app.py` and `requirements.txt` to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo · set **Main file path** to `app.py`
4. Deploy

If you need the model file, add it to the repo (< 100 MB is fine for Git).

---

## Features

- **3-tab interface**: Prediction · Feature Insights · About Model
- **Real-time prediction** from sidebar sliders (no page reload)
- **Confidence bars** for all three classes (Low · Medium · High)
- **Feature risk contribution** chart with protective-factor markers
- **Model performance dashboard**: Accuracy 98.4% · Macro-F1 0.983 · ROC-AUC 0.995
- **EDA findings** summary from the 7,000-sample dataset
- **Full model comparison** table (XGBoost vs RF vs LR vs SVM)

---

## Dataset

Developer Burnout Prediction Dataset (7,000 samples)  
Source: [Kaggle](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples)

**Features (11):** age · experience_years · daily_work_hours · sleep_hours ·
caffeine_intake · bugs_per_day · commits_per_day · meetings_per_day ·
screen_time · exercise_hours · stress_level

**Target:** burnout_level → Low (0) · Medium (1) · High (2)
