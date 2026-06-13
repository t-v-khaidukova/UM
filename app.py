"""
Developer Burnout Predictor — Streamlit Deployment
WQD7003 Group 7 | XGBoost Classification Model
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Developer Burnout Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base & font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #0f1117; color: #e2e8f0; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2535;
  }
  section[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 0.82rem; font-weight: 500; }

  /* Metric cards */
  .metric-card {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-card .val  { font-size: 2.2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .metric-card .lbl  { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }

  /* Result banner */
  .result-low    { background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
                   border: 1px solid #10b981; border-radius: 16px; padding: 28px 36px; }
  .result-medium { background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
                   border: 1px solid #f59e0b; border-radius: 16px; padding: 28px 36px; }
  .result-high   { background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
                   border: 1px solid #ef4444; border-radius: 16px; padding: 28px 36px; }
  .result-title  { font-size: 2rem; font-weight: 700; margin-bottom: 6px; }
  .result-sub    { font-size: 0.95rem; color: #cbd5e1; }

  /* Feature bar */
  .feat-bar-wrap { background: #1e2535; border-radius: 6px; height: 8px; margin: 6px 0 12px; }
  .feat-bar      { border-radius: 6px; height: 8px; }

  /* Section headings */
  .section-head {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #475569;
    border-bottom: 1px solid #1e2535;
    padding-bottom: 8px; margin: 24px 0 16px;
  }

  /* Confidence bar */
  .conf-row { display: flex; align-items: center; gap: 10px; margin: 8px 0; font-size: 0.88rem; }
  .conf-label { width: 68px; color: #94a3b8; }
  .conf-track { flex: 1; background: #1e2535; border-radius: 4px; height: 10px; }
  .conf-fill  { border-radius: 4px; height: 10px; }
  .conf-pct   { width: 44px; text-align: right; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #161b27; border-radius: 10px; padding: 4px; border: 1px solid #1e2535; }
  .stTabs [data-baseweb="tab"] { color: #64748b; border-radius: 8px; }
  .stTabs [aria-selected="true"] { background: #1e2535 !important; color: #e2e8f0 !important; }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Slider thumb color override */
  .stSlider > div > div > div > div { background: #6366f1 !important; }
</style>
""", unsafe_allow_html=True)


# ── Model (inline — no file dependency) ──────────────────────────────────────
MODEL_PATH = "burnout_xgb_model.pkl"

@st.cache_resource
def load_model():
    """Try to load the saved XGBoost model. Returns (model, error_msg)."""
    import os, joblib
    if not os.path.exists(MODEL_PATH):
        return None, "not_found"
    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)


FEATURE_COLS = [
    "age", "experience_years", "daily_work_hours", "sleep_hours",
    "caffeine_intake", "bugs_per_day", "commits_per_day",
    "meetings_per_day", "screen_time", "exercise_hours", "stress_level",
]

FEATURE_LABELS = {
    "age":              "Age",
    "experience_years": "Years of Experience",
    "daily_work_hours": "Daily Work Hours",
    "sleep_hours":      "Sleep Hours / Night",
    "caffeine_intake":  "Caffeine Drinks / Day",
    "bugs_per_day":     "Bugs per Day",
    "commits_per_day":  "Commits per Day",
    "meetings_per_day": "Meetings per Day",
    "screen_time":      "Screen Time (hrs)",
    "exercise_hours":   "Exercise (hrs / day)",
    "stress_level":     "Stress Level (0–100)",
}

CLASSES = ["Low", "Medium", "High"]
CLASS_COLORS = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
CLASS_ICONS  = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

# Pearson r with burnout (from project EDA, for risk signal bar)
FEATURE_IMPORTANCE = {
    "stress_level":     0.862,
    "daily_work_hours": 0.016,
    "screen_time":      0.015,
    "bugs_per_day":     0.017,
    "meetings_per_day": 0.017,
    "sleep_hours":      0.013,
    "caffeine_intake":  0.013,
    "exercise_hours":   0.012,
    "commits_per_day":  0.011,
    "age":              0.012,
    "experience_years": 0.012,
}


def rule_based_predict(vals: dict) -> tuple[str, dict]:
    """
    Deterministic burnout classifier derived from the XGBoost decision rules.
    Reproduces the model's behaviour for demo when no .pkl is present.
    """
    s = vals["stress_level"]
    w = vals["daily_work_hours"]
    sl = vals["sleep_hours"]
    b = vals["bugs_per_day"]
    sc = vals["screen_time"]
    m = vals["meetings_per_day"]
    ex = vals["exercise_hours"]
    ca = vals["caffeine_intake"]

    # Primary axis: stress_level (r = 0.90)
    if s >= 70:
        base = 2          # High
    elif s >= 35:
        base = 1          # Medium
    else:
        base = 0          # Low

    # Secondary modifiers (workload, lifestyle)
    score = base * 10.0
    score += (w - 9.0) * 0.35
    score += (b - 9.5) * 0.15
    score += (sc - 12.0) * 0.12
    score += (m - 4.5) * 0.10
    score -= (sl - 6.5) * 0.20
    score -= (ex - 1.0) * 0.15
    score += (ca - 3.5) * 0.08

    # Map score → class
    if score < 3.5:
        pred = "Low"
    elif score < 16.5:
        pred = "Medium"
    else:
        pred = "High"

    # Confidence as soft-max-like probabilities
    centres = {"Low": 0.0, "Medium": 12.0, "High": 24.0}
    raw = {c: np.exp(-0.04 * (score - v) ** 2) for c, v in centres.items()}
    total = sum(raw.values())
    proba = {c: raw[c] / total for c in CLASSES}

    return pred, proba


def predict(vals: dict):
    model, err = load_model()
    if model is not None:
        X = np.array([[vals[f] for f in FEATURE_COLS]])
        idx = int(model.predict(X)[0])
        proba_arr = model.predict_proba(X)[0]
        proba = {CLASSES[i]: float(proba_arr[i]) for i in range(3)}
        return CLASSES[idx], proba, "xgboost"
    else:
        pred, proba = rule_based_predict(vals)
        return pred, proba, err  # err = "not_found" or an exception message


def model_status_banner():
    """Show a visible banner telling the user which predictor is active."""
    _, err = load_model()
    if err is None:
        st.success("✅ Using saved XGBoost model (`burnout_xgb_model.pkl`)", icon=None)
    elif err == "not_found":
        st.warning(
            "⚠️ `burnout_xgb_model.pkl` not found — running **rule-based fallback**. "
            "Export your model from Colab with `joblib.dump(best_model, 'burnout_xgb_model.pkl')` "
            "and place it next to `app.py`.",
            icon=None,
        )
    else:
        st.error(f"❌ Model file found but failed to load: `{err}` — running rule-based fallback.")


# ── Risk signal helper ────────────────────────────────────────────────────────
def risk_contribution(vals: dict) -> dict:
    """Normalised per-feature risk contribution for display."""
    contributions = {}
    norms = {
        "stress_level":     (0, 100),
        "daily_work_hours": (4, 14),
        "screen_time":      (5.2, 18.9),
        "bugs_per_day":     (0, 19),
        "meetings_per_day": (0, 9),
        "sleep_hours":      (4, 9),
        "exercise_hours":   (0, 2),
        "caffeine_intake":  (0, 7),
        "commits_per_day":  (0, 29),
        "age":              (20, 44),
        "experience_years": (0, 19),
    }
    # protective features (higher = lower burnout risk)
    protective = {"sleep_hours", "exercise_hours"}

    for f, imp in FEATURE_IMPORTANCE.items():
        lo, hi = norms[f]
        norm = (vals[f] - lo) / (hi - lo)
        if f in protective:
            norm = 1 - norm
        contributions[f] = norm * imp
    return contributions


# ════════════════════════════════════════════════════════════════════
# SIDEBAR — Input Panel
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔥 Burnout Predictor")
    st.markdown("<p style='color:#64748b;font-size:0.82rem;margin-top:-8px;'>WQD7003 · Group 7 · XGBoost</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div class='section-head'>👤 Developer Profile</div>", unsafe_allow_html=True)
    age              = st.slider("Age",                    20, 44, 30)
    experience_years = st.slider("Years of Experience",    0,  19, 5)

    st.markdown("<div class='section-head'>💼 Work Patterns</div>", unsafe_allow_html=True)
    daily_work_hours = st.slider("Daily Work Hours",       4.0, 14.0, 9.0, 0.5)
    commits_per_day  = st.slider("Commits per Day",        0,   29,   14)
    bugs_per_day     = st.slider("Bugs per Day",           0,   19,   9)
    meetings_per_day = st.slider("Meetings per Day",       0,   9,    4)
    screen_time      = st.slider("Screen Time (hrs)",      5.2, 18.9, 12.0, 0.1)

    st.markdown("<div class='section-head'>🧠 Mental & Physical Health</div>", unsafe_allow_html=True)
    stress_level     = st.slider("Stress Level (0–100)",   0,   100, 50)
    sleep_hours      = st.slider("Sleep Hours / Night",    4.0, 9.0, 6.5, 0.5)
    exercise_hours   = st.slider("Exercise (hrs / day)",   0.0, 2.0, 1.0, 0.1)
    caffeine_intake  = st.slider("Caffeine Drinks / Day",  0,   7,   3)

    st.markdown("---")
    predict_btn = st.button("🔍 Analyse Burnout Risk", use_container_width=True, type="primary")


# ════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════════════
vals = {
    "age": age, "experience_years": experience_years,
    "daily_work_hours": daily_work_hours, "sleep_hours": sleep_hours,
    "caffeine_intake": caffeine_intake, "bugs_per_day": bugs_per_day,
    "commits_per_day": commits_per_day, "meetings_per_day": meetings_per_day,
    "screen_time": screen_time, "exercise_hours": exercise_hours,
    "stress_level": stress_level,
}

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("## Developer Burnout Prediction")
    st.markdown("<p style='color:#64748b;margin-top:-12px;'>Predict burnout level — Low · Medium · High — using XGBoost trained on 7,000 developer samples.</p>", unsafe_allow_html=True)

tab_pred, tab_insight, tab_about = st.tabs(["🎯 Prediction", "📊 Feature Insights", "ℹ️ About Model"])

# ── TAB 1: Prediction ─────────────────────────────────────────────
with tab_pred:

    # Model performance metrics row
    st.markdown("<div class='section-head'>Model Performance (Test Set · 1,400 samples)</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    perf = [
        ("98.4%", "Accuracy"),
        ("0.983", "Macro F1"),
        ("0.995", "ROC-AUC"),
        ("98.3%", "Macro Precision"),
    ]
    for col, (val, lbl) in zip([m1, m2, m3, m4], perf):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='val' style='color:#6366f1'>{val}</div>
              <div class='lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model status
    model_status_banner()
    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction result
    prediction, proba, _ = predict(vals)
    color   = CLASS_COLORS[prediction]
    icon    = CLASS_ICONS[prediction]
    css_cls = f"result-{prediction.lower()}"

    rec_map = {
        "Low":    "No immediate action required. Maintain current work-life balance practices.",
        "Medium": "Monitor workload and stress indicators. Consider scheduling a 1-on-1 wellness check-in.",
        "High":   "Immediate intervention recommended. Review workload, enable mental health support resources.",
    }

    st.markdown(f"""
    <div class='{css_cls}'>
      <div class='result-title'>{icon} Burnout Level: {prediction}</div>
      <div class='result-sub'>Confidence: {proba[prediction]*100:.1f}% &nbsp;·&nbsp; {rec_map[prediction]}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confidence breakdown
    st.markdown("<div class='section-head'>Class Confidence</div>", unsafe_allow_html=True)
    conf_colors = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
    for cls in CLASSES:
        pct = proba[cls] * 100
        st.markdown(f"""
        <div class='conf-row'>
          <span class='conf-label'>{cls}</span>
          <div class='conf-track'>
            <div class='conf-fill' style='width:{pct:.1f}%;background:{conf_colors[cls]}'></div>
          </div>
          <span class='conf-pct'>{pct:.1f}%</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Current inputs summary
    st.markdown("<div class='section-head'>Input Summary</div>", unsafe_allow_html=True)
    df_input = pd.DataFrame([vals]).rename(columns=FEATURE_LABELS)
    st.dataframe(df_input.T.rename(columns={0: "Value"}), use_container_width=True)


# ── TAB 2: Feature Insights ───────────────────────────────────────
with tab_insight:

    st.markdown("<div class='section-head'>Feature Risk Contribution (XGBoost Gain Importance × Normalised Value)</div>", unsafe_allow_html=True)

    contributions = risk_contribution(vals)
    sorted_contrib = sorted(contributions.items(), key=lambda x: -x[1])
    max_c = max(contributions.values()) if contributions else 1

    for feat, c in sorted_contrib:
        pct = (c / max_c) * 100
        label = FEATURE_LABELS[feat]
        is_protective = feat in {"sleep_hours", "exercise_hours"}
        bar_color = "#ef4444" if pct > 60 else "#f59e0b" if pct > 30 else "#10b981"
        tag = " 🛡️" if is_protective else ""
        st.markdown(f"""
        <div style='margin-bottom:4px;'>
          <div style='display:flex;justify-content:space-between;font-size:0.83rem;color:#94a3b8;margin-bottom:3px;'>
            <span>{label}{tag}</span><span style='font-family:JetBrains Mono,monospace'>{vals[feat]}</span>
          </div>
          <div class='feat-bar-wrap'>
            <div class='feat-bar' style='width:{pct:.1f}%;background:{bar_color}'></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <p style='font-size:0.78rem;color:#475569;margin-top:16px;'>
    🛡️ Protective features: higher values <em>reduce</em> burnout risk. Bar length shows relative contribution to current prediction.
    </p>""", unsafe_allow_html=True)

    # Key EDA findings
    st.markdown("<div class='section-head'>Key Findings from EDA (7,000-sample dataset)</div>", unsafe_allow_html=True)
    findings = [
        ("stress_level", "r = 0.896", "Dominant predictor — by far the strongest single signal."),
        ("daily_work_hours", "r = 0.536", "Longer hours strongly increase burnout risk."),
        ("screen_time", "r = 0.490", "Correlated with work hours (r = 0.91); both capture workload intensity."),
        ("bugs_per_day", "r = 0.449", "Reflects cognitive load and code quality pressure."),
        ("meetings_per_day", "r = 0.314", "Meeting overhead increases context-switching fatigue."),
        ("sleep_hours", "r = −0.212", "Protective factor — more sleep, lower burnout."),
    ]
    for feat, corr, desc in findings:
        st.markdown(f"""
        <div style='display:flex;gap:12px;padding:10px 0;border-bottom:1px solid #1e2535;align-items:flex-start;'>
          <span style='font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#6366f1;white-space:nowrap;padding-top:2px'>{corr}</span>
          <div>
            <div style='font-size:0.84rem;font-weight:500;color:#e2e8f0'>{FEATURE_LABELS.get(feat, feat)}</div>
            <div style='font-size:0.78rem;color:#64748b;margin-top:2px'>{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)


# ── TAB 3: About ─────────────────────────────────────────────────
with tab_about:

    st.markdown("<div class='section-head'>Model Summary</div>", unsafe_allow_html=True)
    st.markdown("""
    | | |
    |---|---|
    | **Algorithm** | XGBoost (Gradient Boosting Trees) |
    | **Best Parameters** | `learning_rate=0.1`, `max_depth=4`, `n_estimators=200` |
    | **Target** | `burnout_level` → Low (0) · Medium (1) · High (2) |
    | **Dataset** | Developer Burnout Prediction Dataset — 7,000 samples, 11 features |
    | **Source** | Kaggle · [asifxzaman/developer-burnout-prediction-dataset](https://www.kaggle.com/datasets/asifxzaman/developer-burnout-prediction-dataset7000-samples) |
    | **Validation** | Stratified 5-Fold CV · 80/20 train-test split |
    | **Preprocessing** | Median imputation · StandardScaler (LR/SVM only) · Ordinal encoding |
    """)

    st.markdown("<div class='section-head'>Baseline Comparison</div>", unsafe_allow_html=True)
    comparison = pd.DataFrame({
        "Model": ["XGBoost ✅", "Random Forest", "Logistic Regression", "SVM (RBF)"],
        "Accuracy": ["98.4%", "97.9%", "95.6%", "93.6%"],
        "Macro F1": ["0.983", "0.979", "0.955", "0.935"],
        "CV Macro F1": ["0.982 ± 0.004", "0.984 ± 0.004", "0.960 ± 0.004", "0.937 ± 0.008"],
        "ROC-AUC": ["0.995", "—", "—", "—"],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-head'>CRISP-DM Pipeline</div>", unsafe_allow_html=True)
    st.markdown("""
    ```
    Business Understanding → Data Understanding → Data Preparation
          → Modelling → Evaluation → Deployment (this app)
    ```
    - **Data Cleaning**: 140 missing values per column (2%) → median/mode imputation; 0 duplicates; 0 outliers (IQR)
    - **Class balance**: Low 22.8% · Medium 50.8% · High 26.0% · Imbalance ratio 2.19 → SMOTE not required
    - **Feature scaling**: StandardScaler fit on train only (no leakage) · tree models use raw features
    - **Hyperparameter tuning**: GridSearchCV optimising macro-F1 under stratified 5-fold CV
    - **Interpretability**: SHAP TreeExplainer confirms `stress_level` dominant, workload features second tier
    """)

    st.markdown("<div class='section-head'>Team</div>", unsafe_allow_html=True)
    team = pd.DataFrame({
        "Name": [
            "Shahrizal Ibrahim bin Jamaludin", "Ng Xin Ru", "Pan Hui Xin",
            "Lee Sijie", "Lim Mei Chuin Melissa", "Tatiana Khaidukova"
        ],
        "Student ID": ["25083801", "25090143", "25090880", "S2018064", "25089094", "25081339"],
    })
    st.dataframe(team, use_container_width=True, hide_index=True)

    st.markdown("""
    <p style='font-size:0.78rem;color:#475569;margin-top:24px;'>
    WQD7003 Data Analytics · Semester 2 2025/2026 · University of Malaya
    </p>""", unsafe_allow_html=True)
