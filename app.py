# app.py
# Customer Churn Prediction Dashboard
# Real Telco Dataset | Random Forest | AUC ~ 0.83
# Built by: Renuka Bhardwaj | Data Scientist | AnalytixLabs Certified

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
import plotly.graph_objects as go

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction | Renuka Bhardwaj",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

  :root {
    --bg      : #0a0d14;
    --surface : #111520;
    --card    : #161c2d;
    --accent  : #00e5ff;
    --green   : #00d68f;
    --red     : #ff6b6b;
    --yellow  : #ffd60a;
    --text    : #e8eaf0;
    --muted   : #6b7280;
    --border  : #1e2640;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
  }

  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }

  h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: var(--text) !important;
  }

  .kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
  }
  .kpi-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
  }
  .kpi-lbl {
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
  }

  .result-box {
    border-radius: 14px;
    padding: 28px;
    text-align: center;
    margin: 12px 0;
  }
  .high { background: linear-gradient(135deg,#1a0a0a,#2d0f0f); border: 2px solid var(--red); }
  .mid  { background: linear-gradient(135deg,#1a1400,#2d2200); border: 2px solid var(--yellow); }
  .low  { background: linear-gradient(135deg,#0a1a12,#0f2d1e); border: 2px solid var(--green); }

  .result-pct {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
  }

  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
  }
  .b-rf    { background:#00e5ff22; color:#00e5ff; border:1px solid #00e5ff44; }
  .b-smote { background:#a78bfa22; color:#a78bfa; border:1px solid #a78bfa44; }

  [data-testid="stButton"] button {
    background: linear-gradient(135deg, #00b4d8, #00e5ff) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 32px !important;
    font-size: 0.95rem !important;
    width: 100% !important;
  }

  .rec-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 14px 16px;
    min-height: 110px;
  }

  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load Model Artifacts ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl",           "rb") as f: model     = pickle.load(f)
    with open("scaler.pkl",          "rb") as f: scaler    = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f: features  = pickle.load(f)
    try:
        with open("threshold.pkl",   "rb") as f: threshold = pickle.load(f)
    except:
        threshold = 0.5
    return model, scaler, features, threshold

try:
    model, scaler, FEATURES, THRESHOLD = load_model()
    loaded = True
except FileNotFoundError:
    loaded = False


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 Churn Predictor")
    st.markdown("---")

    st.markdown("""
    <span class='badge b-rf'>Random Forest</span>&nbsp;
    <span class='badge b-smote'>SMOTE</span>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class='kpi-card'>
          <div class='kpi-val'>0.83</div>
          <div class='kpi-lbl'>AUC-ROC</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='kpi-card'>
          <div class='kpi-val'>7K+</div>
          <div class='kpi-lbl'>Records</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class='kpi-card'>
      <div class='kpi-val' style='font-size:1.1rem'>Real Telco Data</div>
      <div class='kpi-lbl'>Kaggle Dataset</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#6b7280; font-size:0.72rem; line-height:1.8'>
      Built by <b style='color:#00e5ff'>Renuka Bhardwaj</b><br>
      Data Scientist<br>
      AnalytixLabs Certified<br>
      Karnal, Haryana
    </div>
    """, unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:1.9rem; margin-bottom:4px'>Customer Churn Prediction</h1>
<p style='color:#6b7280; font-size:0.9rem; margin-bottom:28px'>
  Real Telco Dataset &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp;
  Enter customer details → get churn probability instantly
</p>
""", unsafe_allow_html=True)

if not loaded:
    st.error("⚠️ Model files not found. Run `python churn_model.py` first.")
    st.stop()


# ── Input Form ───────────────────────────────────────────────────────────────
st.markdown("### 📋 Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Account Info**")
    tenure         = st.slider("Tenure (months)", 0, 72, 12)
    MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0, 0.5)
    TotalCharges   = st.number_input(
        "Total Charges ($)", 0.0, 9000.0,
        value=float(round(tenure * MonthlyCharges, 2)), step=10.0
    )
    SeniorCitizen  = 1 if st.checkbox("Senior Citizen (65+)") else 0

with col2:
    st.markdown("**Contract & Billing**")

    contract_map   = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
    Contract       = contract_map[st.selectbox("Contract Type", list(contract_map.keys()))]

    payment_map    = {
        "Electronic Check": 0, "Mailed Check": 1,
        "Bank Transfer (Auto)": 2, "Credit Card (Auto)": 3
    }
    PaymentMethod  = payment_map[st.selectbox("Payment Method", list(payment_map.keys()))]

    PaperlessBilling = 1 if st.radio("Paperless Billing", ["No", "Yes"], horizontal=True) == "Yes" else 0

    internet_map   = {"No Internet": 0, "DSL": 1, "Fiber Optic": 2}
    InternetService= internet_map[st.selectbox("Internet Service", list(internet_map.keys()))]

    ml_map         = {"No Phone Service": 0, "No": 1, "Yes": 2}
    MultipleLines  = ml_map[st.selectbox("Multiple Lines", list(ml_map.keys()))]

with col3:
    st.markdown("**Services & Personal**")
    OnlineSecurity   = 1 if st.radio("Online Security",   ["No","Yes"], horizontal=True) == "Yes" else 0
    OnlineBackup     = 1 if st.radio("Online Backup",     ["No","Yes"], horizontal=True) == "Yes" else 0
    DeviceProtection = 1 if st.radio("Device Protection", ["No","Yes"], horizontal=True) == "Yes" else 0
    TechSupport      = 1 if st.radio("Tech Support",      ["No","Yes"], horizontal=True) == "Yes" else 0
    StreamingTV      = 1 if st.radio("Streaming TV",      ["No","Yes"], horizontal=True) == "Yes" else 0
    StreamingMovies  = 1 if st.radio("Streaming Movies",  ["No","Yes"], horizontal=True) == "Yes" else 0

    st.markdown("---")
    gender     = 1 if st.radio("Gender", ["Female","Male"], horizontal=True) == "Male" else 0
    Partner    = 1 if st.checkbox("Has Partner")    else 0
    Dependents = 1 if st.checkbox("Has Dependents") else 0

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔮 PREDICT CHURN PROBABILITY")


# ── Prediction ───────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Analysing customer profile..."):
        time.sleep(0.5)

    # Map input values to feature order model was trained on
    input_dict = {
        "gender":           gender,
        "SeniorCitizen":    SeniorCitizen,
        "Partner":          Partner,
        "Dependents":       Dependents,
        "tenure":           tenure,
        "PhoneService":     1,
        "MultipleLines":    MultipleLines,
        "InternetService":  InternetService,
        "OnlineSecurity":   OnlineSecurity,
        "OnlineBackup":     OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport":      TechSupport,
        "StreamingTV":      StreamingTV,
        "StreamingMovies":  StreamingMovies,
        "Contract":         Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod":    PaymentMethod,
        "MonthlyCharges":   MonthlyCharges,
        "TotalCharges":     TotalCharges,
    }

    # Build input array in exact feature order from training
    input_row    = [input_dict.get(f, 0) for f in FEATURES]
    input_scaled = scaler.transform(np.array([input_row]))
    churn_prob   = model.predict_proba(input_scaled)[0][1]
    churn_pct    = round(churn_prob * 100, 1)

    # Risk level
    if churn_prob >= 0.65:
        risk, box_cls, color, emoji = "HIGH RISK",   "high", "#ff6b6b", "🚨"
    elif churn_prob >= 0.35:
        risk, box_cls, color, emoji = "MEDIUM RISK", "mid",  "#ffd60a", "⚠️"
    else:
        risk, box_cls, color, emoji = "LOW RISK",    "low",  "#00d68f", "✅"

    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")

    _, mid_col, _ = st.columns([1, 2, 1])
    with mid_col:
        st.markdown(f"""
        <div class='result-box {box_cls}'>
          <div style='font-size:2rem; margin-bottom:8px'>{emoji}</div>
          <div class='result-pct' style='color:{color}'>{churn_pct}%</div>
          <div style='color:#9ca3af; margin-top:6px'>Churn Probability</div>
          <div style='font-family:Space Mono; font-weight:700;
               color:{color}; margin-top:10px'>{risk}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge + Feature Importance
    g_col, f_col = st.columns(2)

    with g_col:
        st.markdown("**Risk Gauge**")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_pct,
            number={"suffix": "%", "font": {"size": 34, "color": color, "family": "Space Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#6b7280", "tickfont": {"color": "#6b7280"}},
                "bar":  {"color": color, "thickness": 0.25},
                "bgcolor": "#161c2d", "bordercolor": "#1e2640",
                "steps": [
                    {"range": [0,  35], "color": "#0a1a12"},
                    {"range": [35, 65], "color": "#1a1400"},
                    {"range": [65,100], "color": "#1a0a0a"},
                ],
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0a0d14", font_color="#e8eaf0",
            height=230, margin=dict(t=20, b=20, l=30, r=30)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with f_col:
        st.markdown("**Top Risk Factors**")
        imp  = pd.Series(model.feature_importances_, index=FEATURES)
        top6 = imp.sort_values(ascending=True).tail(6)
        fig_bar = go.Figure(go.Bar(
            x=top6.values, y=top6.index,
            orientation="h",
            marker_color="#00e5ff",
            opacity=0.85,
        ))
        fig_bar.update_layout(
            paper_bgcolor="#0a0d14", plot_bgcolor="#161c2d",
            font_color="#e8eaf0", height=230,
            margin=dict(t=10, b=10, l=10, r=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Retention Recommendations
    st.markdown("### 💡 Retention Recommendations")
    recs = []

    if Contract == 0:
        recs.append(("📋 Upgrade Contract",
                      "Month-to-month customer — biggest churn driver. Offer 10-15% discount to switch to annual plan."))
    if MonthlyCharges > 70:
        recs.append(("💰 Review Pricing",
                      f"Monthly charges ${MonthlyCharges} are high. A loyalty discount or bundle offer can reduce churn risk."))
    if tenure < 12:
        recs.append(("🤝 Early Engagement",
                      "First-year customer — critical retention window. Assign dedicated support contact."))
    if not TechSupport:
        recs.append(("🛠 Offer Tech Support",
                      "No tech support active. A free 30-day trial significantly reduces churn."))
    if not OnlineSecurity:
        recs.append(("🔒 Security Bundle",
                      "No online security. Push security + backup bundle — strong retention upsell."))
    if InternetService == 2:
        recs.append(("📡 Fiber Loyalty Perk",
                      "Fiber optic users churn more. Offer a speed upgrade or monthly bill credit."))
    if PaymentMethod == 0:
        recs.append(("🏦 Switch to Auto-Pay",
                      "Electronic check payers churn more. Offer ₹5/month discount for auto-pay setup."))
    if not recs:
        recs.append(("✅ Low Risk Customer",
                      "Strong retention signals. Consider upselling premium add-ons."))

    rec_cols = st.columns(min(len(recs), 3))
    for i, (title, desc) in enumerate(recs[:3]):
        with rec_cols[i]:
            st.markdown(f"""
            <div class='rec-card'>
              <div style='font-weight:700; margin-bottom:8px; color:#e8eaf0'>{title}</div>
              <div style='color:#9ca3af; font-size:0.8rem; line-height:1.5'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    if len(recs) > 3:
        st.markdown("<br>", unsafe_allow_html=True)
        rec_cols2 = st.columns(min(len(recs) - 3, 3))
        for i, (title, desc) in enumerate(recs[3:6]):
            with rec_cols2[i]:
                st.markdown(f"""
                <div class='rec-card'>
                  <div style='font-weight:700; margin-bottom:8px; color:#e8eaf0'>{title}</div>
                  <div style='color:#9ca3af; font-size:0.8rem; line-height:1.5'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # Summary bar
    st.markdown("<br>", unsafe_allow_html=True)
    contract_label = {0:"Month-to-Month", 1:"One Year", 2:"Two Year"}[Contract]
    st.markdown(f"""
    <div style='background:#161c2d; border:1px solid #1e2640; border-radius:8px;
         padding:12px 16px; font-size:0.8rem; color:#6b7280'>
      Tenure: <b style='color:#00e5ff'>{tenure}m</b> &nbsp;|&nbsp;
      Contract: <b style='color:#00e5ff'>{contract_label}</b> &nbsp;|&nbsp;
      Monthly: <b style='color:#00e5ff'>${MonthlyCharges}</b> &nbsp;|&nbsp;
      Risk: <b style='color:{color}'>{risk}</b> &nbsp;|&nbsp;
      Model: Random Forest · AUC=0.83
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#3d4a6e; font-size:0.72rem;
     border-top:1px solid #1e2640; padding:14px; font-family:Space Mono'>
  CUSTOMER CHURN PREDICTION &nbsp;·&nbsp;
  BUILT BY RENUKA BHARDWAJ &nbsp;·&nbsp;
  RANDOM FOREST · AUC=0.83 · REAL TELCO DATA
</div>
""", unsafe_allow_html=True)