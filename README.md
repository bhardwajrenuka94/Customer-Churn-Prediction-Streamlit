# 🔮 Customer Churn Prediction — Live ML App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-prediction-app-9ibyinmgzvydcvdc2pe3qm.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-Real%20Telco-purple)

> **Predict which telecom customers are likely to churn — before they leave.**
> Enter customer details → get instant churn probability + risk label + retention recommendations.

🔗 **[Live App →](https://customer-churn-prediction-app-9ibyinmgzvydcvdc2pe3qm.streamlit.app/)**

---

## 📸 App Preview

> *(Add screenshot here — drag and drop image into GitHub)*

---

## 🎯 Problem Statement

A telecom company was losing customers without knowing who was about to leave. The goal: build a model that predicts churn probability for each customer so the retention team can act before it is too late — reducing revenue loss.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest |
| Dataset | Real Telco (Kaggle) |
| Training Records | 7,043 |
| Features Used | 19 |
| Class Imbalance Handling | SMOTE |
| **AUC-ROC** | **0.84** |
| Deployment | Streamlit Cloud (Free) |

---

## 🧠 Technical Approach

### Why Random Forest?
- Handles non-linear relationships without heavy tuning
- Built-in feature importance — explainable to business stakeholders
- Robust to outliers and missing values
- Outperformed Logistic Regression baseline on this dataset

### Class Imbalance — SMOTE
- Dataset had ~26% churn rate — imbalanced
- SMOTE applied **only on training set** to prevent data leakage
- Test set kept as original real data for genuine evaluation

### Threshold Tuning
- Default threshold (0.5) is not always optimal for imbalanced data
- Used precision-recall curve to find best threshold
- Maximises F1 score while keeping recall high (catching real churners)

### Top 5 Features by Importance
| Feature | Importance |
|---------|-----------|
| Contract Type | 21.1% |
| Online Security | 13.8% |
| Tenure | 10.9% |
| Tech Support | 10.5% |
| Monthly Charges | 9.2% |

All features make business sense — contract type and security services are the biggest churn drivers.

---

## 🚀 App Features

- **Churn Probability** — instant % score with risk label (High / Medium / Low)
- **Risk Gauge** — visual speedometer
- **Top Risk Factors** — bar chart from Random Forest feature importance
- **Retention Recommendations** — auto-generated business actions based on input data

---

## 🗂️ Project Structure

```
Customer-Churn-Prediction/
├── app.py                  ← Streamlit dashboard
├── churn_model.py          ← Model training script
├── requirements.txt        ← Dependencies
├── model.pkl               ← Trained Random Forest
├── scaler.pkl              ← StandardScaler
├── feature_columns.pkl     ← Feature order
└── threshold.pkl           ← Optimal decision threshold
```

---

## ⚙️ Run Locally

```bash
# 1. Clone repo
git clone https://github.com/bhardwajrenuka94/Customer-Churn-Prediction-Streamlit.git
cd Customer-Churn-Prediction-Streamlit

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add dataset
# Download Telco-Customer-Churn.csv from Kaggle and place in this folder

# 5. Train model
python churn_model.py

# 6. Run app
streamlit run app.py
```

---

## 💡 Business Impact

> Telecom company with 1 lakh customers at 26% churn rate = 26,000 churners/year.
> At ₹800/month average revenue = ₹2 crore monthly revenue at risk.
> Model with AUC 0.84 correctly ranks churners above non-churners 84% of the time —
> enabling targeted retention campaigns that can save crores annually.

---

## 🔧 Tech Stack

`Python` · `Scikit-Learn` · `imbalanced-learn (SMOTE)` · `XGBoost` · `Pandas` · `NumPy` · `Plotly` · `Streamlit`

---

## 👩‍💻 Built Bystr

**Renuka Bhardwaj**
Data Scientist | AnalytixLabs Certified | Bengaluru
🔗 [LinkedIn](https://linkedin.com/in/renuka-bhardwaj-9b93b62a7) · [GitHub](https://github.com/bhardwajrenuka94) · [Portfolio](https://bhardwajrenuka94.github.io)
