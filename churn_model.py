# churn_model.py
# Customer Churn Prediction — Telco Dataset
# Model: Random Forest | Handles class imbalance with SMOTE
# Built by: Renuka Bhardwaj

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE

# ── Step 1: Load Data ────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("Telco-Customer-Churn.csv")
print(f"Shape: {df.shape}")

# ── Step 2: Clean Data ───────────────────────────────────────────────────────
# TotalCharges has some empty strings — convert to number
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Drop customer ID — not useful for prediction
df.drop(columns=["customerID"], inplace=True)

# Convert target to 0/1
df["Churn"] = (df["Churn"] == "Yes").astype(int)
print(f"Churn rate: {df['Churn'].mean():.1%}")

# ── Step 3: Encode Categorical Columns ──────────────────────────────────────
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# ── Step 4: Split Features and Target ───────────────────────────────────────
FEATURES = [col for col in df.columns if col != "Churn"]
X = df[FEATURES]
y = df["Churn"]

print(f"Total features: {len(FEATURES)}")
print(f"Features: {FEATURES}")

# ── Step 5: Train-Test Split ─────────────────────────────────────────────────
# stratify=y ensures both train and test have same churn ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── Step 6: Handle Class Imbalance with SMOTE ────────────────────────────────
# IMPORTANT: Apply SMOTE only on training data — never on test data
# This prevents data leakage
sm = SMOTE(random_state=42, k_neighbors=3)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"After SMOTE — train size: {len(X_train_res)}")

# ── Step 7: Scale Features ───────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled  = scaler.transform(X_test)

# ── Step 8: Train Random Forest ──────────────────────────────────────────────
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,       # number of trees
    max_depth=10,           # max depth of each tree
    min_samples_leaf=4,     # minimum samples at leaf node
    class_weight="balanced",# handles class imbalance
    random_state=42,
    n_jobs=-1               # use all CPU cores
)
model.fit(X_train_scaled, y_train_res)

# ── Step 9: Find Best Threshold ──────────────────────────────────────────────
# Default threshold is 0.5 but may not be optimal for imbalanced data
# We use precision-recall curve to find the threshold that maximises F1
y_proba = model.predict_proba(X_test_scaled)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores   = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx    = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

# ── Step 10: Evaluate ────────────────────────────────────────────────────────
y_pred = (y_proba >= best_threshold).astype(int)
f1  = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\n{'='*40}")
print(f"  Best Threshold : {best_threshold:.2f}")
print(f"  F1 Score       : {f1:.2f}")
print(f"  AUC-ROC        : {auc:.2f}")
print(f"{'='*40}")

# Top 5 important features
print("\nTop 5 Features by Importance:")
feat_imp = pd.Series(model.feature_importances_, index=FEATURES)
print(feat_imp.sort_values(ascending=False).head(5).to_string())

# ── Step 11: Save All Artifacts ──────────────────────────────────────────────
with open("model.pkl",           "wb") as f: pickle.dump(model,          f)
with open("scaler.pkl",          "wb") as f: pickle.dump(scaler,         f)
with open("feature_columns.pkl", "wb") as f: pickle.dump(FEATURES,       f)
with open("threshold.pkl",       "wb") as f: pickle.dump(best_threshold,  f)

print("\n✅ Saved: model.pkl | scaler.pkl | feature_columns.pkl | threshold.pkl")
print(f"   F1={f1:.2f} | AUC={auc:.2f}")
print("\nNext step: streamlit run app.py")