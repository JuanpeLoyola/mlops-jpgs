"""
Data Drift Detection via Adversarial Validation
================================================
Strategy:
  - Label train samples as 0 and test samples as 1.
  - Drop target ("failure") and identifier ("id") columns.
  - Train an XGBoost classifier (default params) to distinguish them.
  - Evaluate with 5-fold stratified cross-validation using Accuracy.

Interpretation:
  - Accuracy ≈ 0.50  → No data drift  (model cannot distinguish train vs test)
  - Accuracy >> 0.50 → Data drift detected (distributions differ significantly)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# ── 1. Load data ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

# ── 2. Drop target and identifier columns ─────────────────────────────────────
# "failure" is the prediction target  → must not be used as a feature.
# "id"      is a sequential row index → leaks train/test membership by range.
train = train.drop(columns=["failure", "id"])
test  = test.drop(columns=["id"])

# ── 3. Adversarial labeling ───────────────────────────────────────────────────
train["_origin"] = 0   # train  → 0
test["_origin"]  = 1   # test   → 1

combined = pd.concat([train, test], ignore_index=True)

X = combined.drop(columns=["_origin"])
y = combined["_origin"]

# ── 4. Preprocessing ──────────────────────────────────────────────────────────
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols   = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numerical_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical_cols),
    ]
)

# ── 5. Model (default XGBoost parameters) ────────────────────────────────────
model = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   model),
])

# ── 6. 5-fold Stratified Cross-Validation ────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

# ── 7. Results ────────────────────────────────────────────────────────────────
print("=" * 55)
print("   DATA DRIFT DETECTION — Adversarial Validation")
print("=" * 55)
print(f"  Train samples        : {len(train):>7,}")
print(f"  Test  samples        : {len(test):>7,}")
print(f"  Features used        : {X.shape[1]:>7}  "
      f"({len(numerical_cols)} numeric, {len(categorical_cols)} categorical)")
print("-" * 55)
print(f"  CV Accuracy per fold : {np.round(scores, 4)}")
print(f"  Mean Accuracy        : {scores.mean():.4f}")
print(f"  Std  Accuracy        : {scores.std():.4f}")
print("-" * 55)

if scores.mean() > 0.55:
    print("  ⚠  DRIFT DETECTED — distributions differ significantly.")
else:
    print("  ✓  NO DRIFT — train and test are indistinguishable.")
print("=" * 55)
