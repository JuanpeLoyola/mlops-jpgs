"""
Product Failure Prediction
==========================
Task: Binary classification to predict the probability that an individual
      product will fail, given its attributes and lab measurement results.

Model : XGBoost 
CV    : 5-fold Stratified Cross-Validation
Metric: ROC-AUC 

Output: submission.csv  with columns [id, failure]
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

# ── 2. Separate features / target / ids ───────────────────────────────────────
test_ids = test["id"]

X_train = train.drop(columns=["id", "failure"])
y_train = train["failure"]
X_test  = test.drop(columns=["id"])

# ── 3. Preprocessing ──────────────────────────────────────────────────────────
categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
numerical_cols   = X_train.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numerical_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical_cols),
    ]
)

# ── 4. Model ─────────────────────────────────────
model = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   model),
])

# ── 5. 5-fold Stratified Cross-Validation ────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

# ── 6. Train on full dataset & predict ───────────────────────────────────────
pipeline.fit(X_train, y_train)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# ── 7. Save submission ────────────────────────────────────────────────────────
submission = pd.DataFrame({"id": test_ids, "failure": y_pred_proba})
submission.to_csv(DATA_DIR / "submission.csv", index=False)

# ── 8. Results ────────────────────────────────────────────────────────────────
print("=" * 55)
print("          PRODUCT FAILURE PREDICTION")
print("=" * 55)
print(f"  Train samples        : {len(X_train):>7,}")
print(f"  Test  samples        : {len(X_test):>7,}")
print(f"  Positive rate (train): {y_train.mean():>7.2%}  (class imbalance)")
print(f"  Features used        : {X_train.shape[1]:>7}  "
      f"({len(numerical_cols)} numeric, {len(categorical_cols)} categorical)")
print("-" * 55)
print(f"  CV ROC-AUC per fold  : {np.round(scores, 4)}")
print(f"  Mean ROC-AUC         : {scores.mean():.4f}")
print(f"  Std  ROC-AUC         : {scores.std():.4f}")
print("-" * 55)
print(f"  Submission saved to  : data/submission.csv")
print("=" * 55)
