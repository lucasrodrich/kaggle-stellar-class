"""
Kaggle Playground S6E6 - Predicting Stellar Class (GALAXY / QSO / STAR)
A single, self-contained script you can run in VS Code.

What it does:
  1. Loads train.csv (features + answer) and test.csv (features only).
  2. Builds extra "color index" features (band differences) that help
     separate the three object types.
  3. Trains a LightGBM gradient-boosting model using 5-fold cross-validation
     (so we get an honest accuracy estimate, not an over-optimistic one).
  4. Averages the 5 models' predictions on test.csv and writes submission.csv.

Requirements:  pip install pandas numpy scikit-learn lightgbm
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")


# ----------------------------------------------------------------------
# 2. FEATURE ENGINEERING
#    The 5 photometric bands are u, g, r, i, z (brightness in 5 filters).
#    In astronomy, the DIFFERENCE between two bands = an object's "color",
#    which separates stars/galaxies/quasars better than raw brightness.
#    We also add a couple of redshift-based features.
# ----------------------------------------------------------------------
def add_features(df):
    df = df.copy()
    bands = ["u", "g", "r", "i", "z"]

    # adjacent color indices (the classic, most useful ones)
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]

    # wider color indices (skip a band) - capture broader trends
    df["u_r"] = df["u"] - df["r"]
    df["u_i"] = df["u"] - df["i"]
    df["u_z"] = df["u"] - df["z"]
    df["g_i"] = df["g"] - df["i"]
    df["g_z"] = df["g"] - df["z"]
    df["r_z"] = df["r"] - df["z"]

    # simple summaries across the 5 bands
    df["mag_mean"]  = df[bands].mean(axis=1)
    df["mag_std"]   = df[bands].std(axis=1)
    df["mag_range"] = df[bands].max(axis=1) - df[bands].min(axis=1)

    # redshift interactions (redshift is the strongest physical signal)
    df["z_x_ug"] = df["redshift"] * df["u_g"]
    df["z_x_gr"] = df["redshift"] * df["g_r"]
    df["z_x_ri"] = df["redshift"] * df["r_i"]
    df["redshift_log"] = np.log1p(df["redshift"].clip(lower=0))
    return df

train = add_features(train)
test  = add_features(test)

# Turn the answer column into numbers (the model needs integers, not text)
classes = ["GALAXY", "QSO", "STAR"]
class_to_int = {c: i for i, c in enumerate(classes)}
y = train["class"].map(class_to_int).values

# Tell LightGBM which columns are categorical so it handles them natively
cat_cols = ["spectral_type", "galaxy_population"]
for c in cat_cols:
    train[c] = train[c].astype("category")
    # make test use the SAME category list as train
    test[c]  = pd.Categorical(test[c], categories=train[c].cat.categories)

# The list of feature columns we feed the model (everything except id/answer)
features = [c for c in train.columns if c not in ["id", "class"]]
X  = train[features]
Xt = test[features]
print(f"Using {len(features)} features on {len(X):,} training rows")


# ----------------------------------------------------------------------
# 3. TRAIN with 5-fold cross-validation
#    We split the training data into 5 parts. Each round, 4 parts train
#    the model and the 5th part (never seen during that round) is used to
#    check accuracy. This gives an honest estimate of real-world accuracy.
# ----------------------------------------------------------------------
params = dict(
    objective="multiclass", num_class=3,
    learning_rate=0.03,      # how big each tree's correction is (small = careful)
    num_leaves=127,          # how complex each tree can be
    subsample=0.8, subsample_freq=1,   # use 80% of rows per tree (reduces overfit)
    colsample_bytree=0.7,    # use 70% of features per tree (reduces overfit)
    reg_lambda=2.0, reg_alpha=0.5,     # penalties that keep the model simple
    min_child_samples=40,
    n_estimators=3000,       # max number of trees (early stopping ends it sooner)
    random_state=42, n_jobs=-1, verbose=-1,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred  = np.zeros((len(X), 3))   # predictions on held-out parts (for scoring)
test_pred = np.zeros((len(Xt), 3))  # averaged predictions on the real test set

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx], y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        callbacks=[lgb.early_stopping(80, verbose=False)],  # stop when it stops improving
    )
    oof_pred[va_idx] = model.predict_proba(X.iloc[va_idx])
    test_pred += model.predict_proba(Xt) / 5      # average over the 5 folds
    acc = accuracy_score(y[va_idx], oof_pred[va_idx].argmax(1))
    print(f"  fold {fold}: accuracy = {acc:.5f}  (best tree count: {model.best_iteration_})")

cv_acc = accuracy_score(y, oof_pred.argmax(1))
print(f"\nOverall cross-validated accuracy: {cv_acc:.5f}")


# ----------------------------------------------------------------------
# 4. WRITE SUBMISSION
#    For each test object, pick the class with the highest probability,
#    convert the number back to its name, and save in the required format.
# ----------------------------------------------------------------------
int_to_class = {i: c for c, i in class_to_int.items()}
final = pd.DataFrame({
    "id": test["id"],
    "class": [int_to_class[i] for i in test_pred.argmax(1)],
})
final.to_csv("submission.csv", index=False)
print("Saved submission.csv")
print(final["class"].value_counts())
