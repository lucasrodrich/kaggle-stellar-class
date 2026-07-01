"""
Tier 4: a proper STACK, the technique the leaderboard leaders use.

Instead of averaging near-identical models, we train DIVERSE base models, take
their out-of-fold (OOF) predictions, and train a meta-model (logistic regression)
that learns how to combine them — Chris Deotte's "logistic regression stacker".

Base models (deliberately different families, so they make different mistakes):
  - LightGBM   (gradient-boosted trees)
  - XGBoost    (gradient-boosted trees)
  - LogisticRegression (linear)          <- non-tree diversity
  - MLPClassifier (neural net)           <- non-tree diversity
Meta model: LogisticRegression on the stacked OOF probabilities.

Note: the leaders also include TabPFN (a tabular foundation model). It was
installed here but is unusable in this environment (CPU-only, and the package
requires an interactive cloud login that fails in a non-interactive shell), so the
diversity here is tree/linear/neural only. On GPU with TabPFN this stack would
likely gain more.

Run:  python src/stack.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from xgboost import XGBClassifier

import ensemble as E

BASELINE_LB = 0.95736  # pure LightGBM, private LB — the number that actually matters
N_CLASSES = len(E.CLASSES)


def prepare():
    train = E.add_features(pd.read_csv("data/train.csv"))
    test = E.add_features(pd.read_csv("data/test.csv"))
    y = train["class"].map(E.CLASS_TO_INT).values

    # GBM view: native categoricals
    for c in E.CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)
    feats = [c for c in train.columns if c not in ["id", "class"]]
    X_gbm, Xt_gbm = train[feats], test[feats]

    # Linear/neural view: one-hot the categoricals, everything numeric
    both = pd.concat([train[feats], test[feats]], axis=0)
    both = pd.get_dummies(both, columns=E.CAT_COLS, dummy_na=True)
    X_lin = both.iloc[:len(train)].to_numpy(dtype=np.float32)
    Xt_lin = both.iloc[len(train):].to_numpy(dtype=np.float32)
    return X_gbm, Xt_gbm, X_lin, Xt_lin, y


def oof_and_test(name, fit_predict, X, y, Xt):
    """5-fold OOF + fold-averaged test predictions for one base model."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(y), N_CLASSES))
    test_pred = np.zeros((Xt.shape[0], N_CLASSES))
    for tr, va in skf.split(np.zeros(len(y)), y):
        p_va, p_te = fit_predict(tr, va, X, y, Xt)
        oof[va] = p_va
        test_pred += p_te / 5
    print(f"  base [{name:9s}] OOF acc = {accuracy_score(y, oof.argmax(1)):.5f}")
    return oof, test_pred


def fit_lgb(tr, va, X, y, Xt):
    m = lgb.LGBMClassifier(**E.LGB_PARAMS)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
          callbacks=[lgb.early_stopping(E.EARLY_STOP, verbose=False)])
    return m.predict_proba(X.iloc[va]), m.predict_proba(Xt)


def fit_xgb(tr, va, X, y, Xt):
    m = XGBClassifier(**E.XGB_PARAMS)
    m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
    return m.predict_proba(X.iloc[va]), m.predict_proba(Xt)


def fit_logreg(tr, va, X, y, Xt):
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1))
    m.fit(X[tr], y[tr])
    return m.predict_proba(X[va]), m.predict_proba(Xt)


def fit_mlp(tr, va, X, y, Xt):
    m = make_pipeline(StandardScaler(),
                      MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                    max_iter=60, early_stopping=True,
                                    random_state=42))
    m.fit(X[tr], y[tr])
    return m.predict_proba(X[va]), m.predict_proba(Xt)


def main():
    X_gbm, Xt_gbm, X_lin, Xt_lin, y = prepare()
    print(f"train {len(y):,} rows\n")

    print("Training diverse base models (5-fold OOF each)...")
    oof_lgb, te_lgb = oof_and_test("LightGBM", fit_lgb, X_gbm, y, Xt_gbm)
    oof_xgb, te_xgb = oof_and_test("XGBoost", fit_xgb, X_gbm, y, Xt_gbm)
    oof_lr, te_lr = oof_and_test("LogReg", fit_logreg, X_lin, y, Xt_lin)
    oof_mlp, te_mlp = oof_and_test("MLP", fit_mlp, X_lin, y, Xt_lin)

    # Stack: meta-features = all base OOF probabilities
    meta_X = np.hstack([oof_lgb, oof_xgb, oof_lr, oof_mlp])
    meta_Xt = np.hstack([te_lgb, te_xgb, te_lr, te_mlp])
    meta = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)

    # honest stack CV: meta evaluated out-of-fold on the (leak-free) base OOF
    meta_oof = cross_val_predict(
        meta, meta_X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        method="predict_proba", n_jobs=-1)
    stack_cv = accuracy_score(y, meta_oof.argmax(1))

    print("\n" + "=" * 56)
    print("Base OOF accuracies:")
    for n, o in [("LightGBM", oof_lgb), ("XGBoost", oof_xgb),
                 ("LogReg", oof_lr), ("MLP", oof_mlp)]:
        print(f"  {n:9s} {accuracy_score(y, o.argmax(1)):.5f}")
    print(f"\nSTACK (LogReg meta) CV : {stack_cv:.5f}")
    print(f"LightGBM baseline CV   : 0.96805")
    print("  (reminder: CV is optimistic here ~0.011; LB is the judge)")
    print("=" * 56)

    meta.fit(meta_X, y)
    final = meta.predict(meta_Xt)
    pd.DataFrame({
        "id": pd.read_csv("data/test.csv")["id"],
        "class": [E.INT_TO_CLASS[i] for i in final],
    }).to_csv("submission_stack.csv", index=False)
    print("Saved submission_stack.csv")


if __name__ == "__main__":
    main()
