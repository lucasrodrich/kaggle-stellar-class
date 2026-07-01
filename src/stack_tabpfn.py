"""
Tier 4b: add TabPFN (a tabular foundation model) as a 5th, genuinely-strong-AND-
diverse base model to the stack — the member the leaderboard leaders rely on.

Runs fully LOCALLY: the weights come from a PUBLIC HuggingFace repo (no account,
no gating), and inference is on-CPU with telemetry/browser-login disabled, so no
data ever leaves the machine.

CPU is slow (~a hundred rows/s), so TabPFN uses a subsampled context and predicts
in batches. Its out-of-fold predictions cover a disjoint train subsample (enough to
train the logistic-regression meta-learner); the four fast models keep full OOF.

Run:  python src/stack_tabpfn.py
"""

import os
os.environ["TABPFN_NO_BROWSER"] = "1"
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from huggingface_hub import hf_hub_download
from tabpfn import TabPFNClassifier

import ensemble as E
import stack as S

CTX = 5000          # TabPFN training-context size (CPU speed vs strength trade-off)
OOF_SUB = 30000     # train rows (disjoint from context) used to train the meta
BATCH = 20000       # test prediction batch size


def tabpfn_predict(clf, X):
    """Batched predict_proba so a 247k test set doesn't blow memory."""
    out = np.zeros((len(X), 3))
    for i in range(0, len(X), BATCH):
        out[i:i + BATCH] = clf.predict_proba(X[i:i + BATCH])
        print(f"    TabPFN predicted {min(i + BATCH, len(X)):,}/{len(X):,}", flush=True)
    return out


def main():
    X_gbm, Xt_gbm, X_lin, Xt_lin, y = S.prepare()
    print(f"train {len(y):,} rows\n")

    print("Base models (4 fast, full 5-fold OOF)...")
    oof_lgb, te_lgb = S.oof_and_test("LightGBM", S.fit_lgb, X_gbm, y, Xt_gbm)
    oof_xgb, te_xgb = S.oof_and_test("XGBoost", S.fit_xgb, X_gbm, y, Xt_gbm)
    oof_lr, te_lr = S.oof_and_test("LogReg", S.fit_logreg, X_lin, y, Xt_lin)
    oof_mlp, te_mlp = S.oof_and_test("MLP", S.fit_mlp, X_lin, y, Xt_lin)

    # ---- TabPFN base (local, subsampled context) --------------------------
    print("\nTabPFN base (local CPU)...")
    ckpt = hf_hub_download(repo_id="Prior-Labs/tabpfn_3",
                           filename="tabpfn-v3-classifier-v3_default.ckpt")
    # context set R, and a disjoint OOF subsample D (leak-free: R never in D)
    idx = np.arange(len(y))
    R, rest = train_test_split(idx, train_size=CTX, random_state=0, stratify=y)
    D, _ = train_test_split(rest, train_size=OOF_SUB, random_state=1, stratify=y[rest])
    clf = TabPFNClassifier(model_path=ckpt, device="cpu",
                           ignore_pretraining_limits=True, n_estimators=1)
    clf.fit(X_lin[R], y[R])
    print("  predicting OOF subsample...", flush=True)
    tp_oof_D = tabpfn_predict(clf, X_lin[D])
    print("  predicting test...", flush=True)
    tp_test = tabpfn_predict(clf, Xt_lin)
    print(f"  [TabPFN] OOF-subsample acc = {accuracy_score(y[D], tp_oof_D.argmax(1)):.5f}")

    # ---- meta-learner, evaluated on the disjoint subsample D --------------
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    four = lambda I: np.hstack([oof_lgb[I], oof_xgb[I], oof_lr[I], oof_mlp[I]])
    five = lambda I, tp: np.hstack([four(I), tp])

    meta = LogisticRegression(max_iter=2000, n_jobs=-1)
    acc4 = accuracy_score(y[D], cross_val_predict(meta, four(D), y[D], cv=cv,
                                                  method="predict_proba").argmax(1))
    acc5 = accuracy_score(y[D], cross_val_predict(meta, five(D, tp_oof_D), y[D], cv=cv,
                                                  method="predict_proba").argmax(1))

    print("\n" + "=" * 60)
    print(f"(evaluated on the {len(D):,}-row disjoint subsample)")
    print(f"  4-model stack CV (no TabPFN) : {acc4:.5f}")
    print(f"  5-model stack CV (+ TabPFN)  : {acc5:.5f}   delta {acc5-acc4:+.5f}")
    print(f"  LightGBM baseline (full CV)  : 0.96805")
    print("=" * 60)

    # final: train 5-model meta on D, predict full test
    meta.fit(five(D, tp_oof_D), y[D])
    meta_Xt = np.hstack([te_lgb, te_xgb, te_lr, te_mlp, tp_test])
    final = meta.predict(meta_Xt)
    pd.DataFrame({
        "id": pd.read_csv("data/test.csv")["id"],
        "class": [E.INT_TO_CLASS[i] for i in final],
    }).to_csv("submission_stack_tabpfn.csv", index=False)
    print("Saved submission_stack_tabpfn.csv")


if __name__ == "__main__":
    main()
