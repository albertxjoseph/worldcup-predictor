"""Diagnose the model's confidence/accuracy and test ways to improve them.

Key idea: "more confident" is only good if it's *earned* — i.e. the sharper
probabilities are still calibrated (when it says 70%, it happens ~70% of the
time) and log loss / accuracy don't get worse. We measure that, not just the
average confidence.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from model import build_dataset, FEATURES


def confidence_stats(proba, y):
    top = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(int)
    # expected calibration error on the top class (10 bins)
    ece, bins = 0.0, np.linspace(0, 1, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (top >= lo) & (top < hi)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - top[m].mean())
    # one-vs-rest Brier averaged over the 3 classes
    Y = np.eye(proba.shape[1])[y]
    brier = np.mean([brier_score_loss(Y[:, k], proba[:, k]) for k in range(proba.shape[1])])
    return {
        "acc": accuracy_score(y, pred),
        "logloss": log_loss(y, proba),
        "brier": brier,
        "mean_conf": top.mean(),
        "share>50%": (top > 0.5).mean(),
        "share>65%": (top > 0.65).mean(),
        "acc@conf>65%": correct[top > 0.65].mean() if (top > 0.65).any() else float("nan"),
        "ece": ece,
    }


def show(name, s):
    print(f"{name:28} acc={s['acc']:.3f}  logloss={s['logloss']:.3f}  brier={s['brier']:.3f}  "
          f"meanConf={s['mean_conf']:.3f}  >50%={s['share>50%']:.2f}  >65%={s['share>65%']:.2f}  "
          f"acc@>65%={s['acc@conf>65%']:.3f}  ECE={s['ece']:.3f}")


def main():
    data, *_ = build_dataset()
    split = int(len(data) * 0.85)
    tr, te = data.iloc[:split], data.iloc[split:]
    le = LabelEncoder()
    ytr, yte = le.fit_transform(tr["result"]), le.transform(te["result"])
    Xtr, Xte = tr[FEATURES], te[FEATURES]

    base_rate = pd.Series(yte).value_counts(normalize=True).max()
    print(f"\ntest games: {len(te)}  |  always-predict-most-common accuracy: {base_rate:.3f}")
    print("draw share of results:", round((te['result'] == 'draw').mean(), 3), "\n")

    configs = {
        "current (depth4, 300, lr.05)": dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                                             subsample=0.8, colsample_bytree=0.8),
        "deeper (depth6, 600, lr.03)": dict(n_estimators=600, max_depth=6, learning_rate=0.03,
                                            subsample=0.8, colsample_bytree=0.8),
        "regularised (depth5,500,reg)": dict(n_estimators=500, max_depth=5, learning_rate=0.04,
                                             subsample=0.8, colsample_bytree=0.8,
                                             min_child_weight=5, gamma=0.5, reg_lambda=2.0),
        "shallow+slow (depth3,800,lr.02)": dict(n_estimators=800, max_depth=3, learning_rate=0.02,
                                                subsample=0.8, colsample_bytree=0.8),
    }

    results = {}
    for name, params in configs.items():
        m = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **params)
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)
        s = confidence_stats(p, yte)
        results[name] = (m, p, s)
        show(name, s)

    # Temperature scaling on the best-logloss model: sharpen/soften probabilities
    # by p^(1/T) and renormalise, picking T on the test set just to SHOW the
    # calibration trade-off (not a deployment choice).
    best = min(results, key=lambda k: results[k][2]["logloss"])
    _, p, _ = results[best]
    print(f"\ntemperature sweep on '{best}' (T<1 = more confident):")
    logits = np.log(np.clip(p, 1e-9, None))
    for T in [0.6, 0.8, 1.0, 1.25, 1.5]:
        q = np.exp(logits / T)
        q /= q.sum(axis=1, keepdims=True)
        show(f"  T={T}", confidence_stats(q, yte))


if __name__ == "__main__":
    main()
