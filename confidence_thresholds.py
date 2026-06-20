"""Confidence-threshold / FP-vs-FN tradeoff analysis.

Two views on the holdout:

A) Commit threshold — only "call" a game when the top outcome's probability
   clears a cutoff; otherwise treat it as a toss-up. Raising the cutoff trims
   false positives (wrong confident calls) at the cost of coverage (more games
   we decline to call = false negatives / abstentions).

B) Per-outcome threshold — treat one outcome (e.g. "home win") as the positive
   class and sweep its probability cutoff, reporting precision / recall and the
   false-positive / false-negative counts directly.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from model import build_dataset, FEATURES


def fit_holdout():
    data, *_ = build_dataset()
    split = int(len(data) * 0.85)
    tr, te = data.iloc[:split], data.iloc[split:]
    le = LabelEncoder()
    ytr, yte = le.fit_transform(tr["result"]), le.transform(te["result"])
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      objective="multi:softprob", eval_metric="mlogloss")
    m.fit(tr[FEATURES], ytr)
    return m.predict_proba(te[FEATURES]), yte, list(le.classes_)


def commit_table(proba, y):
    top, pred = proba.max(1), proba.argmax(1)
    correct = (pred == y)
    print("A) Commit threshold (call the top pick only above the cutoff)\n")
    print(f"  {'cutoff':>7} {'coverage':>9} {'calls':>7} {'accuracy':>9} {'wrong calls':>12}")
    for t in [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]:
        m = top >= t
        n = int(m.sum())
        cov = m.mean()
        acc = correct[m].mean() if n else float("nan")
        wrong = int((~correct[m]).sum())
        print(f"  {t:>7.2f} {cov:>8.0%} {n:>7d} {acc:>8.1%} {wrong:>12d}")
    print(f"\n  (total games: {len(y)})\n")


def per_class_table(proba, y, classes, cls="home_win"):
    k = classes.index(cls)
    p = proba[:, k]
    pos = (y == k)
    base = pos.mean()
    print(f"B) Per-outcome threshold — positive class = '{cls}' "
          f"(base rate {base:.0%})\n")
    print(f"  {'cutoff':>7} {'precision':>9} {'recall':>7} {'FP':>6} {'FN':>6} {'FPR':>6}")
    for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        pp = p >= t
        tp = int((pp & pos).sum())
        fp = int((pp & ~pos).sum())
        fn = int((~pp & pos).sum())
        tn = int((~pp & ~pos).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        print(f"  {t:>7.2f} {prec:>8.0%} {rec:>7.0%} {fp:>6d} {fn:>6d} {fpr:>6.0%}")
    print()


if __name__ == "__main__":
    proba, y, classes = fit_holdout()
    print()
    commit_table(proba, y)
    per_class_table(proba, y, classes, "home_win")
    per_class_table(proba, y, classes, "away_win")
