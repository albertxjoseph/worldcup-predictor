"""Test whether a better Elo earns higher, still-honest confidence.

Improvements tried vs the current Elo (fixed K=30, home boost 65, no
margin-of-victory):
  * margin-of-victory via the World-Football goal-difference factor G
    (G=1 for 0-1 goals, 1.5 for 2, (11+gd)/8 beyond) — big wins move ratings
    more, draws still update normally;
  * tuned K (learning rate) and home-advantage boost.

We rebuild the leakage-safe feature table under each Elo, retrain the same
XGBoost, and compare accuracy / log loss / calibration. Keep a change only if
log loss drops AND calibration (ECE) stays tight.
"""

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from model import build_features, expected, FEATURES, RESULTS_URL, COORDS, HOSTS
from experiment import confidence_stats, show


def gfactor(gd, use_mov):
    if not use_mov:
        return 1.0
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def build_features_with_elo(df, k=30, home_boost=65, use_mov=False):
    ratings, recent = {}, {}
    ctx = {"ratings": ratings, "recent": recent}
    rows = []
    for g in df.itertuples():
        ctx["neutral"] = g.neutral
        ctx["match_loc"] = COORDS.get(g.country)
        ctx["match_in_host"] = g.country in HOSTS
        feats = build_features(g.home_team, g.away_team, ctx)

        if g.home_score > g.away_score:
            result, score_home = "home_win", 1.0
        elif g.home_score < g.away_score:
            result, score_home = "away_win", 0.0
        else:
            result, score_home = "draw", 0.5
        rows.append({**feats, "result": result})

        rh = ratings.get(g.home_team, 1500)
        ra = ratings.get(g.away_team, 1500)
        hb = 0 if g.neutral else home_boost
        e = expected(rh + hb, ra)
        gd = abs(g.home_score - g.away_score)
        change = k * gfactor(gd, use_mov) * (score_home - e)
        ratings[g.home_team] = rh + change
        ratings[g.away_team] = ra - change

        recent.setdefault(g.home_team, []).append((g.home_score, g.away_score))
        recent.setdefault(g.away_team, []).append((g.away_score, g.home_score))
        recent[g.home_team] = recent[g.home_team][-5:]
        recent[g.away_team] = recent[g.away_team][-5:]

    data = pd.DataFrame(rows)
    data["date"] = df["date"].values
    return data


def evaluate(data):
    split = int(len(data) * 0.85)
    tr, te = data.iloc[:split], data.iloc[split:]
    le = LabelEncoder()
    ytr, yte = le.fit_transform(tr["result"]), le.transform(te["result"])
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      objective="multi:softprob", eval_metric="mlogloss")
    m.fit(tr[FEATURES], ytr)
    return confidence_stats(m.predict_proba(te[FEATURES]), yte)


def main():
    df = pd.read_csv(RESULTS_URL)
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(int)
    df = df.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)
    print(f"matches: {len(df)}\n")

    configs = {
        "current (K30, hb65, no MOV)": dict(k=30, home_boost=65, use_mov=False),
        "MOV (K30, hb65)":             dict(k=30, home_boost=65, use_mov=True),
        "MOV + K40":                   dict(k=40, home_boost=65, use_mov=True),
        "MOV + K40 + hb80":            dict(k=40, home_boost=80, use_mov=True),
        "MOV + K50 + hb80":            dict(k=50, home_boost=80, use_mov=True),
    }
    for name, params in configs.items():
        data = build_features_with_elo(df, **params)
        show(name, evaluate(data))


if __name__ == "__main__":
    main()
