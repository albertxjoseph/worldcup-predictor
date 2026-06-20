"""World Cup match-result model.

Loads international results, builds leakage-safe Elo + form + context features,
trains an XGBoost 3-way classifier, and saves everything to
predictor_model.joblib. The feature logic lives in ONE place (build_features)
so training and prediction can never drift apart.

Run as a script to (re)train:   python3 model.py
Import for the UI:              from model import load_artifacts, predict, build_features
"""

import os
import pandas as pd
import numpy as np

from wc_data import HOSTS, load_country_coords, haversine, VENUES

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# The model's feature columns. New features get appended here AND produced in
# build_features() — nowhere else.
FEATURES = [
    "elo_diff", "home_elo", "away_elo", "neutral",
    "home_form_gf", "home_form_ga", "away_form_gf", "away_form_ga",
    "home_is_host", "away_is_host",
    "home_strength", "away_strength", "strength_diff",
    "home_support_index", "away_support_index",
]

# Country-centroid coordinates, used as a travel-proximity proxy for crowd
# support. This is a proxy, NOT real ticketing/attendance data.
COORDS = load_country_coords()
SUPPORT_FALLBACK_KM = 8000.0   # used when a team's coords are unknown
SUPPORT_SCALE_KM = 3000.0      # distance at which support is roughly halved


def support_index(team, match_loc, match_in_host):
    """Scaled inverse distance from a team's country to the match location,
    boosted when the team is a 2026 host playing in a host country.
    Returns ~1.0 for a home/very-near team down toward 0 for a distant one."""
    d = haversine(COORDS.get(team), match_loc) if match_loc else None
    if d is None:
        d = SUPPORT_FALLBACK_KM
    base = 1.0 / (1.0 + d / SUPPORT_SCALE_KM)
    if match_in_host and team in HOSTS:
        base = min(1.0, base * 1.25)
    return base


def load_squad_strength():
    """team -> 0..100 squad strength, plus the median used for missing teams."""
    df = pd.read_csv(os.path.join(HERE, "data", "squad_strength.csv"))
    table = dict(zip(df["team"], df["strength"]))
    return table, float(df["strength"].median())


SQUAD, SQUAD_MEDIAN = load_squad_strength()


def strength_of(team):
    return SQUAD.get(team, SQUAD_MEDIAN)


def expected(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def mov_factor(goal_diff):
    """Margin-of-victory multiplier (World-Football goal-difference factor):
    bigger wins move Elo more, draws/one-goal games update normally. Measured to
    improve accuracy and log loss while keeping calibration tight."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def form(team, recent):
    games = recent.get(team, [])
    if not games:
        return 1.0, 1.0
    goals_for = sum(g[0] for g in games) / len(games)
    goals_against = sum(g[1] for g in games) / len(games)
    return goals_for, goals_against


def build_features(home, away, context):
    """The single source of truth for one match's features.

    context carries:
      ratings  - dict team -> Elo (pre-match state)
      recent   - dict team -> last few (gf, ga) tuples
      neutral  - 1/0 neutral-venue flag (default 1)
    Used identically in the training loop and at prediction time.
    """
    ratings = context["ratings"]
    recent = context["recent"]

    rh = ratings.get(home, 1500)
    ra = ratings.get(away, 1500)
    h_gf, h_ga = form(home, recent)
    a_gf, a_ga = form(away, recent)

    h_str = strength_of(home)
    a_str = strength_of(away)

    match_loc = context.get("match_loc")
    match_in_host = context.get("match_in_host", False)
    h_sup = support_index(home, match_loc, match_in_host)
    a_sup = support_index(away, match_loc, match_in_host)

    return {
        "elo_diff": rh - ra,
        "home_elo": rh,
        "away_elo": ra,
        "neutral": context.get("neutral", 1),
        "home_form_gf": h_gf, "home_form_ga": h_ga,
        "away_form_gf": a_gf, "away_form_ga": a_ga,
        "home_is_host": 1 if home in HOSTS else 0,
        "away_is_host": 1 if away in HOSTS else 0,
        "home_strength": h_str, "away_strength": a_str,
        "strength_diff": h_str - a_str,
        "home_support_index": h_sup, "away_support_index": a_sup,
    }


def predict(home, away, context):
    """Return a dict of probabilities + the underlying features. UI-friendly.

    context additionally carries the trained `model`, `encoder`, and `features`.
    """
    feats = build_features(home, away, context)
    model = context["model"]
    le = context["encoder"]
    cols = context["features"]
    row = pd.DataFrame([feats])[cols]
    p = model.predict_proba(row)[0]
    probs = dict(zip(le.classes_, p))
    return {
        "home": home, "away": away,
        "home_win": float(probs["home_win"]),
        "draw": float(probs["draw"]),
        "away_win": float(probs["away_win"]),
        "features": feats,
    }


def predict_print(home, away, context):
    """Thin terminal wrapper around predict()."""
    out = predict(home, away, context)
    print(f"{home} win : {round(out['home_win'] * 100)}%")
    print(f"draw     : {round(out['draw'] * 100)}%")
    print(f"{away} win : {round(out['away_win'] * 100)}%")
    return out


def load_artifacts(path="predictor_model.joblib"):
    """Load the saved bundle and return a ready-to-use prediction context."""
    import joblib
    bundle = joblib.load(path)
    ctx = {
        "ratings": bundle["ratings"],
        "recent": bundle["recent"],
        "model": bundle["model"],
        "encoder": bundle["encoder"],
        "features": bundle["features"],
        "neutral": 1,
    }
    return ctx, bundle


# ─────────────────────────────────────────────────────────────────────────────
# Training pipeline (only runs when executed directly)
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset():
    """Walk the full match history once, producing the leakage-safe feature
    table plus final Elo/form state and the 2026-WC ledger rows. Shared by
    train() and the experiment harness."""
    df_all = pd.read_csv(RESULTS_URL)
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all["neutral"] = df_all["neutral"].astype(int)
    df = (df_all.dropna(subset=["home_score", "away_score"])
                .sort_values("date").reset_index(drop=True))
    print("matches loaded:", len(df))

    ratings, recent = {}, {}
    context = {"ratings": ratings, "recent": recent}

    WC_START = pd.Timestamp("2026-01-01")
    wc_rows = []   # leakage-safe pre-match features for every 2026 WC game

    rows = []
    for g in df.itertuples():
        context["neutral"] = g.neutral
        context["match_loc"] = COORDS.get(g.country)
        context["match_in_host"] = g.country in HOSTS
        feats = build_features(g.home_team, g.away_team, context)

        if g.home_score > g.away_score:
            result, score_home = "home_win", 1
        elif g.home_score < g.away_score:
            result, score_home = "away_win", 0
        else:
            result, score_home = "draw", 0.5

        rows.append({**feats, "result": result})

        # record pre-match features for played 2026 WC games (for the ledger)
        if g.tournament == "FIFA World Cup" and g.date >= WC_START:
            wc_rows.append({**feats, "date": g.date,
                            "home_team": g.home_team, "away_team": g.away_team,
                            "home_score": g.home_score, "away_score": g.away_score})

        # update Elo for next time (margin-of-victory weighted)
        rh, ra = ratings.get(g.home_team, 1500), ratings.get(g.away_team, 1500)
        home_boost = 0 if g.neutral else 65
        e = expected(rh + home_boost, ra)
        change = 30 * mov_factor(g.home_score - g.away_score) * (score_home - e)
        ratings[g.home_team] = rh + change
        ratings[g.away_team] = ra - change

        recent.setdefault(g.home_team, []).append((g.home_score, g.away_score))
        recent.setdefault(g.away_team, []).append((g.away_score, g.home_score))
        recent[g.home_team] = recent[g.home_team][-5:]
        recent[g.away_team] = recent[g.away_team][-5:]

    # unplayed 2026 WC fixtures: pre-match features from the final walk-forward
    # state, derived identically to the played branch so a game's prediction is
    # the same whether captured before or after it is played.
    fctx = {"ratings": ratings, "recent": recent}
    fut = df_all[(df_all["tournament"] == "FIFA World Cup")
                 & (df_all["date"] >= WC_START)
                 & (df_all["home_score"].isna())]
    for r in fut.itertuples():
        fctx["neutral"] = int(r.neutral)
        fctx["match_loc"] = COORDS.get(r.country)
        fctx["match_in_host"] = r.country in HOSTS
        feats = build_features(r.home_team, r.away_team, fctx)
        wc_rows.append({**feats, "date": r.date,
                        "home_team": r.home_team, "away_team": r.away_team,
                        "home_score": np.nan, "away_score": np.nan})

    data = pd.DataFrame(rows)
    data["date"] = df["date"].values
    return data, ratings, recent, wc_rows


def train():
    data, ratings, recent, wc_rows = build_dataset()
    print(data.tail())

    split = int(len(data) * 0.85)
    train_df, test_df = data.iloc[:split], data.iloc[split:]
    print("training on:", len(train_df), "matches")
    print("testing on:", len(test_df), "matches")

    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["result"])
    y_test = le.transform(test_df["result"])

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", eval_metric="mlogloss",
    )
    model.fit(train_df[FEATURES], y_train)
    print("model trained")

    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.linear_model import LogisticRegression

    proba = model.predict_proba(test_df[FEATURES])
    preds = model.predict(test_df[FEATURES])
    print("model accuracy :", round(accuracy_score(y_test, preds), 3))
    print("model log loss :", round(log_loss(y_test, proba), 3))

    base = LogisticRegression(max_iter=1000)
    base.fit(train_df[["elo_diff"]], y_train)
    base_proba = base.predict_proba(test_df[["elo_diff"]])
    print("elo-only log loss:", round(log_loss(y_test, base_proba), 3))

    # quick sanity prediction
    ctx = {"ratings": ratings, "recent": recent, "model": model,
           "encoder": le, "features": FEATURES, "neutral": 1}
    print("\nSpain vs Cape Verde (neutral):")
    predict_print("Spain", "Cape Verde", ctx)

    import joblib
    joblib.dump(
        {"model": model, "encoder": le, "ratings": ratings,
         "recent": recent, "features": FEATURES, "wc_games": wc_rows},
        "predictor_model.joblib",
    )
    print(f"saved ({len(wc_rows)} WC 2026 games logged)")

    # calibration chart
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    home_idx = list(le.classes_).index("home_win")
    actual_home_win = (y_test == home_idx).astype(int)
    frac, mean_pred = calibration_curve(actual_home_win, proba[:, home_idx], n_bins=10)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    plt.plot(mean_pred, frac, "o-", label="your model")
    plt.xlabel("predicted chance")
    plt.ylabel("how often it actually happened")
    plt.title("Is the model honest about its confidence?")
    plt.legend()
    plt.savefig("calibration.png", dpi=130)
    print("chart saved as calibration.png")


if __name__ == "__main__":
    train()
