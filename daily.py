"""Daily fixtures, results, and the running accuracy ledger.

Predictions come from the leakage-safe pre-match features saved in
predictor_model.joblib ("wc_games"), so every prediction is the one we would
have shown BEFORE kickoff and is fully reproducible from the committed model.

The first time a game appears it is frozen into data/predictions_log.csv
(append-only), so the cumulative accuracy never shifts if the model is later
retrained. Actual results are read fresh from the live results CSV.

Accuracy rule (per the project owner): a game is "correct" when the single
most-likely outcome (home win / draw / away win) equals the actual result.
The ledger counts every 2026 World Cup game from 16 Jun 2026 onward.
"""

import os
import numpy as np
import pandas as pd

from model import load_artifacts, RESULTS_URL

LEDGER_START = pd.Timestamp("2026-06-16")
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "predictions_log.csv")
PICK_LABEL = {"home_win": "home win", "draw": "draw", "away_win": "away win"}

# Games whose top outcome clears this probability are flagged "confident pick";
# closer ones are "toss-up". Chosen from the holdout precision/coverage curve
# (see confidence_thresholds.py): ~0.60 calls about half the games at ~76%
# accuracy, a balanced false-alarm / miss tradeoff.
CONFIDENT_THRESHOLD = 0.60

# Feature columns handed to the LLM preview.
PREVIEW_FEATURE_KEYS = [
    "home_elo", "away_elo", "elo_diff", "home_form_gf", "home_form_ga",
    "away_form_gf", "away_form_ga", "home_strength", "away_strength",
    "home_is_host", "away_is_host", "home_support_index", "away_support_index",
]


def build_preview_data(row, dc=None):
    """Assemble the payload generate_preview() expects from a game row.
    Shared by the app and the pre-warm script so they stay identical."""
    data = {
        "home": row["home_team"], "away": row["away_team"],
        "home_win": float(row["p_home"]), "draw": float(row["p_draw"]),
        "away_win": float(row["p_away"]),
        "neutral": 1, "match_date": str(pd.to_datetime(row["date"]).date()),
        "features": {k: row[k] for k in PREVIEW_FEATURE_KEYS if k in row},
    }
    if dc is not None:
        d = dc.predict(row["home_team"], row["away_team"], neutral=1)
        data["dc"] = d
        data["dc_scoreline"] = d["scoreline"]
    return data


def actual_outcome(hs, as_):
    if pd.isna(hs) or pd.isna(as_):
        return None
    if hs > as_:
        return "home_win"
    if hs < as_:
        return "away_win"
    return "draw"


def _live_wc():
    df = pd.read_csv(RESULTS_URL)
    df["date"] = pd.to_datetime(df["date"])
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-01-01")].copy()
    wc["outcome"] = [actual_outcome(h, a) for h, a in zip(wc["home_score"], wc["away_score"])]
    return wc


def predicted_games(bundle):
    """Per-game leakage-safe probabilities, pick, and the underlying features."""
    wc = pd.DataFrame(bundle["wc_games"]).copy()
    wc["date"] = pd.to_datetime(wc["date"])
    model, le = bundle["model"], bundle["encoder"]
    proba = model.predict_proba(wc[bundle["features"]])
    cls = list(le.classes_)
    P = {c: proba[:, i] for i, c in enumerate(cls)}
    wc["p_home"], wc["p_draw"], wc["p_away"] = P["home_win"], P["draw"], P["away_win"]
    stacked = np.vstack([wc["p_home"], wc["p_draw"], wc["p_away"]]).T
    wc["pick"] = np.array(["home_win", "draw", "away_win"])[stacked.argmax(axis=1)]
    return wc


def update_log(preds, freeze_until):
    """Freeze each game's first-seen prediction. Append-only; never overwrites.

    Only games already played or on the current matchday (date <= freeze_until)
    are frozen. Fixtures further out are left until they become the live matchday,
    so each prediction is frozen using that game's true pre-kickoff ratings rather
    than today's stale ones."""
    cols = ["date", "home_team", "away_team", "p_home", "p_draw", "p_away", "pick"]
    keep = preds[(preds["date"] >= LEDGER_START) & (preds["date"] <= freeze_until)][cols]
    if os.path.exists(LOG_PATH):
        log = pd.read_csv(LOG_PATH, parse_dates=["date"])
    else:
        log = pd.DataFrame(columns=cols)
    if len(log):
        seen = set(zip(log["date"].dt.strftime("%Y-%m-%d"), log["home_team"], log["away_team"]))
    else:
        seen = set()
    new = [r for _, r in keep.iterrows()
           if (r["date"].strftime("%Y-%m-%d"), r["home_team"], r["away_team"]) not in seen]
    if new:
        log = pd.concat([log, pd.DataFrame(new)], ignore_index=True)
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        log.to_csv(LOG_PATH, index=False)
    return log


def build_tables():
    """Everything the UI needs: today's fixtures, yesterday's results, and the
    cumulative accuracy — all keyed on schedule-anchored day selection."""
    _, bundle = load_artifacts()
    preds = predicted_games(bundle)
    live = _live_wc()

    # schedule-anchored: "today" = next matchday with unplayed games,
    # "yesterday" = most recent matchday that has results.
    played, unplayed = live[live["home_score"].notna()], live[live["home_score"].isna()]
    today = unplayed["date"].min() if len(unplayed) else None
    yesterday = played["date"].max() if len(played) else None

    # freeze predictions for everything up to and including today's matchday
    freeze_until = today if today is not None else live["date"].max()
    log = update_log(preds, freeze_until)        # frozen predictions (accuracy source)

    # Frozen predictions + live actuals + the pre-match features for display.
    feat_cols = [c for c in bundle["features"]] + ["home_team", "away_team", "date"]
    feats = preds[feat_cols]
    L = (log
         .merge(feats, on=["date", "home_team", "away_team"], how="left")
         .merge(live[["date", "home_team", "away_team", "home_score", "away_score",
                      "outcome", "city"]],
                on=["date", "home_team", "away_team"], how="left"))
    L["played"] = L["home_score"].notna()
    L["correct"] = np.where(L["played"], L["pick"] == L["outcome"], np.nan)

    today_games = L[L["date"] == today].copy() if today is not None else L.iloc[0:0]
    yest_games = L[L["date"] == yesterday].copy() if yesterday is not None else L.iloc[0:0]

    done = L[(L["date"] >= LEDGER_START) & L["played"]]
    accuracy = {"correct": int(done["correct"].sum()), "total": int(len(done)),
                "pct": (100.0 * done["correct"].mean() if len(done) else None)}

    return {"bundle": bundle, "today": today, "yesterday": yesterday,
            "today_games": today_games, "yesterday_games": yest_games,
            "accuracy": accuracy, "all": L}


if __name__ == "__main__":
    t = build_tables()
    acc = t["accuracy"]
    print(f"Today ({t['today'].date() if t['today'] is not None else '—'}): "
          f"{len(t['today_games'])} games")
    print(f"Yesterday ({t['yesterday'].date() if t['yesterday'] is not None else '—'}): "
          f"{len(t['yesterday_games'])} games")
    print(f"Cumulative accuracy since {LEDGER_START.date()}: "
          f"{acc['correct']}/{acc['total']}"
          + (f" = {acc['pct']:.0f}%" if acc["pct"] is not None else ""))
    print("\nYesterday — prediction vs actual:")
    for _, r in t["yesterday_games"].iterrows():
        mark = "✓" if r["correct"] else "✗"
        print(f"  {mark} {r['home_team']} {int(r['home_score'])}-{int(r['away_score'])} "
              f"{r['away_team']}  | picked {PICK_LABEL[r['pick']]} "
              f"({max(r['p_home'], r['p_draw'], r['p_away'])*100:.0f}%)")
