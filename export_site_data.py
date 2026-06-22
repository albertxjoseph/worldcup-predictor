"""Export a single data.json snapshot for the Next.js frontend.

Runs the existing models/ledger/simulator (and bakes in the LLM previews at
export time, so the API key never touches the frontend), then writes everything
the site needs to web/public/data.json.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...   # optional; enables previews
  python3 export_site_data.py
"""

import json
import os
from datetime import datetime, timezone

import joblib

from daily import build_tables, build_preview_data, PICK_LABEL, LEDGER_START, CONFIDENT_THRESHOLD

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "src", "data.json")
TOP_N_TITLE = 8


def _conf(row):
    return float(max(row["p_home"], row["p_draw"], row["p_away"]))


def _features(row):
    keys = ["home_elo", "away_elo", "home_form_gf", "home_form_ga",
            "away_form_gf", "away_form_ga", "home_strength", "away_strength",
            "home_is_host", "away_is_host", "home_support_index", "away_support_index"]
    return {k: (float(row[k]) if k in row else None) for k in keys}


def main():
    tables = build_tables()
    try:
        dc = joblib.load("dc_model.joblib")
    except Exception:
        dc = None

    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if have_key:
        from preview import generate_preview

    # today's games
    today_games = []
    for _, row in tables["today_games"].iterrows():
        home, away = row["home_team"], row["away_team"]
        g = {
            "home": home, "away": away,
            "city": row.get("city", ""),
            "date": str(row["date"])[:10],
            "p_home": float(row["p_home"]), "p_draw": float(row["p_draw"]),
            "p_away": float(row["p_away"]),
            "confidence": _conf(row),
            "confident": _conf(row) >= CONFIDENT_THRESHOLD,
            "pick": row["pick"], "pick_label": PICK_LABEL[row["pick"]],
            "features": _features(row),
            "dc": None, "preview": None,
        }
        if dc is not None:
            d = dc.predict(home, away, neutral=1)
            g["dc"] = {"scoreline": [int(d["scoreline"][0]), int(d["scoreline"][1])],
                       "xg_home": round(d["exp_home_goals"], 2),
                       "xg_away": round(d["exp_away_goals"], 2),
                       "home": float(d["home_win"]), "draw": float(d["draw"]),
                       "away": float(d["away_win"])}
        if have_key:
            try:
                g["preview"] = generate_preview(home, away, build_preview_data(row, dc))
            except Exception as e:
                print(f"  preview failed for {home} vs {away}: {str(e).splitlines()[0]}")
        today_games.append(g)

    # yesterday's results
    yesterday_games = []
    for _, row in tables["yesterday_games"].iterrows():
        yesterday_games.append({
            "home": row["home_team"], "away": row["away_team"],
            "home_score": int(row["home_score"]), "away_score": int(row["away_score"]),
            "pick": row["pick"], "pick_label": PICK_LABEL[row["pick"]],
            "confidence": _conf(row),
            "confident": _conf(row) >= CONFIDENT_THRESHOLD,
            "correct": bool(row["correct"]),
        })

    # title odds
    from sim import run_tracker
    table, n_sims = run_tracker(n=5000, seed=0)
    title_odds = [{"rank": i + 1, "team": t, "pct": round(p * 100, 1)}
                  for i, (t, p) in enumerate(table[:TOP_N_TITLE])]

    acc = tables["accuracy"]
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ledger_start": str(LEDGER_START.date()),
        "confident_threshold": CONFIDENT_THRESHOLD,
        "accuracy": {"pct": acc["pct"], "correct": acc["correct"], "total": acc["total"]},
        "today": {"date": str(tables["today"].date()) if tables["today"] is not None else None,
                  "games": today_games},
        "yesterday": {"date": str(tables["yesterday"].date()) if tables["yesterday"] is not None else None,
                      "games": yesterday_games},
        "title_odds": title_odds,
        "meta": {"matches": 49000, "teams": 48, "sims": n_sims, "previews": have_key},
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {OUT}")
    print(f"  today: {len(today_games)} games | yesterday: {len(yesterday_games)} | "
          f"accuracy {acc['correct']}/{acc['total']} | title odds top {len(title_odds)} | "
          f"previews: {'on' if have_key else 'off (no key)'}")


if __name__ == "__main__":
    main()
