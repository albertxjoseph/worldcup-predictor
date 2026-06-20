"""Pre-generate LLM match previews ahead of time, in one batch.

Run this right after the day's fixtures/results update (or from cron) so previews
are already in the disk cache before any visitor loads the app — nobody waits,
and you control exactly when the API is billed. Already-cached matchups are
skipped, so re-running is free.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python3 prewarm_previews.py                # today's matchday (schedule-anchored)
  python3 prewarm_previews.py 2026-06-19     # a specific date
"""

import sys
import joblib
import pandas as pd

from model import load_artifacts
from daily import predicted_games, build_preview_data, build_tables
from preview import generate_preview, is_cached


def main(date_arg=None):
    _, bundle = load_artifacts()
    try:
        dc = joblib.load("dc_model.joblib")
    except Exception:
        dc = None

    preds = predicted_games(bundle)
    target = pd.Timestamp(date_arg) if date_arg else build_tables()["today"]
    if target is None:
        print("No upcoming matchday found.")
        return 0

    rows = preds[preds["date"] == target]
    if rows.empty:
        print(f"No 2026 World Cup games scheduled on {target.date()}.")
        return 0

    print(f"Pre-warming previews for {target.date()} ({len(rows)} games)…")
    generated = skipped = failed = 0
    for _, row in rows.iterrows():
        home, away = row["home_team"], row["away_team"]
        data = build_preview_data(row, dc)
        if is_cached(home, away, data):
            print(f"  • {home} vs {away}: already cached, skipped")
            skipped += 1
            continue
        try:
            generate_preview(home, away, data)
            print(f"  ✓ {home} vs {away}: generated and cached")
            generated += 1
        except RuntimeError as e:
            print(f"  ✗ {home} vs {away}: {str(e).splitlines()[0]}")
            failed += 1

    print(f"\nDone: {generated} generated, {skipped} cached, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
