"""Build data/squad_strength.csv from the FIFA 22 complete player dataset.

Strength = mean `overall` of each nation's top-23 rated players (a national
squad size), min-max scaled to 0-100 across nations with a real player pool.
Team names are rewritten to the results-dataset spelling.

Source: FIFA 22 complete player dataset (a single 2021 snapshot, results-
independent). This is NOT time-varying — a known simplification.
"""

import pandas as pd

SRC = "data/players_22.csv"
OUT = "data/squad_strength.csv"

# FIFA nationality spelling -> results-dataset spelling
RENAME = {
    "Cape Verde Islands": "Cape Verde",
    "Curacao": "Curaçao",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "China PR": "China",
    "United States": "United States",
}

df = pd.read_csv(SRC, low_memory=False)
df = df[["nationality_name", "overall"]].dropna()

rows = []
for nat, grp in df.groupby("nationality_name"):
    top = grp["overall"].sort_values(ascending=False).head(23)
    rows.append({"nat": nat, "raw": top.mean(), "n": len(grp)})

agg = pd.DataFrame(rows)

# Scale using nations with a credible pool (>=15 players) as the reference range.
ref = agg[agg["n"] >= 15]["raw"]
lo, hi = ref.min(), ref.max()
agg["strength"] = ((agg["raw"] - lo) / (hi - lo) * 100).clip(0, 100).round(1)

agg["team"] = agg["nat"].replace(RENAME)
out = agg[["team", "strength"]].sort_values("strength", ascending=False)
out.to_csv(OUT, index=False)

print(f"wrote {OUT}: {len(out)} nations")
print("top 10:")
print(out.head(10).to_string(index=False))
print("\nselected WC teams:")
for t in ["Spain", "France", "Brazil", "United States", "Cape Verde",
          "South Korea", "Ivory Coast", "DR Congo", "Curaçao", "Qatar"]:
    r = out[out["team"] == t]
    print(f"  {t:16} {r['strength'].iloc[0] if len(r) else 'MISSING (-> median fill)'}")
