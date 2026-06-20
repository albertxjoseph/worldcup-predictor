"""'Road to the Cup' — Monte Carlo simulator for the 2026 World Cup.

Plays out the remaining tournament many times using our match models and reports
each team's chance of winning it all. NOT real-time: it reflects results already
in the dataset and is re-run on demand.

Format encoded (FIFA 2026, official):
  * 48 teams, 12 groups of 4 (derived from the real group fixtures in the data).
  * Group tiebreakers: points -> goal difference -> goals for -> head-to-head
    (points/GD/GF among tied) -> random (stands in for fair-play/lots).
  * Round of 32 = 12 group winners + 12 runners-up + 8 best third-placed teams,
    slotted via the official Annex-C R32 skeleton (the eight third-place slots
    are filled by a valid assignment respecting each slot's allowed group set).
  * Knockout draws go to penalties: a near coin-flip with a small edge to the
    higher-Elo side.

Result sampling uses the XGBoost match model's win/draw/loss probabilities (the
headline model). Scorelines for group tiebreakers are drawn from the Dixon-Coles
goals model, conditioned on the sampled result.
"""

import numpy as np
import pandas as pd
import joblib

from model import load_artifacts, build_features, RESULTS_URL
from wc_data import derive_groups, OFFICIAL_GROUPS

# ── Official 2026 bracket skeleton ───────────────────────────────────────────
# Eight Round-of-32 slots filled by a third-placed team: R32 match number,
# the group winner they face, and the set of groups that third may come from.
THIRD_SLOTS = [
    {"match": 74, "winner": "E", "allowed": set("ABCDF")},
    {"match": 77, "winner": "I", "allowed": set("CDFGH")},
    {"match": 79, "winner": "A", "allowed": set("CEFHI")},
    {"match": 80, "winner": "L", "allowed": set("EHIJK")},
    {"match": 81, "winner": "D", "allowed": set("BEFIJ")},
    {"match": 82, "winner": "G", "allowed": set("AEHIJ")},
    {"match": 85, "winner": "B", "allowed": set("EFGIJ")},
    {"match": 87, "winner": "K", "allowed": set("DEIJL")},
]

# R32 pairings. ("1","A") = winner of A, ("2","C") = runner-up of C,
# ("3", match_no) = the third-placed team assigned to that slot.
R32 = {
    73: (("2", "A"), ("2", "B")),
    74: (("1", "E"), ("3", 74)),
    75: (("1", "F"), ("2", "C")),
    76: (("1", "C"), ("2", "F")),
    77: (("1", "I"), ("3", 77)),
    78: (("2", "E"), ("2", "I")),
    79: (("1", "A"), ("3", 79)),
    80: (("1", "L"), ("3", 80)),
    81: (("1", "D"), ("3", 81)),
    82: (("1", "G"), ("3", 82)),
    83: (("2", "K"), ("2", "L")),
    84: (("1", "H"), ("2", "J")),
    85: (("1", "B"), ("3", 85)),
    86: (("1", "J"), ("2", "H")),
    87: (("1", "K"), ("3", 87)),
    88: (("2", "D"), ("2", "G")),
}
R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}
FINAL = (101, 102)


def _bipartite_assign(qualified_groups):
    """Assign each qualifying third-place group to a distinct R32 slot whose
    allowed set contains it (augmenting-path matching). Returns {match_no: group}
    or None if no valid assignment exists."""
    slots = THIRD_SLOTS
    match_of_slot = {}      # slot index -> group
    def try_assign(g, seen):
        for si, slot in enumerate(slots):
            if g in slot["allowed"] and si not in seen:
                seen.add(si)
                if si not in match_of_slot or try_assign(match_of_slot[si], seen):
                    match_of_slot[si] = g
                    return True
        return False
    for g in qualified_groups:
        if not try_assign(g, set()):
            return None
    return {slots[si]["match"]: g for si, g in match_of_slot.items()}


class TournamentSimulator:
    def __init__(self):
        self.ctx, _ = load_artifacts()
        self.ratings = self.ctx["ratings"]
        try:
            self.dc = joblib.load("dc_model.joblib")
        except Exception:
            self.dc = None
        self._load_fixtures()
        self._precompute_probs()
        self._dc_cache = {}

    # ── data ─────────────────────────────────────────────────────────────────
    def _load_fixtures(self):
        df = pd.read_csv(RESULTS_URL)
        df["date"] = pd.to_datetime(df["date"])
        wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-01-01")].copy()

        # Official group letters (so the bracket skeleton is correct), but
        # cross-check membership against the actual fixtures; fall back to the
        # data-derived grouping if they ever disagree.
        derived = derive_groups(wc)
        derived_sets = {frozenset(v) for v in derived.values()}
        official_sets = {frozenset(v) for v in OFFICIAL_GROUPS.values()}
        if derived_sets == official_sets:
            self.groups = {g: list(ts) for g, ts in OFFICIAL_GROUPS.items()}
            self.group_source = "official 2026 draw"
        else:
            self.groups = derived
            self.group_source = "derived from fixtures (official labels unavailable)"
        self.team_group = {t: g for g, ts in self.groups.items() for t in ts}
        self.teams = sorted(self.team_group)
        # group fixtures: (home, away, gh, ga or None)
        self.group_matches = {g: [] for g in self.groups}
        for _, r in wc.iterrows():
            g = self.team_group.get(r["home_team"])
            if g is None or self.team_group.get(r["away_team"]) != g:
                continue   # ignore any non-group rows
            gh = None if pd.isna(r["home_score"]) else int(r["home_score"])
            ga = None if pd.isna(r["away_score"]) else int(r["away_score"])
            self.group_matches[g].append((r["home_team"], r["away_team"], gh, ga))

    def _precompute_probs(self):
        """Symmetric W/D/L for every ordered team pair (neutral, no venue)."""
        ctx = dict(self.ctx)
        ctx["neutral"] = 1
        ctx["match_loc"] = None
        ctx["match_in_host"] = False
        rows, pairs = [], []
        for a in self.teams:
            for b in self.teams:
                if a == b:
                    continue
                rows.append(build_features(a, b, ctx))
                pairs.append((a, b))
        X = pd.DataFrame(rows)[self.ctx["features"]]
        proba = self.ctx["model"].predict_proba(X)
        classes = list(self.ctx["encoder"].classes_)
        hi, di, ai = classes.index("home_win"), classes.index("draw"), classes.index("away_win")
        raw = {}
        for (a, b), p in zip(pairs, proba):
            raw[(a, b)] = (p[hi], p[di], p[ai])
        # symmetrise so the listed-home ordering can't bias a neutral game
        self.wdl = {}
        for a in self.teams:
            for b in self.teams:
                if a == b:
                    continue
                ph1, pd1, pa1 = raw[(a, b)]
                pa2, pd2, ph2 = raw[(b, a)]   # reversed perspective
                aw = (ph1 + ph2) / 2
                dr = (pd1 + pd2) / 2
                bw = (pa1 + pa2) / 2
                s = aw + dr + bw
                self.wdl[(a, b)] = (aw / s, dr / s, bw / s)

    # ── sampling helpers ─────────────────────────────────────────────────────
    def _dc_matrix(self, a, b):
        if self.dc is None:
            return None
        key = (a, b)
        if key not in self._dc_cache:
            self._dc_cache[key] = self.dc.score_matrix(a, b, neutral=1)
        return self._dc_cache[key]

    def _sample_group_score(self, rng, a, b):
        """Sample (gh, ga): result from XGBoost, scoreline from Dixon-Coles
        conditioned on that result (fallback to simple goals if no DC)."""
        aw, dr, bw = self.wdl[(a, b)]
        u = rng.random()
        result = "a" if u < aw else ("d" if u < aw + dr else "b")
        m = self._dc_matrix(a, b)
        if m is None:
            # crude fallback consistent with the result
            if result == "d":
                g = rng.integers(0, 3); return g, g
            hi = rng.integers(1, 4); lo = rng.integers(0, hi)
            return (hi, lo) if result == "a" else (lo, hi)
        n = m.shape[0]
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        if result == "a":
            mask = ii > jj
        elif result == "b":
            mask = ii < jj
        else:
            mask = ii == jj
        w = (m * mask).ravel()
        w = w / w.sum()
        k = rng.choice(w.size, p=w)
        return int(k // n), int(k % n)

    def _knockout_winner(self, rng, a, b):
        aw, dr, bw = self.wdl[(a, b)]
        u = rng.random()
        if u < aw:
            return a
        if u < aw + bw:
            return b
        # draw -> penalties: near coin flip, small edge to higher Elo
        ea = self.ratings.get(a, 1500)
        eb = self.ratings.get(b, 1500)
        pa = min(0.65, max(0.35, 0.5 + 0.0004 * (ea - eb)))
        return a if rng.random() < pa else b

    # ── one full tournament ──────────────────────────────────────────────────
    def _rank_group(self, rng, g):
        teams = self.groups[g]
        pts = {t: 0 for t in teams}
        gf = {t: 0 for t in teams}
        ga = {t: 0 for t in teams}
        played = []   # (home, away, gh, ga)
        for (h, a, sh, sa) in self.group_matches[g]:
            if sh is None:
                sh, sa = self._sample_group_score(rng, h, a)
            played.append((h, a, sh, sa))
            gf[h] += sh; ga[h] += sa; gf[a] += sa; ga[a] += sh
            if sh > sa:
                pts[h] += 3
            elif sh < sa:
                pts[a] += 3
            else:
                pts[h] += 1; pts[a] += 1
        gd = {t: gf[t] - ga[t] for t in teams}

        def h2h(subset):
            p = {t: 0 for t in subset}; d = {t: 0 for t in subset}; f = {t: 0 for t in subset}
            for (h, a, sh, sa) in played:
                if h in subset and a in subset:
                    f[h] += sh; f[a] += sa; d[h] += sh - sa; d[a] += sa - sh
                    if sh > sa: p[h] += 3
                    elif sh < sa: p[a] += 3
                    else: p[h] += 1; p[a] += 1
            return p, d, f

        # overall: points, GD, GF; then head-to-head; then random
        order = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
        out = []
        i = 0
        while i < len(order):
            j = i
            while (j + 1 < len(order)
                   and (pts[order[j+1]], gd[order[j+1]], gf[order[j+1]])
                   == (pts[order[i]], gd[order[i]], gf[order[i]])):
                j += 1
            cluster = order[i:j + 1]
            if len(cluster) > 1:
                p, d, f = h2h(cluster)
                cluster = sorted(cluster, key=lambda t: (p[t], d[t], f[t], rng.random()), reverse=True)
            out.extend(cluster)
            i = j + 1
        return out, pts, gd, gf

    def simulate_once(self, rng):
        winners, runners, thirds = {}, {}, []
        for g in self.groups:
            order, pts, gd, gf = self._rank_group(rng, g)
            winners[g] = order[0]
            runners[g] = order[1]
            thirds.append({"group": g, "team": order[2],
                           "pts": pts[order[2]], "gd": gd[order[2]], "gf": gf[order[2]]})

        # best 8 third-placed teams
        thirds.sort(key=lambda d: (d["pts"], d["gd"], d["gf"], rng.random()), reverse=True)
        top8 = thirds[:8]
        assign = _bipartite_assign([d["group"] for d in top8])
        if assign is None:
            # extremely rare; fall back to slot order
            assign = {THIRD_SLOTS[i]["match"]: top8[i]["group"] for i in range(8)}
        third_team_by_match = {m: next(d["team"] for d in top8 if d["group"] == grp)
                               for m, grp in assign.items()}

        def resolve(slot):
            kind, ref = slot
            if kind == "1":
                return winners[ref]
            if kind == "2":
                return runners[ref]
            return third_team_by_match[ref]

        wins = {}
        for m, (s1, s2) in R32.items():
            wins[m] = self._knockout_winner(rng, resolve(s1), resolve(s2))
        for stage in (R16, QF, SF):
            for m, (m1, m2) in stage.items():
                wins[m] = self._knockout_winner(rng, wins[m1], wins[m2])
        return self._knockout_winner(rng, wins[FINAL[0]], wins[FINAL[1]])

    def run(self, n=5000, seed=0):
        rng = np.random.default_rng(seed)
        champ = {}
        for _ in range(n):
            c = self.simulate_once(rng)
            champ[c] = champ.get(c, 0) + 1
        table = sorted(champ.items(), key=lambda kv: kv[1], reverse=True)
        return [(t, c / n) for t, c in table], n


def run_tracker(n=5000, seed=0):
    sim = TournamentSimulator()
    table, n = sim.run(n=n, seed=seed)
    return table, n


if __name__ == "__main__":
    import time
    t0 = time.time()
    table, n = run_tracker(n=5000)
    print(f"Road to the Cup — {n} simulations ({time.time()-t0:.1f}s)\n")
    print("Top 5 most likely champions:")
    for team, p in table[:5]:
        print(f"  {team:18} {p*100:5.1f}%")
