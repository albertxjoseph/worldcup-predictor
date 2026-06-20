"""Dixon-Coles bivariate-Poisson goals model (self-contained).

Why not the `penaltyblog` package: on this machine (numpy 1.26, macOS/LibreSSL)
penaltyblog unconditionally imports a Bayesian model built on aesara/pymc, which
fails to import (`np.__config__.get_info` was removed in numpy 1.26) and would
also break a Streamlit Cloud deploy. So we implement the *same* Dixon-Coles
model directly with scipy — attack/defence strengths, home advantage, the
low-score dependence parameter rho, and exponential time-decay (recency)
weighting, exactly as in Dixon & Coles (1997).

Outputs per fixture: an expected scoreline and an INDEPENDENT set of
win/draw/loss probabilities (from the full score matrix), kept separate from the
XGBoost model so the two can be compared rather than blended.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

MAX_GOALS = 10  # truncate the score matrix at 10-10


def _log_pois(k, lam):
    return k * np.log(lam) - lam - gammaln(k + 1)


def _tau(hs, as_, lam, mu, rho):
    """Dixon-Coles low-score correction (vectorised)."""
    t = np.ones_like(lam)
    m00 = (hs == 0) & (as_ == 0)
    m01 = (hs == 0) & (as_ == 1)
    m10 = (hs == 1) & (as_ == 0)
    m11 = (hs == 1) & (as_ == 1)
    t = np.where(m00, 1 - lam * mu * rho, t)
    t = np.where(m01, 1 + lam * rho, t)
    t = np.where(m10, 1 + mu * rho, t)
    t = np.where(m11, 1 - rho, t)
    return t


class DixonColes:
    def __init__(self, half_life_days=730.0, max_goals=MAX_GOALS):
        self.half_life_days = half_life_days
        self.max_goals = max_goals
        self.teams = []
        self.idx = {}
        self.attack = {}
        self.defence = {}
        self.home_adv = 0.0
        self.rho = 0.0

    def fit(self, df, since_years=12, min_games=8):
        """df needs: date, home_team, away_team, home_score, away_score, neutral."""
        df = df.dropna(subset=["home_score", "away_score"]).copy()
        df["date"] = pd.to_datetime(df["date"])
        cutoff = df["date"].max() - pd.Timedelta(days=365 * since_years)
        df = df[df["date"] >= cutoff]

        # keep teams with enough games for a stable estimate
        counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        keep = set(counts[counts >= min_games].index)
        df = df[df["home_team"].isin(keep) & df["away_team"].isin(keep)]

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        T = len(self.teams)

        hi = df["home_team"].map(self.idx).to_numpy()
        ai = df["away_team"].map(self.idx).to_numpy()
        hs = df["home_score"].to_numpy(float)
        as_ = df["away_score"].to_numpy(float)
        nonneutral = (~df["neutral"].astype(bool)).to_numpy(float)

        # exponential time-decay weights (recency)
        age_days = (df["date"].max() - df["date"]).dt.days.to_numpy(float)
        w = 0.5 ** (age_days / self.half_life_days)

        def unpack(p):
            att = p[:T]
            dfc = p[T:2 * T]
            hfa = p[2 * T]
            rho = p[2 * T + 1]
            return att, dfc, hfa, rho

        def nll(p):
            att, dfc, hfa, rho = unpack(p)
            log_lam = att[hi] - dfc[ai] + hfa * nonneutral
            log_mu = att[ai] - dfc[hi]
            lam = np.exp(np.clip(log_lam, -3, 3))
            mu = np.exp(np.clip(log_mu, -3, 3))
            ll = _log_pois(hs, lam) + _log_pois(as_, mu)
            tau = _tau(hs, as_, lam, mu, rho)
            ll = ll + np.log(np.clip(tau, 1e-9, None))
            # identifiability anchors + light ridge
            pen = 1e3 * (att.mean() ** 2 + dfc.mean() ** 2) + 1e-3 * (att @ att + dfc @ dfc)
            return -(w * ll).sum() + pen

        p0 = np.zeros(2 * T + 2)
        p0[2 * T] = 0.25      # initial home advantage
        p0[2 * T + 1] = -0.05  # initial rho
        bounds = [(-3, 3)] * (2 * T) + [(-1.0, 1.0), (-0.2, 0.2)]
        res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200})

        att, dfc, hfa, rho = unpack(res.x)
        self.attack = {t: float(att[i]) for t, i in self.idx.items()}
        self.defence = {t: float(dfc[i]) for t, i in self.idx.items()}
        self.home_adv = float(hfa)
        self.rho = float(rho)
        return self

    # ── prediction ──────────────────────────────────────────────────────────

    def _lambdas(self, home, away, neutral=1):
        a_h = self.attack.get(home, 0.0)
        d_h = self.defence.get(home, 0.0)
        a_a = self.attack.get(away, 0.0)
        d_a = self.defence.get(away, 0.0)
        hfa = 0.0 if neutral else self.home_adv
        lam = np.exp(np.clip(a_h - d_a + hfa, -3, 3))
        mu = np.exp(np.clip(a_a - d_h, -3, 3))
        return lam, mu

    def score_matrix(self, home, away, neutral=1):
        lam, mu = self._lambdas(home, away, neutral)
        n = self.max_goals + 1
        k = np.arange(n)
        ph = np.exp(_log_pois(k, lam))
        pa = np.exp(_log_pois(k, mu))
        m = np.outer(ph, pa)
        # apply DC correction to the 2x2 low-score block
        m[0, 0] *= 1 - lam * mu * self.rho
        m[0, 1] *= 1 + lam * self.rho
        m[1, 0] *= 1 + mu * self.rho
        m[1, 1] *= 1 - self.rho
        m = np.clip(m, 0, None)
        return m / m.sum()

    def predict(self, home, away, neutral=1):
        """Return expected scoreline + independent W/D/L probabilities."""
        lam, mu = self._lambdas(home, away, neutral)
        m = self.score_matrix(home, away, neutral)
        home_win = np.tril(m, -1).sum()
        away_win = np.triu(m, 1).sum()
        draw = np.trace(m)
        ml = np.unravel_index(np.argmax(m), m.shape)
        return {
            "exp_home_goals": float(lam),
            "exp_away_goals": float(mu),
            "scoreline": (int(ml[0]), int(ml[1])),
            "home_win": float(home_win),
            "draw": float(draw),
            "away_win": float(away_win),
        }


def train_and_save(path="dc_model.joblib"):
    import joblib
    from model import RESULTS_URL
    print("loading results for Dixon-Coles...")
    df = pd.read_csv(RESULTS_URL)
    dc = DixonColes().fit(df)
    joblib.dump(dc, path)
    print(f"Dixon-Coles fitted on {len(dc.teams)} teams; "
          f"home_adv={dc.home_adv:.3f}, rho={dc.rho:.3f}; saved {path}")
    for h, a in [("Spain", "Cape Verde"), ("Brazil", "Argentina"),
                 ("United States", "Mexico")]:
        r = dc.predict(h, a, neutral=1)
        print(f"  {h} vs {a}: xG {r['exp_home_goals']:.2f}-{r['exp_away_goals']:.2f}, "
              f"likely {r['scoreline'][0]}-{r['scoreline'][1]}, "
              f"W/D/L {r['home_win']:.0%}/{r['draw']:.0%}/{r['away_win']:.0%}")
    return dc


if __name__ == "__main__":
    # Import through the module path so the pickled class is dixon_coles.DixonColes
    # (not __main__.DixonColes), keeping the saved model loadable elsewhere.
    from dixon_coles import train_and_save as _train_and_save
    _train_and_save()
