"""World Cup 2026 Predictor — Streamlit UI.

Section order: track record (accuracy) → yesterday's results → today's games →
road to the cup. Off-white + pitch-green World Cup theme.

Predictions are the leakage-safe, pre-kickoff numbers frozen in the ledger
(see daily.py); there is no on-demand prediction path that could run up API costs.
"""

import os
import joblib
import pandas as pd
import streamlit as st

from daily import (build_tables, build_preview_data, PICK_LABEL, LEDGER_START,
                   CONFIDENT_THRESHOLD)

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")

# On Streamlit Cloud the API key is added as a secret; mirror it into the
# environment so preview.py (which reads os.environ) finds it regardless of how
# the host exposes secrets. Harmless locally (export still works).
try:
    if not os.environ.get("ANTHROPIC_API_KEY") and "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;700;800&family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500&display=swap');
:root{
  --bg:#F3EFE4; --surface:#FBF8EF; --ink:#16241B; --muted:#5E6B61;
  --green:#16A35A; --green-deep:#0B5C36; --gold:#C9982E; --coral:#CF5638;
  --bar-win:#0C7C42; --bar-draw:#9C6F15;
  --line:#E4DCCB; --track:#ECE6D6;
}
.stApp{ background:var(--bg); }
.block-container{ max-width:1040px; padding-top:2.6rem; padding-bottom:4.5rem; }
html, body, [data-testid="stAppViewContainer"], .stMarkdown, p, span, div, label, li{
  font-family:'Inter', system-ui, sans-serif; color:var(--ink);
}
h1,h2,h3,h4{ font-family:'Space Grotesk', sans-serif !important; color:var(--ink) !important;
  letter-spacing:-0.02em; }
a{ color:var(--green-deep); }
hr{ border:none; border-top:1px solid var(--line); margin:2.6rem 0; }

.hero-title{ font-family:'Baloo 2', system-ui, sans-serif; font-weight:800; font-size:3.9rem;
  line-height:1.02; letter-spacing:-0.01em; color:var(--green); margin:0 0 .5rem; }
.hero-intro{ font-size:1.28rem; line-height:1.45; color:#2c352e; max-width:620px; margin:0 0 .55rem; }
.hero-by{ font-size:.9rem; color:var(--muted); }
.hero-by b{ color:var(--green-deep); font-weight:500; }
.hstrip{ display:flex; gap:2.75rem; margin:1.7rem 0 .2rem; flex-wrap:wrap; }
.hstat-n{ font-family:'Baloo 2', system-ui, sans-serif; font-weight:800; font-size:1.85rem;
  line-height:1; color:var(--green-deep); font-variant-numeric:tabular-nums; }
.hstat-l{ font-size:.78rem; color:var(--muted); margin-top:.3rem; }

.sec{ display:flex; gap:14px; align-items:baseline; margin:.2rem 0 1.1rem; }
.sec-k{ font-family:'Space Grotesk'; font-weight:600; font-size:.82rem; color:var(--green-deep);
  letter-spacing:.16em; }
.sec-t{ font-family:'Space Grotesk'; font-weight:600; font-size:1.55rem; }
.sec-s{ font-size:.9rem; color:var(--muted); }

.stat-num{ font-family:'Baloo 2', system-ui, sans-serif; font-weight:800; font-size:4.6rem;
  line-height:.95; color:var(--green); font-variant-numeric:tabular-nums; }
.stat-sub{ color:#2c352e; font-size:1.0rem; line-height:1.55; max-width:520px; }
.stat-sub b{ color:var(--green-deep); }

[data-testid="stVerticalBlockBorderWrapper"]{ background:var(--surface);
  border:1px solid var(--line) !important; border-radius:14px; }

.gm-title{ font-family:'Space Grotesk'; font-weight:600; font-size:1.18rem; margin:0; }
.gm-meta{ color:var(--muted); font-size:.85rem; margin:.1rem 0 .2rem; }
.lab{ font-family:'Space Grotesk'; font-weight:600; font-size:.72rem; letter-spacing:.14em;
  color:var(--muted); text-transform:uppercase; margin:0 0 .5rem; }

.wdl{ display:flex; flex-direction:column; gap:9px; }
.wdl-row{ display:grid; grid-template-columns:118px 1fr 42px; align-items:center; gap:10px; }
.wdl-label{ font-size:.86rem; color:#3c463e; }
.wdl-track{ height:11px; background:var(--track); border-radius:6px; overflow:hidden; }
.wdl-fill{ height:100%; border-radius:6px; }
.win{ background:var(--bar-win); } .draw{ background:var(--bar-draw); } .loss{ background:var(--coral); }
.wdl-val{ font-family:'Space Grotesk'; font-weight:600; font-size:.9rem; text-align:right;
  font-variant-numeric:tabular-nums; }

.dc-score{ font-family:'Baloo 2', system-ui, sans-serif; font-weight:700; font-size:2.9rem;
  line-height:1; color:var(--ink); font-variant-numeric:tabular-nums; }
.dc-meta{ color:var(--muted); font-size:.85rem; margin-top:.3rem; font-variant-numeric:tabular-nums; }

.preview{ font-size:.98rem; line-height:1.6; color:#2c352e; }

.badge{ display:inline-block; font-weight:500; font-size:.82rem; padding:3px 11px; border-radius:999px; }
.badge.ok{ background:#E2F0E3; color:#1B6B36; } .badge.no{ background:#F6E2DB; color:#9E3B22; }
.tag{ display:inline-block; font-weight:500; font-size:.76rem; padding:2px 10px; border-radius:999px;
  border:1px solid; background:transparent; }
.tag.conf{ color:var(--green-deep); border-color:var(--green-deep); }
.tag.toss{ color:#6B6453; border-color:#CBBFA6; }

.yc-score{ font-family:'Space Grotesk'; font-weight:600; font-size:1.05rem; margin:0; }
.yc-pred{ color:var(--muted); font-size:.88rem; margin:.15rem 0 .5rem; }
.yc-pred b{ color:var(--ink); font-weight:500; }

.trk-row{ display:grid; grid-template-columns:26px 158px 1fr 58px; align-items:center; gap:12px;
  padding:11px 0; border-bottom:1px solid var(--line); }
.trk-rank{ font-family:'Space Grotesk'; font-weight:600; color:var(--green-deep); font-size:1.05rem;
  font-variant-numeric:tabular-nums; }
.trk-team{ font-weight:500; }
.trk-track{ height:9px; background:var(--track); border-radius:6px; overflow:hidden; }
.trk-fill{ height:100%; background:var(--bar-win); border-radius:6px; }
.trk-fill.lead{ background:var(--bar-draw); }
.trk-val{ font-family:'Space Grotesk'; font-weight:600; text-align:right;
  font-variant-numeric:tabular-nums; }

.how{ width:100%; border-collapse:collapse; font-size:.9rem; }
.how td{ padding:7px 0; border-bottom:1px solid var(--line); }
.how td:first-child{ color:var(--muted); }
.how td.v{ font-family:'Space Grotesk'; font-weight:500; text-align:right;
  font-variant-numeric:tabular-nums; }

.stButton button{ border-radius:999px; border:1px solid var(--green-deep) !important;
  color:var(--green-deep) !important; background:transparent !important; font-weight:500; }
.stButton button:hover{ background:#E7F0E7 !important; }
.foot{ color:var(--muted); font-size:.82rem; line-height:1.6; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# ── cached resources ─────────────────────────────────────────────────────────
@st.cache_resource
def get_dc():
    try:
        return joblib.load("dc_model.joblib")
    except Exception:
        return None


@st.cache_resource
def get_simulator():
    from sim import TournamentSimulator
    return TournamentSimulator()


@st.cache_data(show_spinner="Crunching the day's numbers…")
def get_tables(nonce):
    return build_tables()


@st.cache_data(show_spinner="Simulating the tournament…")
def run_tracker(nonce, n=5000):
    return get_simulator().run(n=n, seed=0)


def auto_preview(row, dc):
    from preview import generate_preview
    return generate_preview(row["home_team"], row["away_team"], build_preview_data(row, dc))


@st.cache_resource(show_spinner="Preparing today's match previews…")
def prewarm_today(nonce):
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return 0
    ready = 0
    for _, row in get_tables(nonce)["today_games"].iterrows():
        try:
            auto_preview(row, get_dc())
            ready += 1
        except Exception:
            pass
    return ready


# ── small render helpers ─────────────────────────────────────────────────────
def section(kicker, title, subtitle=None):
    sub = f'<span class="sec-s">{subtitle}</span>' if subtitle else ""
    st.markdown(f'<div class="sec"><span class="sec-k">{kicker}</span>'
                f'<span class="sec-t">{title}</span>{sub}</div>', unsafe_allow_html=True)


def conf_tag(row):
    """Confident-pick / toss-up tag from the top outcome's probability."""
    conf = max(row["p_home"], row["p_draw"], row["p_away"])
    if conf >= CONFIDENT_THRESHOLD:
        return '<span class="tag conf">Confident pick</span>'
    return '<span class="tag toss">Toss-up</span>'


def wdl_bars(home, away, ph, pdr, pa):
    rows = [(f"{home} win", ph, "win"), ("Draw", pdr, "draw"), (f"{away} win", pa, "loss")]
    html = '<div class="wdl">'
    for label, p, cls in rows:
        html += (f'<div class="wdl-row"><div class="wdl-label">{label}</div>'
                 f'<div class="wdl-track"><div class="wdl-fill {cls}" style="width:{p*100:.0f}%"></div></div>'
                 f'<div class="wdl-val">{p*100:.0f}%</div></div>')
    st.markdown(html + "</div>", unsafe_allow_html=True)


def how_table(home, away, row):
    rows = [
        ("Elo rating", f"{row['home_elo']:.0f}", f"{row['away_elo']:.0f}"),
        ("Recent form (GF/GA)", f"{row['home_form_gf']:.1f} / {row['home_form_ga']:.1f}",
         f"{row['away_form_gf']:.1f} / {row['away_form_ga']:.1f}"),
        ("Squad strength (0–100)", f"{row['home_strength']:.0f}", f"{row['away_strength']:.0f}"),
        ("Host nation", "yes" if row["home_is_host"] else "no",
         "yes" if row["away_is_host"] else "no"),
        ("Crowd-support proxy", f"{row['home_support_index']:.2f}", f"{row['away_support_index']:.2f}"),
    ]
    body = (f'<tr><td></td><td class="v">{home}</td><td class="v">{away}</td></tr>'
            + "".join(f'<tr><td>{m}</td><td class="v">{h}</td><td class="v">{a}</td></tr>'
                      for m, h, a in rows))
    st.markdown(f'<table class="how">{body}</table>', unsafe_allow_html=True)
    st.caption("Squad strength is a single 2021 FIFA-ratings snapshot (not time-varying). "
               "Crowd support is a travel-proximity proxy, not real ticketing data.")


# ── data ─────────────────────────────────────────────────────────────────────
if "nonce" not in st.session_state:
    st.session_state.nonce = 0

dc = get_dc()
tables = get_tables(st.session_state.nonce)

# ── hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-title">World Cup 2026 Predictor</div>'
    '<p class="hero-intro">Every match predicted — and graded against real results.</p>'
    '<p class="hero-by">Built by <b>Albert Joseph</b></p>'
    '<div class="hstrip">'
    '<div><div class="hstat-n">49K</div><div class="hstat-l">matches analyzed</div></div>'
    '<div><div class="hstat-n">48</div><div class="hstat-l">teams simulated</div></div>'
    '<div><div class="hstat-n">5K</div><div class="hstat-l">simulations / refresh</div></div>'
    '</div>',
    unsafe_allow_html=True,
)

c1, _ = st.columns([1, 3])
with c1:
    if st.button("↻ Refresh with latest results"):
        get_tables.clear()
        get_simulator.clear()
        run_tracker.clear()
        st.session_state.nonce += 1
        st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# ── 01 · Track record ────────────────────────────────────────────────────────
acc = tables["accuracy"]
section("01", "Track record")
left, right = st.columns([1, 2])
with left:
    st.markdown(f'<div class="stat-num">{acc["pct"]:.0f}%</div>'
                if acc["pct"] is not None else '<div class="stat-num">—</div>',
                unsafe_allow_html=True)
with right:
    st.markdown(
        f'<div class="stat-sub"><b>{acc["correct"]} of {acc["total"]}</b> predictions correct '
        f'since {LEDGER_START.date()}, when this went live — most-likely outcome vs the '
        f'actual result.</div>',
        unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── 02 · Yesterday's results ─────────────────────────────────────────────────
yesterday = tables["yesterday"]
section("02", "Yesterday's results",
        f"{yesterday.date()} · prediction vs actual" if yesterday is not None else None)
if yesterday is not None and len(tables["yesterday_games"]):
    cards = st.columns(len(tables["yesterday_games"]))
    for card, (_, row) in zip(cards, tables["yesterday_games"].iterrows()):
        home, away = row["home_team"], row["away_team"]
        conf = max(row["p_home"], row["p_draw"], row["p_away"]) * 100
        ok = bool(row["correct"])
        with card:
            with st.container(border=True):
                st.markdown(f'<p class="yc-score">{home} {int(row["home_score"])}–'
                            f'{int(row["away_score"])} {away}</p>'
                            f'<p class="yc-pred">Predicted <b>{PICK_LABEL[row["pick"]]}</b> '
                            f'({conf:.0f}%)</p>'
                            f'<span class="badge {"ok" if ok else "no"}">'
                            f'{"✓ Correct" if ok else "✗ Missed"}</span> &nbsp; '
                            f'{conf_tag(row)}',
                            unsafe_allow_html=True)
else:
    st.caption("No completed fixtures yet.")

st.markdown("<hr>", unsafe_allow_html=True)

# ── 03 · Today's games ───────────────────────────────────────────────────────
today = tables["today"]
prewarm_today(st.session_state.nonce)
section("03", "Today's games",
        f"{today.date()} · odds, scoreline & preview per game" if today is not None else None)
st.caption(f"Confident pick = model ≥{CONFIDENT_THRESHOLD*100:.0f}% sure; closer games are toss-ups.")

for _, row in tables["today_games"].iterrows():
    home, away = row["home_team"], row["away_team"]
    with st.container(border=True):
        st.markdown(f'<p class="gm-title">{home} vs {away} &nbsp; {conf_tag(row)}</p>'
                    f'<p class="gm-meta">{pd.to_datetime(row["date"]).date()} · '
                    f'{row.get("city", "")}</p>', unsafe_allow_html=True)

        lcol, rcol = st.columns([3, 2])
        with lcol:
            st.markdown('<p class="lab">Win / draw / loss</p>', unsafe_allow_html=True)
            wdl_bars(home, away, row["p_home"], row["p_draw"], row["p_away"])
        with rcol:
            st.markdown('<p class="lab">Dixon-Coles scoreline</p>', unsafe_allow_html=True)
            if dc is not None:
                d = dc.predict(home, away, neutral=1)
                st.markdown(
                    f'<div class="dc-score">{d["scoreline"][0]}–{d["scoreline"][1]}</div>'
                    f'<div class="dc-meta">xG {d["exp_home_goals"]:.2f} – '
                    f'{d["exp_away_goals"]:.2f} · independent W/D/L '
                    f'{d["home_win"]*100:.0f}/{d["draw"]*100:.0f}/{d["away_win"]*100:.0f}</div>',
                    unsafe_allow_html=True)
            else:
                st.caption("Dixon-Coles model not loaded.")

        st.markdown('<p class="lab" style="margin-top:1rem">Match preview</p>', unsafe_allow_html=True)
        try:
            st.markdown(f'<p class="preview">{auto_preview(row, dc)}</p>', unsafe_allow_html=True)
        except Exception as e:
            st.info(str(e) or "Preview unavailable.")

        with st.expander("How we got this"):
            how_table(home, away, row)

st.markdown("<hr>", unsafe_allow_html=True)

# ── 04 · Road to the Cup ─────────────────────────────────────────────────────
section("04", "Road to the Cup", "title odds, simulated")
table, n_sims = run_tracker(st.session_state.nonce)
top = table[:5]
top_p = top[0][1] if top else 1
html = ""
for i, (team, p) in enumerate(top):
    lead = "lead" if i == 0 else ""
    html += (f'<div class="trk-row"><div class="trk-rank">{i+1}</div>'
             f'<div class="trk-team">{team}</div>'
             f'<div class="trk-track"><div class="trk-fill {lead}" '
             f'style="width:{p/top_p*100:.0f}%"></div></div>'
             f'<div class="trk-val">{p*100:.1f}%</div></div>')
st.markdown(html, unsafe_allow_html=True)
st.caption(f"{n_sims:,} simulations · official 2026 bracket · updates on refresh.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<p class="foot">Data: martj42 international results · FIFA 22 ratings. '
    'Predictions are leakage-safe and frozen pre-kickoff. Built by Albert Joseph.</p>',
    unsafe_allow_html=True)
