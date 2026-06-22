import raw from "@/data.json";

// ── types ──────────────────────────────────────────────────────────────
type Game = {
  home: string; away: string; city: string; date: string;
  p_home: number; p_draw: number; p_away: number;
  confidence: number; confident: boolean; pick: string; pick_label: string;
  features: Record<string, number | null>;
  dc: { scoreline: [number, number]; xg_home: number; xg_away: number;
        home: number; draw: number; away: number } | null;
  preview: string | null;
};
type YGame = { home: string; away: string; home_score: number; away_score: number;
  pick_label: string; confidence: number; confident: boolean; correct: boolean };
type SiteData = {
  ledger_start: string; confident_threshold: number;
  accuracy: { pct: number | null; correct: number; total: number };
  today: { date: string | null; games: Game[] };
  yesterday: { date: string | null; games: YGame[] };
  title_odds: { rank: number; team: string; pct: number }[];
  meta: { matches: number; teams: number; sims: number; previews: boolean };
};
const data = raw as unknown as SiteData;
const pct = (x: number) => `${Math.round(x * 100)}%`;
const fmtDate = (iso: string) => {
  const [y, m, d] = iso.split("-");
  return `${m}/${d}/${y}`;
};

const GLOSSARY: [string, string][] = [
  ["Elo rating", "A running strength score updated after every match — higher means stronger."],
  ["Recent form", "Average goals scored and conceded over the team's last 5 games."],
  ["Squad strength", "How strong the player pool is, from FIFA player ratings (0–100)."],
  ["Host nation", "Whether the team is playing in its home country (USA, Canada, or Mexico)."],
  ["Crowd-support proxy", "An estimate of fan presence, based on travel distance to the venue."],
];

function Reveal({ children }: { children: React.ReactNode }) {
  return <div className="reveal">{children}</div>;
}

function Bars({ home, away, g }: { home: string; away: string; g: Game }) {
  const rows: [string, number, string][] = [
    [`${home} win`, g.p_home, "win"],
    ["Draw", g.p_draw, "draw"],
    [`${away} win`, g.p_away, "loss"],
  ];
  return (
    <div>
      {rows.map(([lab, p, cls]) => (
        <div className="bar-row" key={lab}>
          <div className="bar-lab">{lab}</div>
          <div className="bar-track"><div className={`bar-fill ${cls}`} style={{ width: pct(p) }} /></div>
          <div className="bar-val">{pct(p)}</div>
        </div>
      ))}
    </div>
  );
}

function HowWeGotThis({ f }: { f: Record<string, number | null> }) {
  const n = (k: string, d = 0) => (f[k] ?? 0).toFixed(d);
  const rows: [string, string, string][] = [
    ["Elo rating", n("home_elo"), n("away_elo")],
    ["Recent form (GF/GA)", `${n("home_form_gf", 1)} / ${n("home_form_ga", 1)}`,
      `${n("away_form_gf", 1)} / ${n("away_form_ga", 1)}`],
    ["Squad strength (0–100)", n("home_strength"), n("away_strength")],
    ["Host nation", f.home_is_host ? "yes" : "no", f.away_is_host ? "yes" : "no"],
    ["Crowd-support proxy", n("home_support_index", 2), n("away_support_index", 2)],
  ];
  return (
    <details className="how">
      <summary>How we got this ↓</summary>
      <table className="howtab"><tbody>
        {rows.map((r) => (
          <tr key={r[0]}><td>{r[0]}</td><td className="v">{r[1]}</td><td className="v">{r[2]}</td></tr>
        ))}
      </tbody></table>
    </details>
  );
}

export default function Home() {
  const acc = data.accuracy;
  const lead = data.title_odds[0]?.pct ?? 1;

  return (
    <main className="flex-1">
      {/* ── HERO ── */}
      <header className="hero">
        <div className="hero-media" />
        <div className="hero-grid" />
        <div className="hero-inner wrap">
          <p className="eyebrow rise d1">Machine-learning football intelligence</p>
          <h1 className="display hero-title rise d2">World Cup 2026<br /><span className="l2">Predictor</span></h1>
          <p className="hero-tag rise d3">Every match predicted — and graded against real results.</p>
          <p className="hero-by rise d4">Built by <b>Albert Joseph</b></p>
          <div className="strip rise d5">
            <div><div className="num strip-n">49K</div><div className="strip-l">matches analyzed</div></div>
            <div><div className="num strip-n">{data.meta.teams}</div><div className="strip-l">teams simulated</div></div>
            <div><div className="num strip-n">5K</div><div className="strip-l">simulations / refresh</div></div>
          </div>
        </div>
        <div className="scrollcue">scroll</div>
      </header>

      {/* ── 01 TRACK RECORD ── */}
      <section className="band">
        <div className="wrap">
          <Reveal>
            <div className="sec-head"><span className="sec-no">01</span><h2 className="display sec-title">Track record</h2></div>
          </Reveal>
          <Reveal>
            <div className="tr">
              <div className="num tr-num">{acc.pct !== null ? `${Math.round(acc.pct)}%` : "—"}</div>
              <p className="tr-sub">
                <b>{acc.correct} of {acc.total}</b> predictions correct since {fmtDate(data.ledger_start)},
                when this went live.{" "}
                <span className="muted">A pick is right when the most-likely outcome matches the result.</span>
              </p>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ── 02 TODAY'S GAMES ── */}
      <section className="band">
        <div className="wrap">
          <Reveal>
            <div className="sec-head">
              <span className="sec-no">02</span><h2 className="display sec-title">Today&apos;s games</h2>
              {data.today.date && <span className="sec-sub">{fmtDate(data.today.date)}</span>}
            </div>
          </Reveal>
          <Reveal>
            <div className="legend">
              {GLOSSARY.map(([t, d]) => (
                <div key={t}><div className="legend-t">{t}</div><div className="legend-d">{d}</div></div>
              ))}
            </div>
          </Reveal>
          <div className="cards">
            {data.today.games.map((g) => (
              <Reveal key={`${g.home}-${g.away}`}>
                <article className="gcard">
                  <div className="gcard-top">
                    <span className="gcard-teams">{g.home} vs {g.away}</span>
                    {g.confident ? <span className="tag conf">Confident pick</span> : <span className="tag toss">Toss-up</span>}
                  </div>
                  <div className="gcard-meta">{fmtDate(g.date)} · {g.city}</div>
                  <div className="gcard-grid">
                    <div>
                      <p className="lab">Win / draw / loss</p>
                      <Bars home={g.home} away={g.away} g={g} />
                    </div>
                    <div>
                      <p className="lab">Dixon-Coles scoreline</p>
                      {g.dc ? (
                        <>
                          <div className="num dc-score">{g.dc.scoreline[0]}–{g.dc.scoreline[1]}</div>
                          <div className="dc-meta">
                            xG {g.dc.xg_home.toFixed(2)} – {g.dc.xg_away.toFixed(2)} · independent W/D/L{" "}
                            {pct(g.dc.home)}/{pct(g.dc.draw)}/{pct(g.dc.away)}
                          </div>
                        </>
                      ) : <div className="dc-meta">—</div>}
                    </div>
                  </div>
                  {g.preview
                    ? <p className="preview">{g.preview}</p>
                    : <p className="preview preview-none">Match preview appears here once the build runs with an Anthropic API key.</p>}
                  <HowWeGotThis f={g.features} />
                </article>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ── 03 ROAD TO THE CUP ── */}
      <section className="band">
        <div className="wrap">
          <Reveal>
            <div className="sec-head">
              <span className="sec-no">03</span><h2 className="display sec-title">Road to the Cup</h2>
              <span className="sec-sub">title odds, simulated</span>
            </div>
          </Reveal>
          <div className="trk">
            {data.title_odds.map((o, i) => (
              <Reveal key={o.team}>
                <div className="trk-row">
                  <div className={`num trk-rank${i === 0 ? " lead" : ""}`}>{o.rank}</div>
                  <div className="trk-body">
                    <div className="trk-team">{o.team}</div>
                    <div className="trk-bar"><div className={`trk-bar-fill${i === 0 ? " lead" : ""}`} style={{ width: `${(o.pct / lead) * 100}%` }} /></div>
                    <div className="num trk-val">{o.pct.toFixed(1)}%</div>
                  </div>
                </div>
              </Reveal>
            ))}
          </div>
          <p className="sec-sub" style={{ marginTop: "1.6rem" }}>
            {data.meta.sims.toLocaleString()} simulations · official 2026 bracket · updates on refresh.
          </p>
        </div>
      </section>

      {/* ── 04 YESTERDAY ── */}
      <section className="band">
        <div className="wrap">
          <Reveal>
            <div className="sec-head">
              <span className="sec-no">04</span><h2 className="display sec-title">Yesterday&apos;s results</h2>
              {data.yesterday.date && <span className="sec-sub">{fmtDate(data.yesterday.date)} · prediction vs actual</span>}
            </div>
          </Reveal>
          <div className="ygrid">
            {data.yesterday.games.map((y) => (
              <Reveal key={`${y.home}-${y.away}`}>
                <div className={`ycard${y.correct ? " iswin" : ""}`}>
                  <div className="yc-score">{y.home} {y.home_score}–{y.away_score} {y.away}</div>
                  <div className="yc-pred">Predicted <b>{y.pick_label}</b> ({Math.round(y.confidence * 100)}%)</div>
                  <div className={`verdict ${y.correct ? "ok" : "no"}`}>{y.correct ? "✓ Correct" : "✗ Missed"}</div>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <footer>
        <div className="wrap">
          Data: martj42 international results · FIFA 22 ratings. Predictions are leakage-safe and
          frozen pre-kickoff. Built by Albert Joseph.
        </div>
      </footer>
    </main>
  );
}
