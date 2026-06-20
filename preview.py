"""LLM match preview via the Anthropic API.

generate_preview(home, away, data) turns the model's numbers into a 2-3 sentence
punchy preview that names the single biggest factor. The API key is read from the
ANTHROPIC_API_KEY environment variable and is never hardcoded. Results are cached
per matchup so the same fixture is not billed twice.
"""

import os
import json

# Latest capable Sonnet; override with ANTHROPIC_MODEL if desired.
DEFAULT_MODEL = "claude-sonnet-4-6"

# Disk-persisted cache so a matchup is generated (and billed) at most once, even
# across server restarts / redeploys. Keyed by match date + teams.
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "preview_cache.json")
_CACHE = {}


def _disk_key(home, away, data):
    return f"{data.get('match_date', '')}|{home}|{away}"


def _load_disk():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_disk(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, CACHE_FILE)


def is_cached(home, away, data):
    """True if this matchup's preview is already cached (no API call needed)."""
    key = _disk_key(home, away, data)
    return key in _CACHE or key in _load_disk()


def _build_prompt(home, away, data):
    f = data.get("features", {})
    dc = data.get("dc", {})
    lines = [
        f"Match: {home} vs {away} (neutral venue: {bool(data.get('neutral', 1))})",
        "",
        "Model win/draw/loss probabilities (XGBoost):",
        f"  {home} win: {data['home_win']*100:.0f}%",
        f"  draw: {data['draw']*100:.0f}%",
        f"  {away} win: {data['away_win']*100:.0f}%",
        "",
        "Underlying numbers:",
        f"  Elo: {home} {f.get('home_elo', 0):.0f} vs {away} {f.get('away_elo', 0):.0f} "
        f"(gap {f.get('elo_diff', 0):+.0f})",
        f"  Recent form goals for/against: {home} {f.get('home_form_gf', 0):.1f}/{f.get('home_form_ga', 0):.1f}, "
        f"{away} {f.get('away_form_gf', 0):.1f}/{f.get('away_form_ga', 0):.1f}",
        f"  Squad strength (0-100): {home} {f.get('home_strength', 0):.0f} vs {away} {f.get('away_strength', 0):.0f}",
        f"  Host nation: {home} {'yes' if f.get('home_is_host') else 'no'}, "
        f"{away} {'yes' if f.get('away_is_host') else 'no'}",
        f"  Crowd-support proxy (travel proximity, 0-1): {home} {f.get('home_support_index', 0):.2f} vs "
        f"{away} {f.get('away_support_index', 0):.2f}",
    ]
    if dc:
        lines += [
            "",
            "Dixon-Coles goals model:",
            f"  expected goals {home} {dc.get('exp_home_goals', 0):.2f} - "
            f"{dc.get('exp_away_goals', 0):.2f} {away}",
            f"  most likely scoreline {dc.get('scoreline', ('?', '?'))[0]}-{dc.get('scoreline', ('?', '?'))[1]}",
        ]
    lines += [
        "",
        "Write a 2-3 sentence match preview in plain, punchy language for a general "
        "football audience. Explain who the model favours and why, and explicitly name "
        "the single biggest factor driving the pick. No betting advice, no hedging "
        "filler, no bullet points.",
    ]
    return "\n".join(lines)


def generate_preview(home, away, data, model=None):
    """Return a short LLM-written preview string. Cached per matchup.

    Raises RuntimeError with a clear message if the API key is missing or the
    SDK is unavailable, so the UI can show a friendly note instead of crashing.
    """
    key = _disk_key(home, away, data)
    if key in _CACHE:
        return _CACHE[key]
    disk = _load_disk()
    if key in disk:
        _CACHE[key] = disk[key]
        return disk[key]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export your key to enable previews:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("The `anthropic` package is not installed (pip install anthropic).") from e

    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model=model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
            max_tokens=200,
            messages=[{"role": "user", "content": _build_prompt(home, away, data)}],
        )
    except anthropic.AuthenticationError as e:
        raise RuntimeError(
            "Anthropic rejected the API key (401). Make sure ANTHROPIC_API_KEY is your "
            "real key from console.anthropic.com — it starts with 'sk-ant-' and must not "
            "be the literal placeholder 'sk-ant-...'."
        ) from e
    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error while generating the preview: {e}") from e

    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    _CACHE[key] = text
    disk[key] = text
    _save_disk(disk)
    return text


if __name__ == "__main__":
    # Smoke test against the saved models (needs ANTHROPIC_API_KEY).
    from model import load_artifacts, predict
    import joblib

    ctx, _ = load_artifacts()
    out = predict("Spain", "Cape Verde", ctx)
    try:
        dc = joblib.load("dc_model.joblib")
        out["dc"] = dc.predict("Spain", "Cape Verde", neutral=1)
        out["dc_scoreline"] = out["dc"]["scoreline"]
    except Exception:
        pass
    out["neutral"] = 1
    try:
        print(generate_preview("Spain", "Cape Verde", out))
    except RuntimeError as e:
        print("[preview unavailable]", e)
