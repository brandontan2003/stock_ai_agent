"""
classifier.py — Dynamic regime classifier + message builder.

Regime thresholds are computed from rolling VIX percentiles, not hardcoded numbers.
VIX=25 in a calm year is fear. VIX=25 after a crash is calm.
The message builder pulls from learned outcome stats when available,
falling back to historical priors when the tracker has insufficient data.
"""

import yfinance as yf

# Fallback priors — used until the outcome tracker has real data
HISTORICAL_PRIORS = {
    "extreme_panic": {
        "label": "🚨 EXTREME PANIC",
        "prior_note": "VIX>90th pct — historically preceded +18% median gain over 60d (2008, 2020, 2022)",
        "signal": "Strong contrarian buy zone — if you have dry powder",
    },
    "panic": {
        "label": "🔴 PANIC",
        "prior_note": "VIX 70–90th pct — associated with +11% median gain over 30d",
        "signal": "Elevated risk, potential opportunity for long-term buyers",
    },
    "fear_confirmed": {
        "label": "🟠 FEAR (confirmed)",
        "prior_note": "Price decline + elevated VIX — watch for VIX peak before entering",
        "signal": "Not yet a buy signal. Monitor for stabilisation.",
    },
    "fear": {
        "label": "🟡 FEAR",
        "prior_note": "VIX 40–70th pct — moderate caution",
        "signal": "Hold positions, avoid panic selling",
    },
    "calm": {
        "label": "🟢 CALM",
        "prior_note": "No directional signal",
        "signal": "Normal market conditions",
    },
}

MIN_SAMPLES = 3  # minimum outcomes before we trust learned stats over priors


def get_dynamic_thresholds(lookback_days: int = 252) -> dict:
    """
    Compute VIX regime thresholds from rolling percentiles.
    Uses the past lookback_days of VIX data (default: 1 trading year).
    Returns percentile-based cutoffs rather than hardcoded 20/30/40.
    """
    vix_hist = yf.Ticker("^VIX").history(period=f"{lookback_days}d")["Close"].dropna()

    thresholds = {
        "calm_max":        round(float(vix_hist.quantile(0.40)), 2),  # below 40th pct = calm
        "fear_max":        round(float(vix_hist.quantile(0.70)), 2),  # 40-70th pct = fear
        "panic_max":       round(float(vix_hist.quantile(0.90)), 2),  # 70-90th pct = panic
        # above 90th pct = extreme panic
        "current_vix_pct": round(float(vix_hist.rank(pct=True).iloc[-1]) * 100, 1),
    }
    return thresholds


def classify_regime(vix: float, rsi: float, spx_change_pct: float = None,
                    thresholds: dict = None) -> str:
    """
    Classify market regime using dynamic thresholds when available,
    falling back to sensible hardcoded defaults.
    """
    if thresholds is None:
        thresholds = {"calm_max": 20, "fear_max": 30, "panic_max": 40}

    if vix > thresholds["panic_max"]:
        return "extreme_panic"
    elif vix > thresholds["fear_max"]:
        return "panic"
    elif vix > thresholds["calm_max"] or rsi < 35:
        if spx_change_pct is not None and spx_change_pct < -3:
            return "fear_confirmed"
        return "fear"
    return "calm"


def build_message(signals: dict, regime: str, stock_symbol: str,
                  thresholds: dict = None, learned_stats: dict = None) -> str:
    """
    Build a Telegram message. Uses learned outcome stats when available
    (>= MIN_SAMPLES outcomes for this regime), otherwise falls back to priors.
    """
    prior = HISTORICAL_PRIORS[regime]
    label = prior["label"]
    action = prior["signal"]

    # Use learned stats if we have enough samples for this regime
    regime_stats = (learned_stats or {}).get(regime, {})
    n = regime_stats.get("sample_size", 0)

    if n >= MIN_SAMPLES:
        median_ret = regime_stats["median_return"]
        win_rate   = regime_stats["win_rate"]
        perf_line  = (
            f"📊 *This agent's track record for {label}:* "
            f"`{median_ret:+.1f}%` median 30d return, "
            f"`{win_rate:.0f}%` win rate across {n} signals"
        )
    else:
        perf_line = f"📚 *Prior (insufficient data yet):* _{prior['prior_note']}_"
        if n > 0:
            perf_line += f"\n_({n} signal{'s' if n > 1 else ''} logged — need {MIN_SAMPLES} to switch to learned stats)_"

    # VIX percentile context
    thresh_line = ""
    if thresholds:
        pct = thresholds.get("current_vix_pct", "?")
        thresh_line = f"_VIX is at the *{pct}th percentile* of the past year_\n"

    spx_chg = signals.get("5D_CHG")
    spx_line = f" | 5d chg: `{spx_chg:+.1f}%`" if spx_chg is not None else ""

    return (
        f"*Market Signal Alert*\n\n"
        f"Regime: {label}\n"
        f"VIX: `{signals['VIX']}` | {stock_symbol}: `{signals[stock_symbol]}` "
        f"| RSI: `{signals['RSI']}`{spx_line}\n"
        f"{thresh_line}\n"
        f"*Read:* {action}\n"
        f"{perf_line}\n\n"
        f"_Not financial advice._"
    )
