"""
tracker.py — Outcome tracker for past signals.

Every time the agent fires a signal, we log:
  - timestamp
  - regime called
  - price at signal time
  - VIX + RSI at signal time

30 days later, the agent checks the actual return and updates the accuracy log.
The classifier reads this log to replace hardcoded stats with real observed performance.
"""

import json
import os
from datetime import datetime, timezone

SIGNALS_FILE = "data/signals.json"
OUTCOMES_FILE = "data/outcomes.json"


def _load(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def _save(path: str, data: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def log_signal(regime: str, signals: dict, stock_ticker: str):
    """Called every time the agent fires a regime-change alert."""
    records = _load(SIGNALS_FILE)
    records.append({
        "id": len(records),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "price": signals[stock_ticker],
        "vix": signals["VIX"],
        "rsi": signals["RSI"],
        "ticker": stock_ticker,
        "resolved": False,
    })
    _save(SIGNALS_FILE, records)


def resolve_outcomes(current_prices: dict):
    """
    Called on every poll. Checks if any unresolved signals are 30+ days old.
    If so, computes the actual return and writes to outcomes.json.
    current_prices: {ticker: current_price}
    """
    records = _load(SIGNALS_FILE)
    outcomes = _load(OUTCOMES_FILE)
    updated = False

    for record in records:
        if record["resolved"]:
            continue

        signal_time = datetime.fromisoformat(record["timestamp"])
        age_days = (datetime.now(timezone.utc) - signal_time).days

        if age_days < 30:
            continue

        ticker = record["ticker"]
        if ticker not in current_prices:
            continue

        entry_price = record["price"]
        exit_price = current_prices[ticker]
        actual_return_pct = ((exit_price - entry_price) / entry_price) * 100

        outcome = {
            "signal_id": record["id"],
            "regime": record["regime"],
            "entry_price": entry_price,
            "exit_price": round(exit_price, 2),
            "actual_return_pct": round(actual_return_pct, 2),
            "days_held": age_days,
            "signal_timestamp": record["timestamp"],
            "resolved_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        outcomes.append(outcome)
        record["resolved"] = True
        updated = True
        print(f"Resolved signal #{record['id']}: {record['regime']} → {actual_return_pct:+.1f}% over {age_days}d")

    if updated:
        _save(SIGNALS_FILE, records)
        _save(OUTCOMES_FILE, outcomes)

    return outcomes


def compute_learned_stats() -> dict:
    """
    Reads outcomes.json and computes per-regime observed stats.
    Returns a dict keyed by regime with median return and sample size.
    Falls back gracefully if no data yet.
    """
    outcomes = _load(OUTCOMES_FILE)
    if not outcomes:
        return {}

    from collections import defaultdict
    import statistics

    by_regime = defaultdict(list)
    for o in outcomes:
        by_regime[o["regime"]].append(o["actual_return_pct"])

    stats = {}
    for regime, returns in by_regime.items():
        stats[regime] = {
            "median_return": round(statistics.median(returns), 1),
            "mean_return": round(statistics.mean(returns), 1),
            "sample_size": len(returns),
            "win_rate": round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1),
        }
    return stats
