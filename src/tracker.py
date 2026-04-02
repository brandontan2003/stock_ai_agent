"""
tracker.py — Outcome tracker backed by SQLite.
"""
import logging
import os
import sqlite3
from datetime import datetime, timezone

import yfinance as yf

from src.classifier import classify_regime
from src.data import compute_rsi, get_trade_outcome

DB_PATH = os.getenv("DB_PATH", "data/signals.db")
logger = logging.getLogger(__name__)


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   DATETIME    NOT NULL,
                regime      TEXT    NOT NULL,
                price       REAL    NOT NULL,
                vix         REAL    NOT NULL,
                rsi         REAL    NOT NULL,
                ticker      TEXT    NOT NULL,
                resolved    INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS outcomes (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id           INTEGER NOT NULL REFERENCES signals(id),
                regime              TEXT    NOT NULL,
                entry_price         REAL    NOT NULL,
                exit_price          REAL    NOT NULL,
                actual_return_pct   REAL    NOT NULL,
                days_held           INTEGER NOT NULL,
                signal_timestamp    DATETIME    NOT NULL,
                resolved_timestamp  DATETIME    NOT NULL
            );
        """)


def _rolling_thresholds(vix_series, current_idx: int, window: int = 252) -> dict:
    """
    Compute regime thresholds using only VIX data available BEFORE current_idx.

    This is the fix for lookahead bias: each historical day is classified
    using only the percentile distribution that existed at that point in time
    — exactly as the live system does it.

    A day in March 2023 uses only VIX data up to March 2023.
    It never sees October 2023 or beyond.
    """
    start = max(0, current_idx - window)
    past_vix = vix_series.iloc[start:current_idx]  # strictly before today

    if len(past_vix) < 30:
        # Not enough history yet — fall back to hardcoded defaults
        return {"calm_max": 20.0, "fear_max": 30.0, "panic_max": 40.0, "current_vix_pct": None}

    current_vix = float(vix_series.iloc[current_idx])
    pct_rank = float((past_vix < current_vix).mean() * 100)

    return {
        "calm_max": round(float(past_vix.quantile(0.40)), 2),
        "fear_max": round(float(past_vix.quantile(0.70)), 2),
        "panic_max": round(float(past_vix.quantile(0.90)), 2),
        "current_vix_pct": round(pct_rank, 1),
    }


def backfill_signals(stock_ticker: str, days: int = 365):
    with _connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        if count > 0:
            logger.info(f"Skipping backfill — {count} signals already in DB")
            return

    # Fetch 2x the window so each day in the backfill has a full rolling
    # lookback available — avoids cold-start bias at day 1.
    fetch_days = days + 365
    logger.info(f"Backfilling {days} days of historical signals (fetching {fetch_days}d for warm-up)...")

    vix_hist = yf.Ticker("^VIX").history(period=f"{fetch_days}d")["Close"]
    stock_hist = yf.Ticker(stock_ticker).history(period=f"{fetch_days}d")["Close"]

    vix_list = list(vix_hist.items())
    stock_list = list(stock_hist.items())
    stock_by_date = {d.date(): i for i, (d, _) in enumerate(stock_list)}

    # Only log signals within the requested backfill window (most recent `days`)
    cutoff_idx = len(vix_list) - days

    last = None
    for vix_idx, (date, vix_val) in enumerate(vix_list):
        if vix_idx < cutoff_idx:
            continue  # warm-up period — accumulate history, don't log

        day = date.date()
        if day not in stock_by_date:
            continue

        stock_idx = stock_by_date[day]
        if stock_idx < 14:
            continue

        # Rolling thresholds: only past VIX data, never future
        thresholds = _rolling_thresholds(vix_hist, vix_idx)

        price = float(stock_list[stock_idx][1])
        rsi_val = float(compute_rsi(stock_hist.iloc[:stock_idx + 1]).iloc[-1])
        chg = (
            ((price - float(stock_list[stock_idx - 5][1])) / float(stock_list[stock_idx - 5][1])) * 100
            if stock_idx >= 5 else None
        )
        regime = classify_regime(float(vix_val), rsi_val, chg, thresholds)

        if regime != last:
            with _connect() as conn:
                conn.execute(
                    """INSERT INTO signals
                       (timestamp, regime, price, vix, rsi, ticker)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        date.isoformat(),
                        regime,
                        round(price, 2),
                        round(float(vix_val), 2),
                        round(rsi_val, 2),
                        stock_ticker,
                    ),
                )

            logger.info(
                f"Backfilled: {day} -> {regime} (calm<{thresholds['calm_max']}, panic>{thresholds['panic_max']})")
            last = regime

    logger.info("Backfill complete.")


def log_signal(regime: str, signals: dict, stock_ticker: str):
    """Log a new regime-change signal."""
    with _connect() as conn:
        conn.execute(
            """INSERT INTO signals (timestamp, regime, price, vix, rsi, ticker)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                regime,
                signals[stock_ticker],
                signals["VIX"],
                signals["RSI"],
                stock_ticker,
            ),
        )
    logger.info(f"Signal logged: {regime}")


def resolve_outcomes():
    """Resolve any signals that are 30+ trading days old."""
    with _connect() as conn:
        unresolved = conn.execute(
            "SELECT * FROM signals WHERE resolved = 0"
        ).fetchall()

        for record in unresolved:
            signal_time = datetime.fromisoformat(record["timestamp"])

            ticker = record["ticker"]
            entry_price = record["price"]
            result = get_trade_outcome(ticker, signal_time)
            if result is None:
                continue

            exit_date = datetime.fromisoformat(str(result["exit_date"]))
            exit_price = result["exit_price"]
            if exit_price is None:
                continue

            actual_return_pct = result["return_pct"]
            age_days = (exit_date - signal_time).days

            conn.execute(
                """INSERT INTO outcomes
                   (signal_id, regime, entry_price, exit_price,
                    actual_return_pct, days_held, signal_timestamp, resolved_timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["id"],
                    record["regime"],
                    entry_price,
                    round(exit_price, 2),
                    round(actual_return_pct, 2),
                    age_days,
                    record["timestamp"],
                    exit_date.isoformat(),
                ),
            )
            conn.execute(
                "UPDATE signals SET resolved = 1 WHERE id = ?", (record["id"],)
            )
            logger.info(f"Resolved signal #{record['id']}: "
                        f"{record['regime']} → {actual_return_pct:+.1f}% over {age_days}d")


def compute_learned_stats() -> dict:
    """Compute per-regime performance stats from resolved outcomes."""
    with _connect() as conn:
        rows = conn.execute("SELECT regime, actual_return_pct FROM outcomes").fetchall()

    if not rows:
        return {}

    from collections import defaultdict
    import statistics

    by_regime = defaultdict(list)
    for row in rows:
        by_regime[row["regime"]].append(row["actual_return_pct"])

    stats = {}
    for regime, returns in by_regime.items():
        stats[regime] = {
            "median_return": round(statistics.median(returns), 1),
            "mean_return": round(statistics.mean(returns), 1),
            "sample_size": len(returns),
            "win_rate": round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1),
        }
    return stats


def get_all_signals() -> list:
    """Return all signals as a list of dicts, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY timestamp DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_outcomes() -> list:
    """Return all resolved outcomes as a list of dicts."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM outcomes ORDER BY resolved_timestamp DESC"
        ).fetchall()
    return [dict(r) for r in rows]
