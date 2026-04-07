"""
tracker.py — Outcome tracker backed by SQLite.
"""
import logging
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd
from scipy.stats import percentileofscore
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

            CREATE TABLE IF NOT EXISTS metadata (
                key     TEXT PRIMARY KEY,
                value   TEXT NOT NULL
            );
        """)


def is_backfill_complete() -> bool:
    """Check whether a successful backfill has been recorded in metadata."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'backfill_complete'"
        ).fetchone()
    return row is not None and row["value"] == "1"


def mark_backfill_complete(days: int, ticker: str):
    """Record that backfill completed successfully, with context."""
    with _connect() as conn:
        for key, value in [
            ("backfill_complete", "1"),
            ("backfill_days", str(days)),
            ("backfill_ticker", ticker),
            ("backfill_timestamp", datetime.now(timezone.utc).isoformat()),
        ]:
            conn.execute(
                """INSERT INTO metadata (key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (key, value),
            )


def _rolling_thresholds(vix_series, current_idx: int, window: int = 60) -> dict:
    """
    Compute regime thresholds using only VIX data available BEFORE current_idx.
    Avoids lookahead bias by considering only past data.
    Returns calm_max, fear_max, panic_max, and current percentile of current VIX.
    """
    if current_idx == 0:
        # first day → no history, return defaults
        return {"calm_max": 20.0, "fear_max": 30.0, "panic_max": 40.0, "current_vix_pct": None}

    past_vix = vix_series.iloc[:current_idx].dropna()  # strictly before today
    if len(past_vix) < window:
        # insufficient data → use fallback defaults
        return {"calm_max": 20.0, "fear_max": 30.0, "panic_max": 40.0, "current_vix_pct": None}

    current_vix = vix_series.iloc[current_idx]
    calm_max = round(float(past_vix.quantile(0.40)), 2)
    fear_max = round(float(past_vix.quantile(0.70)), 2)
    panic_max = round(float(past_vix.quantile(0.90)), 2)

    # Percentile rank using scipy for accuracy
    current_vix_pct = round(percentileofscore(past_vix, current_vix, kind="weak"), 1)

    return {
        "calm_max": calm_max,
        "fear_max": fear_max,
        "panic_max": panic_max,
        "current_vix_pct": current_vix_pct,
    }


def backfill_signals(stock_ticker: str, days: int = 365):
    """
    Backfill historical regime signals into the database.

    Skips if a successful backfill is already recorded in metadata.

    Crash safety: all inserts are batched and committed in a single atomic
    transaction at the end. If the process crashes mid-loop, nothing is
    written — the DB stays clean and the backfill retries on next startup.
    The backfill_complete flag is only set after the transaction succeeds.
    """
    if is_backfill_complete():
        logger.info("Skipping backfill — already completed (metadata.backfill_complete=1)")
        return

    # Any signals present without a backfill_complete flag are orphans from
    # a previous crash. Clear them so we start clean.
    with _connect() as conn:
        orphaned = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        if orphaned > 0:
            logger.warning(
                f"Found {orphaned} orphaned signals without backfill_complete — "
                f"clearing and re-running backfill."
            )
            conn.execute("DELETE FROM signals")

    # yfinance period="Nd" is calendar days, not trading days.
    # Trading days ≈ calendar days × (252/365).
    # We need `days` trading days to log + 252 trading days of warm-up.
    # Convert to calendar days: (days + 252) ÷ (252/365) + 10-day buffer.
    total_trading_needed = days + 252
    fetch_days = int(total_trading_needed / (252 / 365)) + 10
    logger.info(
        f"Backfilling {days} trading days of signals "
        f"(fetching {fetch_days} calendar days for warm-up)..."
    )

    vix_hist = yf.Ticker("^VIX").history(period=f"{fetch_days}d")["Close"]
    stock_hist = yf.Ticker(stock_ticker).history(period=f"{fetch_days}d")["Close"]

    # Convert both to UTC
    vix_hist = vix_hist.tz_convert("UTC")
    stock_hist = stock_hist.tz_convert("UTC")

    # Convert index to dates (strip timestamp)
    vix_hist.index = vix_hist.index.normalize()
    stock_hist.index = stock_hist.index.normalize()

    # Align dates
    combined_df = pd.DataFrame({
        "vix": vix_hist,
        "price": stock_hist
    }).dropna()

    print(combined_df)
    if len(combined_df) < days:
        logger.warning(f"Not enough overlapping trading days: {len(combined_df)} < {days}")

    # Collect all inserts first — write atomically at the end.
    # Nothing touches the DB until the full loop completes successfully.
    inserts = []
    last = None

    # Only log signals for the most recent `days` trading days
    start_idx = max(0, len(combined_df) - days)

    for idx in range(start_idx, len(combined_df)):
        row = combined_df.iloc[idx]
        thresholds = _rolling_thresholds(combined_df["vix"], idx)

        price = float(row["price"])
        vix_val = float(row["vix"])
        rsi_val = float(compute_rsi(combined_df["price"].iloc[:idx + 1]).iloc[-1])

        # 5-day price change, or None if not enough history
        if idx >= 5:
            chg = ((price - combined_df["price"].iloc[idx - 5]) / combined_df["price"].iloc[idx - 5]) * 100
        else:
            chg = None

        regime = classify_regime(float(vix_val), rsi_val, chg, thresholds)
        timestamp = combined_df.index[idx]

        if regime != last:
            inserts.append((
                timestamp.to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
                regime,
                round(price, 2),
                round(float(vix_val), 2),
                round(rsi_val, 2),
                stock_ticker,
            ))
            logger.info(
                f"Backfilled {timestamp.date()} -> {regime} "
                f"(calm<{thresholds['calm_max']}, fear<{thresholds['fear_max']}, panic<{thresholds['panic_max']}) "
                f"VIX percentile={thresholds['current_vix_pct']}"
            )
            last = regime

    # Single atomic commit: all signals land together or not at all.
    with _connect() as conn:
        conn.executemany(
            """INSERT INTO signals (timestamp, regime, price, vix, rsi, ticker)
               VALUES (?, ?, ?, ?, ?, ?)""",
            inserts,
        )

    # Only flag complete AFTER the transaction succeeds.
    mark_backfill_complete(days, stock_ticker)
    logger.info(f"Backfill complete — {len(inserts)} signals written.")


def log_signal(regime: str, signals: dict, stock_ticker: str):
    """Log a new regime-change signal."""
    with _connect() as conn:
        conn.execute(
            """INSERT INTO signals (timestamp, regime, price, vix, rsi, ticker)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
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
