"""
tracker.py — Outcome tracker backed by SQLite.

Replaces signals.json and outcomes.json with a single persistent DB.
Mount a Railway volume at /app/data to survive deploys.
"""

import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = os.getenv("DB_PATH", "data/signals.db")


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
                timestamp   TEXT    NOT NULL,
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
                signal_timestamp    TEXT    NOT NULL,
                resolved_timestamp  TEXT    NOT NULL
            );
        """)


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
    print(f"Signal logged: {regime}")


def resolve_outcomes(current_prices: dict):
    """Resolve any signals that are 30+ days old."""
    with _connect() as conn:
        unresolved = conn.execute(
            "SELECT * FROM signals WHERE resolved = 0"
        ).fetchall()

        for record in unresolved:
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
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.execute(
                "UPDATE signals SET resolved = 1 WHERE id = ?", (record["id"],)
            )
            print(
                f"Resolved signal #{record['id']}: "
                f"{record['regime']} → {actual_return_pct:+.1f}% over {age_days}d"
            )


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
            "mean_return":   round(statistics.mean(returns), 1),
            "sample_size":   len(returns),
            "win_rate":      round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1),
        }
    return stats


def get_all_signals() -> list:
    """Return all signals as a list of dicts — useful for debugging."""
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
