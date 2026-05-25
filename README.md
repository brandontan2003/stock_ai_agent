# Market Signal Bot

A live market monitoring system that tracks the VIX index, classifies the current volatility regime, and sends intelligent alerts to Telegram. Over time, it replaces generic priors with its own measured track record.

---

## What it does

The bot polls market data on a configurable interval, classifies the current VIX reading into one of five volatility regimes, and sends a Telegram alert when the regime changes. Every signal is logged to SQLite. Thirty trading days later, the system fetches the actual exit price and records the return — building a per-regime track record that replaces hardcoded priors with evidence over time.

---

## Architecture

```
main.py
├── src/classifier.py    — Rolling percentile regime classification
├── src/tracker.py       — SQLite signal logger + outcome resolver
├── src/reasoner.py      — Two-stage LLM alert gate (Anthropic API)
├── src/data.py          — Market data layer (yfinance)
├── src/bot.py           — Telegram dispatcher
├── src/dashboard.py     — Streamlit live dashboard
└── src/config.py        — Environment config
```

### Core loop (`main.py`)

On startup:
1. Initialises the SQLite database
2. Backfills one year of historical signals (with lookahead bias fix — see below)
3. Seeds `last_regime` from the database so restarts don't re-alert

Every poll cycle:
1. Resolves any signals that are 30+ trading days old
2. Fetches live VIX, RSI, and 5-day price change
3. Classifies the current regime using rolling percentile thresholds
4. If the regime has changed — runs the two-stage LLM gate
5. If the LLM approves — sends the Telegram alert and logs the signal

---

## Components

### `classifier.py`

Classifies the current VIX reading into one of five regimes:

| Regime | Threshold |
|--------|-----------|
| 🟢 CALM | VIX below 40th percentile |
| 🟡 FEAR | VIX 40th–70th percentile |
| 🟠 FEAR (confirmed) | FEAR + RSI below 40 or 5-day decline |
| 🔴 PANIC | VIX 70th–90th percentile |
| 🚨 EXTREME PANIC | VIX above 90th percentile |

Thresholds are computed from the trailing 252 trading days of VIX data — not hardcoded numbers. VIX at 30 after a year of calm is extreme. VIX at 30 after months of elevated volatility is ordinary. Fixed thresholds cannot tell the difference; rolling percentiles can.

Also exposes `build_message()` — a template-based fallback alert builder used when the LLM call fails.

### `tracker.py`

SQLite-backed outcome tracker. Three tables:

- `signals` — every regime change: timestamp, regime, entry price, VIX, RSI, ticker, resolved flag
- `outcomes` — resolved results: exit price, actual return %, days held, resolved timestamp
- `metadata` — system state: `backfill_complete` flag, backfill parameters

**Backfill** runs on startup if `backfill_complete` is not set. Fetches enough history for a genuine 252-trading-day warm-up window, computes rolling thresholds for each historical day using only prior data, and writes all signals atomically in a single transaction. If the process crashes mid-backfill, the database stays clean and retries on next startup.

**Outcome resolution** runs every poll cycle. Signals older than 30 trading days (excluding weekends and holidays) are fetched, exit price recorded, return computed, and marked resolved.

**Learned statistics** — `compute_learned_stats()` returns per-regime median return, mean return, win rate, and sample size from resolved outcomes. The classifier uses these in place of hardcoded priors once `MIN_SAMPLES = 3` outcomes are available per regime.

### `reasoner.py`

Two-stage LLM reasoning layer using the Anthropic API (`claude-sonnet-4-20250514`).

**Stage 1 — Noise gate.** Given the current signal, the last 10 regime changes, dynamic thresholds, and the system's own track record, the model decides whether this regime change is worth alerting or is oscillation noise (e.g. three transitions in one session as VIX oscillates around a threshold boundary).

**Stage 2 — Alert authoring.** If the signal passes the gate, the model writes the Telegram alert using only numbers that appear in the context it was given. It is explicitly forbidden from inventing statistics or citing generic VIX lore.

If the LLM call fails entirely, the system falls back to `build_message()` — a template-built alert. The reasoning layer is an enhancement, not a dependency. It cannot break the system.

### `data.py`

- `get_live_signals(ticker)` — fetches current VIX, stock price, RSI, and 5-day price change
- `compute_rsi(series, period=14)` — standard RSI calculation
- `get_trade_outcome(ticker, signal_time)` — fetches the price 30 trading days after the signal timestamp, excluding weekends and holidays

### `bot.py`

Thin Telegram dispatcher. Sends formatted messages using `python-telegram-bot`. Called by `main.py` after the LLM gate approves an alert.

### `dashboard.py`

Streamlit dashboard running alongside the main loop via `asyncio.gather`. Shows:
- Live signal history with regime, VIX, RSI, and entry price
- Resolved outcomes with actual returns
- Per-regime learned statistics

---

## Key design decisions

### Lookahead bias fix

The backfill initially computed thresholds from the full year of VIX data, then applied those thresholds to every historical day. A signal in March 2024 was being judged against percentile boundaries that included volatility from months that hadn't happened yet.

A backtest that uses future data to classify past events is a hallucination of performance.

**Fix:** Each historical day's thresholds are computed strictly from VIX data that preceded it using `vix_series.iloc[start:current_idx]` — never from future data.

**Warm-up window:** To avoid cold-start error at day one, the system fetches enough data for 252 trading days of warm-up before the logged window begins. The warm-up data is never logged — it exists purely to seed the percentile distribution.

**Calendar vs trading day correction:** `yfinance period="Nd"` measures calendar days, not trading days. 730 calendar days returns only ~504 trading days. The fetch window is calculated as:

```python
fetch_days = int((days + 252) / (252 / 365)) + 10
```

### Atomic backfill transaction

All backfill inserts are collected in a list and committed in a single `executemany` transaction. The `backfill_complete` flag in the `metadata` table is only set after the transaction succeeds. A crash mid-backfill leaves the database unchanged — the next startup clears any orphaned signals and retries from scratch.

The old guard `if count > 0: return` made partial backfills permanently unrecoverable. The new guard checks `backfill_complete` explicitly.

### NaN-safe percentile rank

```python
# Before (wrong with data gaps):
pct_rank = float((past_vix < current_vix).mean() * 100)

# After:
pct_rank = float((past_vix.dropna() < current_vix).mean() * 100)
```

`NaN < anything` evaluates `False` in pandas, silently deflating the percentile rank when the series has gaps. `pandas quantile()` already skips NaN, so only `pct_rank` needed the fix.

### MIN_SAMPLES = 3

The system requires at least 3 resolved outcomes per regime before switching from priors to learned statistics. Three is not statistically meaningful — it is a floor, not a confidence threshold. The sample size is disclosed in every alert.

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for deployment)
- Telegram bot token and chat ID
- Anthropic API key

### Environment variables

Copy `.env.example` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `STOCK_SYMBOL` | Ticker to track (e.g. `VOO`) |
| `BOT_TOKEN` | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Telegram chat ID to send alerts to |
| `POLL_INTERVAL` | Polling interval in seconds (e.g. `600`) |
| `ANTHROPIC_API_KEY` | Anthropic API key for the reasoning layer |
| `HISTORY_WINDOW` | Days of history for backfill (default: `365`) |

### Run locally

```bash
pip install -r requirements.txt
python main.py
```

### Run with Docker

```bash
docker compose up --build
```

### Deploy to Railway

1. Push to GitHub
2. Create a new Railway project from the repo
3. Add a persistent volume mounted at `/app/data` (SQLite lives here)
4. Set all environment variables in the Railway dashboard
5. Deploy

---

## Known limitations

- **yfinance reliability** — no rate-limit handling. If yfinance goes down, the system logs the error and skips that poll cycle. Data gaps appear in the dashboard as missing polls, not false signals.
- **RSI window** — RSI is computed on 20 days of stock history, giving approximately 6 meaningful values at the end of the series. Directionally useful as a confirming signal only, not a primary one.
- **MIN_SAMPLES = 3** — most regimes run on priors for the first several months of live operation. Regime transitions occur roughly weekly; 3 resolved outcomes per regime takes time to accumulate.
- **Single ticker** — the system tracks one ticker (default: VOO). No cross-validation across correlated assets.
- **SQLite** — appropriate for a single-process, low write-volume system. If extended to multiple tickers or users, migrate to Postgres.

---

## Project structure

```
.
├── main.py                 # Entry point — async main loop
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── src/
    ├── __init__.py
    ├── bot.py              # Telegram dispatcher
    ├── classifier.py       # Regime classifier + message builder
    ├── config.py           # Environment config
    ├── dashboard.py        # Streamlit dashboard
    ├── data.py             # Market data layer
    ├── reasoner.py         # Two-stage LLM gate
    └── tracker.py          # SQLite signal logger + outcome resolver
```
