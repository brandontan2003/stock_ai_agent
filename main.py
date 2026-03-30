"""
main.py — Agent entrypoint.

Every poll cycle:
1. Fetch live signals
2. Compute dynamic VIX thresholds (percentile-based)
3. Resolve any outcomes that are 30+ days old
4. Load learned stats from resolved outcomes
5. Classify regime
6. Alert on regime change, with message informed by learned stats
"""

import asyncio
from stock_ai_agent.src.bot import send_alert
from stock_ai_agent.src.classifier import classify_regime, build_message, get_dynamic_thresholds
from stock_ai_agent.src.config import stock_symbol, poll_interval
from stock_ai_agent.src.data import get_live_signals
from stock_ai_agent.src.tracker import log_signal, resolve_outcomes, compute_learned_stats

last_regime = None


async def run_agent():
    global last_regime
    print(f"Agent running — tracking {stock_symbol} every {poll_interval}s")
    await send_alert(f"🤖 *Agent started*\nTracking `{stock_symbol}` every {poll_interval}s")

    while True:
        try:
            # 1. Live market data
            signals = get_live_signals(stock_symbol)

            # 2. Dynamic thresholds from rolling VIX percentiles
            thresholds = get_dynamic_thresholds()

            # 3. Resolve any signals that are 30+ days old
            resolve_outcomes({stock_symbol: signals[stock_symbol]})

            # 4. Load what the agent has learned so far
            learned_stats = compute_learned_stats()

            # 5. Classify regime
            regime = classify_regime(
                signals["VIX"],
                signals["RSI"],
                signals.get("5D_CHG"),
                thresholds,
            )

            print(
                f"Regime: {regime} | VIX: {signals['VIX']} "
                f"(pct: {thresholds['current_vix_pct']}th) | "
                f"RSI: {signals['RSI']} | 5d chg: {signals.get('SPX_5D_CHG')}%"
            )

            # 6. Alert only on regime change
            if regime != last_regime:
                msg = build_message(signals, regime, stock_symbol, thresholds, learned_stats)
                await send_alert(msg)
                log_signal(regime, signals, stock_symbol)
                print(f"Alert sent: {regime}")
                last_regime = regime
            else:
                print(f"No change — regime: {regime}")

        except Exception as e:
            print(f"Error: {e}")

        await asyncio.sleep(poll_interval)


if __name__ == "__main__":
    asyncio.run(run_agent())
