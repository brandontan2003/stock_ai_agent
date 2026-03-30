import asyncio

from src.bot import send_alert
from src.classifier import classify_regime, build_message, get_dynamic_thresholds
from src.config import stock_symbol, poll_interval
from src.dashboard import start_dashboard
from src.data import get_live_signals
from src.tracker import init_db, log_signal, resolve_outcomes, compute_learned_stats

last_regime = None


async def run_agent():
    global last_regime
    init_db()
    print(f"Agent running — tracking {stock_symbol} every {poll_interval}s")
    await send_alert(f"🤖 *Agent started*\nTracking `{stock_symbol}` every {poll_interval}s")

    while True:
        try:
            signals = get_live_signals(stock_symbol)
            thresholds = get_dynamic_thresholds()
            resolve_outcomes({stock_symbol: signals[stock_symbol]})
            learned_stats = compute_learned_stats()
            regime = classify_regime(
                signals["VIX"], signals["RSI"],
                signals.get("5D_CHG"), thresholds,
            )
            print(f"Regime: {regime} | VIX: {signals['VIX']} | RSI: {signals['RSI']}")
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


async def main():
    await asyncio.gather(
        start_dashboard(),
        run_agent(),
    )

if __name__ == "__main__":
    asyncio.run(main())
