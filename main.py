import asyncio
import logging
import traceback

from src.bot import send_alert
from src.classifier import classify_regime, build_message, get_dynamic_thresholds
from src.config import stock_symbol, poll_interval
from src.dashboard import start_dashboard
from src.data import get_live_signals
from src.reasoner import should_alert, build_reasoned_message
from src.tracker import init_db, log_signal, resolve_outcomes, compute_learned_stats, backfill_signals, get_all_signals


last_regime = None
logger = logging.getLogger(__name__)

async def run_agent():
    global last_regime

    # 1. Initialise DB
    init_db()

    # 2. Backfill 1 year of historical signals on first run
    backfill_signals(stock_symbol, days=365)

    # 3. Seed last_regime from DB so restarts don't re-alert
    existing = get_all_signals()
    if existing:
        last_regime = existing[0]["regime"]
        logger.info(f"Resuming — last known regime: {last_regime}")

    logger.info(f"Agent running — tracking {stock_symbol} every {poll_interval}s")
    await send_alert(f"🤖 *Agent started*\nTracking `{stock_symbol}` every {poll_interval}s")

    while True:
        try:
            resolve_outcomes()

            signals = get_live_signals(stock_symbol)
            if not signals:
                logger.info(f"No live signals found for {stock_symbol}, skipping this cycle")
                await asyncio.sleep(poll_interval)
                continue

            thresholds = get_dynamic_thresholds()
            learned_stats = compute_learned_stats()
            recent_signals = get_all_signals()  # already sorted DESC

            regime = classify_regime(
                signals["VIX"], signals["RSI"],
                signals.get("5D_CHG"), thresholds,
            )
            logger.info(f"Regime: {regime} | VIX: {signals['VIX']} | RSI: {signals['RSI']}")

            if regime != last_regime:
                # ── Stage 1: LLM gate ─────────────────────────────────────────
                # The LLM reasons over recent history to decide if this
                # regime change is signal or noise before we alert.
                send, rationale = should_alert(
                    signals, regime, thresholds, learned_stats, recent_signals
                )
                logger.info(f"[LLM gate] send={send} | {rationale}")

                if send:
                    # ── Stage 2: LLM reasoning ────────────────────────────────
                    # The LLM writes a genuine analyst note grounded in the
                    # system's own track record — not hardcoded priors.
                    msg = build_reasoned_message(
                        signals, regime, stock_symbol,
                        thresholds, learned_stats, recent_signals
                    )
                    if msg is None:
                        # Graceful fallback to template if LLM call fails
                        msg = build_message(signals, regime, stock_symbol, thresholds, learned_stats)

                    await send_alert(msg)
                    log_signal(regime, signals, stock_symbol)
                    logger.info(f"Alert sent: {regime}")
                else:
                    logger.info(f"[LLM gate] Suppressed: {rationale}")

                # Update last_regime whether we alerted or suppressed —
                # suppression is a decision, not an oversight.
                last_regime = regime
            else:
                logger.info(f"No change — regime: {regime}")

        except Exception as e:
            logger.error(f"Error: {e}")
            traceback.print_exc()
        await asyncio.sleep(poll_interval)


async def main():
    await asyncio.gather(
        start_dashboard(),
        run_agent(),
    )

if __name__ == "__main__":
    asyncio.run(main())
