"""
reasoner.py — LLM-powered two-stage gate for regime alerts.

Stage 1 (filter): Given the current signal and recent history, should this
                  regime change trigger an alert at all? Returns True/False + rationale.

Stage 2 (reason): If it passes the gate, produce a structured analyst note
                  that replaces the template-built message.

The LLM receives:
  - Current signals (VIX, RSI, price, 5d change)
  - Dynamic thresholds + current VIX percentile
  - Learned outcome stats from the DB (the system's own track record)
  - Last N signals from the DB (recent regime history)

This is what makes the system agentic: the LLM reasons over the system's
own logged history to decide whether to act and what to say.
"""

import json
import logging

import httpx

from src.config import api_key, history_window

logger = logging.getLogger(__name__)
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"


def _call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    response = httpx.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["content"][0]["text"].strip()


def _build_context(signals: dict, regime: str, thresholds: dict,
                   learned_stats: dict, recent_signals: list) -> str:
    """Serialize all system context into a compact JSON block for the LLM."""

    # Trim recent_signals to the window and strip heavy fields
    trimmed_history = [
        {
            "timestamp": s["timestamp"][:10],
            "regime": s["regime"],
            "vix": s["vix"],
            "rsi": s["rsi"],
            "price": s["price"],
            "resolved": bool(s["resolved"]),
        }
        for s in recent_signals[:history_window]
    ]

    context = {
        "current_signal": {
            "regime": regime,
            "vix": signals.get("VIX"),
            "rsi": signals.get("RSI"),
            "price_5d_change_pct": signals.get("5D_CHG"),
        },
        "dynamic_thresholds": thresholds,
        "system_track_record": learned_stats,
        "recent_signal_history": trimmed_history,
    }
    return json.dumps(context, indent=2)


def should_alert(signals: dict, regime: str, thresholds: dict,
                 learned_stats: dict, recent_signals: list) -> tuple[bool, str]:
    """
    Stage 1: LLM decides whether this regime change is worth alerting.

    Returns (should_send: bool, rationale: str).

    The LLM can suppress an alert if e.g.:
      - The regime just oscillated back from a brief spike
      - The signal looks like noise relative to recent history
      - The VIX move is marginal at the boundary
    """
    system_prompt = (
        "You are a disciplined risk-management filter for a market monitoring system. "
        "Your job is to prevent alert fatigue by suppressing low-signal regime changes. "
        "You will be given the current market signal, recent regime history, dynamic "
        "thresholds, and this system's own historical track record. "
        "\n\n"
        "Suppress the alert if:\n"
        "  - The regime just changed back within 2 signals (oscillation noise)\n"
        "  - VIX is within 0.5 points of a threshold boundary (borderline reading)\n"
        "  - The new regime has no meaningfully different action implication vs the prior\n"
        "\n"
        "Send the alert if the regime change is:\n"
        "  - A clear directional shift (e.g. calm → panic, or extreme_panic → fear)\n"
        "  - Confirmed by RSI and 5d price change pointing the same direction\n"
        "  - At a VIX level with a strong track record in this system's own data\n"
        "\n"
        "Respond ONLY with a JSON object, no markdown, no preamble:\n"
        '{"send": true or false, "rationale": "one sentence"}'
    )

    context = _build_context(signals, regime, thresholds, learned_stats, recent_signals)
    user_prompt = f"Market context:\n{context}\n\nShould this regime change trigger an alert?"

    try:
        raw = _call_claude(system_prompt, user_prompt, max_tokens=150)
        # Strip any accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return bool(parsed["send"]), str(parsed.get("rationale", ""))
    except Exception as e:
        # Fail open: if reasoning breaks, send the alert
        logger.exception(f"[reasoner] Stage 1 failed ({e}), defaulting to send=True")
        return True, "filter unavailable — defaulting to send"


def build_reasoned_message(signals: dict, regime: str, stock_symbol: str,
                           thresholds: dict, learned_stats: dict,
                           recent_signals: list) -> str:
    """
    Stage 2: LLM writes the alert message as a genuine analyst note.

    The message must reference the system's own track record (not generic
    market wisdom), explain what about the current signal is notable, and
    give a concrete, hedged action implication.
    """
    system_prompt = (
        "You are writing a concise Telegram alert for a personal portfolio monitoring "
        "system. The investor dollar-cost averages $100/month into VOO and uses this "
        "system to decide whether to deploy extra capital during volatility spikes.\n\n"
        "Write an analyst note that:\n"
        "  1. States the regime and what the VIX percentile means in plain English\n"
        "  2. References this system's OWN track record for this regime (from system_track_record) "
        "— not generic historical stats. If insufficient data, say so honestly.\n"
        "  3. Notes any confirming or conflicting signals (RSI, 5d price change)\n"
        "  4. Gives one concrete, hedged action implication for a DCA investor\n"
        "  5. Ends with: _Not financial advice._\n\n"
        "Format for Telegram Markdown. Keep it under 200 words. "
        "Do not invent statistics. Do not use generic VIX lore. "
        "Only cite numbers that appear in the context you are given."
    )

    context = _build_context(signals, regime, thresholds, learned_stats, recent_signals)
    user_prompt = (
        f"Generate a Telegram alert for ticker {stock_symbol}.\n\n"
        f"Market context:\n{context}"
    )

    try:
        return _call_claude(system_prompt, user_prompt, max_tokens=400)
    except Exception as e:
        logger.exception(f"[reasoner] Stage 2 failed ({e}), falling back to template message")
        return None  # caller falls back to build_message()
