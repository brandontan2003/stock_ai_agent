"""
dashboard.py — Lightweight read-only web dashboard.
Exposes /signals and /outcomes as JSON endpoints.
Serves a simple HTML dashboard at /.
"""
import logging
from datetime import datetime

from aiohttp import web
from src.tracker import get_all_signals, get_all_outcomes, compute_learned_stats
import json

logger = logging.getLogger(__name__)
PORT = 8080

async def handle_index(request):
    date_format = "%d/%m/%Y"
    timestamp_format = "%Y-%m-%dT%H:%M:%S%z"
    signals  = get_all_signals()
    outcomes = get_all_outcomes()
    stats    = compute_learned_stats()

    rows = "".join(
        f"<tr><td>{s['id']}</td><td>{datetime.strptime(s['timestamp'], timestamp_format).strftime(date_format)}</td>"
        f"<td>{s['regime']}</td><td>{s['vix']}</td>"
        f"<td>{s['rsi']}</td><td>{s['price']}</td>"
        f"<td>{'✅' if s['resolved'] else '⏳'}</td></tr>"
        for s in signals
    )

    outcome_rows = "".join(
        f"<tr><td>{o['regime']}</td>"
        f"<td>{datetime.strptime(o['signal_timestamp'], timestamp_format).strftime(date_format)}</td>"
        f"<td>{datetime.strptime(o['resolved_timestamp'], timestamp_format).strftime(date_format)}</td>"
        f"<td>{o['entry_price']}</td>"
        f"<td>{o['exit_price']}</td>"
        f"<td style='color:{'green' if o['actual_return_pct'] > 0 else 'red'}'>"
        f"{o['actual_return_pct']:+.1f}%</td>"
        f"<td>{o['days_held']}d</td></tr>"
        for o in outcomes
    )

    stats_rows = "".join(
        f"<tr><td>{regime}</td>"
        f"<td>{s['median_return']:+.1f}%</td>"
        f"<td>{s['win_rate']:.0f}%</td>"
        f"<td>{s['sample_size']}</td></tr>"
        for regime, s in stats.items()
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Market Signal Bot</title>
  <style>
    body {{ font-family: monospace; background: #0f1923; color: #00d4ff; padding: 2rem; }}
    h2   {{ color: #ffffff; margin-top: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
    th   {{ text-align: left; border-bottom: 1px solid #00d4ff44; padding: 6px 12px; color: #ffffff; }}
    td   {{ padding: 6px 12px; border-bottom: 1px solid #ffffff11; }}
    .empty {{ color: #ffffff44; font-style: italic; }}
  </style>
</head>
<body>
  <h1>Market Signal Bot</h1>

  <h2>Learned stats</h2>
  <table>
    <tr><th>Regime</th><th>Median return</th><th>Win rate</th><th>Signals</th></tr>
    {stats_rows or '<tr><td colspan=4 class="empty">No resolved outcomes yet</td></tr>'}
  </table>

  <h2>Signal log</h2>
  <table>
    <tr><th>ID</th><th>Time (UTC)</th><th>Regime</th><th>VIX</th><th>RSI</th><th>Price</th><th>Resolved</th></tr>
    {rows or '<tr><td colspan=7 class="empty">No signals yet</td></tr>'}
  </table>

  <h2>Outcomes</h2>
  <table>
    <tr><th>Regime</th><th>Entry Date</th><th>Exit Date</th><th>Entry</th><th>Exit</th><th>Return</th><th>Days Held</th></tr>
    {outcome_rows or '<tr><td colspan=5 class="empty">No outcomes yet</td></tr>'}
  </table>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


async def handle_signals(request):
    return web.Response(
        text=json.dumps(get_all_signals(), indent=2),
        content_type="application/json"
    )

async def handle_outcomes(request):
    return web.Response(
        text=json.dumps(get_all_outcomes(), indent=2),
        content_type="application/json"
    )

async def start_dashboard():
    app = web.Application()
    app.router.add_get("/",         handle_index)
    app.router.add_get("/signals",  handle_signals)
    app.router.add_get("/outcomes", handle_outcomes)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    logger.info(f"Dashboard running on port {PORT}")
