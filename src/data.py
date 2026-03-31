import yfinance as yf


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def get_live_signals(stock_ticker):
    vix = yf.Ticker("^VIX").fast_info["last_price"]

    stock = yf.Ticker(stock_ticker)
    stock_price = stock.fast_info["last_price"]
    stock_hist = stock.history(period="20d")["Close"]

    rsi = compute_rsi(stock_hist).iloc[-1]

    # 5-day price change %
    change_percentage = ((stock_hist.iloc[-1] - stock_hist.iloc[-6]) / stock_hist.iloc[-6]) * 100

    return {
        "VIX": round(vix, 2),
        stock_ticker: round(stock_price, 2),
        "RSI": round(float(rsi), 2),
        "5D_CHG": round(float(change_percentage), 2),
    }


def get_trade_outcome(ticker: str, signal_date, hold_days: int = 30):
    """
    Compute entry and exit prices using trading-day logic.

    - Entry: first trading day ON or AFTER signal_date
    - Exit: exactly `hold_days` trading days after entry

    Returns:
        dict | None
    """

    history = yf.Ticker(ticker).history(period="2y")

    if history.empty:
        raise ValueError(f"No data for ticker {ticker}")

    history = history.sort_index()

    # --- Step 2: Find entry point (first valid trading day >= signal_date)
    future_data = history.loc[signal_date:]
    if future_data.empty:
        return None  # signal is in the future or beyond dataset

    entry_ts = future_data.index[0]
    entry_idx = history.index.get_loc(entry_ts)

    # --- Step 3: Compute exit index
    exit_idx = entry_idx + hold_days
    if exit_idx >= len(history):
        return None  # not enough future data yet

    exit_ts = history.index[exit_idx]

    # --- Step 4: Extract prices
    entry_price = float(history["Close"].iloc[entry_idx])
    exit_price  = float(history["Close"].iloc[exit_idx])

    # --- Step 5: Compute return
    return_pct = (exit_price - entry_price) / entry_price * 100

    return {
        "entry_date": entry_ts,
        "exit_date": exit_ts,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "return_pct": round(return_pct, 2),
        "holding_days": hold_days
    }