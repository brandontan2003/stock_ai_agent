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
