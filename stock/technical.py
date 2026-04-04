"""
technical.py — Indian Stock Technical Analysis
================================================
Computes 18+ technical indicators for NSE/BSE stocks using yfinance + pandas-ta.
Returns OHLCV data + indicators for chart rendering.
"""

import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger("MoneyMattersAI.Stock.Technical")


def _nse_ticker(symbol: str) -> str:
    sym = symbol.upper().strip()
    if sym.endswith(".NS") or sym.endswith(".BO"):
        return sym
    return f"{sym}.NS"


def get_technical_data(symbol: str, period: str = "1y") -> dict:
    """
    Fetches OHLCV price history and computes technical indicators.

    Args:
        symbol: NSE ticker e.g. 'RELIANCE', 'TCS.NS'
        period: yfinance period string — '3mo', '6mo', '1y', '2y'

    Returns:
        dict with keys:
          candles        — list of {date, open, high, low, close, volume}
          indicators     — dict of indicator arrays aligned to candles
          latest         — dict of latest indicator values (for summary card)
          signals        — human-readable signal strings
          error          — None if ok
    """
    ticker_sym = _nse_ticker(symbol)
    result = {
        "ticker": ticker_sym,
        "candles": [],
        "indicators": {},
        "latest": {},
        "signals": [],
        "error": None,
    }

    try:
        try:
            import pandas_ta as ta
        except ImportError:
            raise ImportError("pandas-ta not installed. Run: pip install pandas-ta")

        df = yf.download(ticker_sym, period=period, interval="1d", progress=False, auto_adjust=True)

        if df is None or df.empty:
            # BSE fallback
            ticker_sym_bo = ticker_sym.replace(".NS", ".BO")
            df = yf.download(ticker_sym_bo, period=period, interval="1d", progress=False, auto_adjust=True)
            if df.empty:
                result["error"] = f"No price data found for {symbol}"
                return result
            result["ticker"] = ticker_sym_bo

        # Flatten MultiIndex columns if present (yfinance sometimes returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()
        df.index = pd.to_datetime(df.index)

        # ── Append indicators using pandas-ta ──────────────────────────────
        # Moving averages
        df["EMA_20"]  = ta.ema(df["Close"], length=20)
        df["EMA_50"]  = ta.ema(df["Close"], length=50)
        df["SMA_20"]  = ta.sma(df["Close"], length=20)
        df["SMA_50"]  = ta.sma(df["Close"], length=50)
        df["SMA_200"] = ta.sma(df["Close"], length=200)

        # Momentum
        df["RSI_14"]  = ta.rsi(df["Close"], length=14)

        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df["MACD"]        = macd.get("MACD_12_26_9")
            df["MACD_signal"] = macd.get("MACDs_12_26_9")
            df["MACD_hist"]   = macd.get("MACDh_12_26_9")

        # Volatility
        df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

        bb = ta.bbands(df["Close"], length=20, std=2)
        if bb is not None and not bb.empty:
            df["BB_upper"] = bb.get("BBU_20_2.0")
            df["BB_mid"]   = bb.get("BBM_20_2.0")
            df["BB_lower"] = bb.get("BBL_20_2.0")

        # Volume
        df["Volume_MA_20"] = df["Volume"].rolling(20).mean()

        # ── Serialise last 252 trading days (1 year) ──────────────────────
        df_out = df.tail(252).copy()

        def f(val):
            """Float or None."""
            try:
                v = float(val)
                return None if (v != v) else round(v, 4)   # NaN check
            except Exception:
                return None

        candles = []
        for idx, row in df_out.iterrows():
            candles.append({
                "date":   idx.strftime("%Y-%m-%d"),
                "open":   f(row.get("Open")),
                "high":   f(row.get("High")),
                "low":    f(row.get("Low")),
                "close":  f(row.get("Close")),
                "volume": f(row.get("Volume")),
            })

        indicators = {}
        for col in ["EMA_20", "EMA_50", "SMA_20", "SMA_50", "SMA_200",
                    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
                    "ATR_14", "BB_upper", "BB_mid", "BB_lower", "Volume_MA_20"]:
            if col in df_out.columns:
                indicators[col] = [f(v) for v in df_out[col]]

        # ── Latest values snapshot ─────────────────────────────────────────
        last = df_out.iloc[-1]
        latest = {
            "close":        f(last.get("Close")),
            "EMA_20":       f(last.get("EMA_20")),
            "EMA_50":       f(last.get("EMA_50")),
            "SMA_20":       f(last.get("SMA_20")),
            "SMA_50":       f(last.get("SMA_50")),
            "SMA_200":      f(last.get("SMA_200")),
            "RSI_14":       f(last.get("RSI_14")),
            "MACD":         f(last.get("MACD")),
            "MACD_signal":  f(last.get("MACD_signal")),
            "MACD_hist":    f(last.get("MACD_hist")),
            "ATR_14":       f(last.get("ATR_14")),
            "BB_upper":     f(last.get("BB_upper")),
            "BB_lower":     f(last.get("BB_lower")),
        }

        # ── Human-readable signals ─────────────────────────────────────────
        signals = []
        rsi = latest.get("RSI_14")
        if rsi is not None:
            if rsi < 30:
                signals.append({"indicator": "RSI", "value": round(rsi, 1),
                                 "signal": "bullish", "note": "Oversold — potential reversal zone"})
            elif rsi > 70:
                signals.append({"indicator": "RSI", "value": round(rsi, 1),
                                 "signal": "bearish", "note": "Overbought — caution"})
            else:
                signals.append({"indicator": "RSI", "value": round(rsi, 1),
                                 "signal": "neutral", "note": "Neutral territory"})

        macd_val  = latest.get("MACD")
        macd_sig  = latest.get("MACD_signal")
        if macd_val is not None and macd_sig is not None:
            if macd_val > macd_sig:
                signals.append({"indicator": "MACD", "value": round(macd_val, 3),
                                 "signal": "bullish", "note": "MACD above signal line"})
            else:
                signals.append({"indicator": "MACD", "value": round(macd_val, 3),
                                 "signal": "bearish", "note": "MACD below signal line"})

        price = latest.get("close")
        ema20 = latest.get("EMA_20")
        ema50 = latest.get("EMA_50")
        if price and ema20 and ema50:
            if price > ema20 > ema50:
                signals.append({"indicator": "Trend (EMA)", "value": None,
                                 "signal": "bullish", "note": "Price above EMA20 & EMA50 — uptrend"})
            elif price < ema20 < ema50:
                signals.append({"indicator": "Trend (EMA)", "value": None,
                                 "signal": "bearish", "note": "Price below EMA20 & EMA50 — downtrend"})
            else:
                signals.append({"indicator": "Trend (EMA)", "value": None,
                                 "signal": "neutral", "note": "Mixed trend signals"})

        result["candles"]    = candles
        result["indicators"] = indicators
        result["latest"]     = latest
        result["signals"]    = signals

    except Exception as e:
        logger.error(f"Technical analysis error for {symbol}: {e}")
        result["error"] = str(e)

    return result
