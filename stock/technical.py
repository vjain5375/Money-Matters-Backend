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
            import ta
        except ImportError:
            raise ImportError("ta not installed. Run: pip install ta")

        import time
        import requests

        interval = "5m" if period == "1d" else "15m" if period == "5d" else "1d"
        interval_mins = 5 if period == "1d" else 15 if period == "5d" else 1440
        
        days_map = {"1d": 2, "5d": 6, "1mo": 35, "3mo": 100, "6mo": 200, "1y": 370, "2y": 740, "5y": 1850}
        days = days_map.get(period, 370)
        
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        ticker_clean = ticker_sym.replace(".NS", "").replace(".BO", "")
        url = f"https://groww.in/v1/api/charting_service/v2/chart/exchange/NSE/segment/CASH/{ticker_clean}?intervalInMinutes={interval_mins}&startTimeInMillis={start_time}&endTimeInMillis={end_time}"
        
        logger.info(f"Fetching technical data for {ticker_sym} (period={period})")
        df = pd.DataFrame()
        
        # Try Groww API first (fastest for Indian stocks)
        try:
            logger.info(f"Attempting Groww API for {ticker_clean}...")
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            logger.info(f"Groww API response status: {res.status_code}")
            if res.status_code == 200:
                data = res.json()
                if "candles" in data and data["candles"]:
                    logger.info(f"Groww API returned {len(data['candles'])} candles")
                    df = pd.DataFrame(data["candles"], columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
                    # Groww timestamps are in seconds
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
                    df.set_index("timestamp", inplace=True)
                else:
                    logger.warning(f"Groww API returned no candles for {ticker_clean}")
            else:
                logger.warning(f"Groww API returned status {res.status_code}")
        except Exception as e:
            logger.error(f"Groww charting failed for {ticker_clean}: {e}", exc_info=True)

        # Fallback to yfinance with multiple attempts
        if df.empty:
            logger.info(f"Falling back to yfinance for {ticker_sym} (period={period}, interval={interval})")
            
            # Try NSE first
            for attempt in range(2):
                try:
                    logger.info(f"yfinance NSE attempt {attempt + 1}/2")
                    ticker_obj = yf.Ticker(ticker_sym)
                    df = ticker_obj.history(period=period, interval=interval, auto_adjust=True)
                    if not df.empty:
                        logger.info(f"yfinance NSE returned {len(df)} rows for {ticker_sym}")
                        break
                    else:
                        logger.warning(f"yfinance NSE returned empty dataframe (attempt {attempt + 1})")
                except Exception as e:
                    logger.error(f"yfinance NSE attempt {attempt + 1} failed: {e}")
                    if attempt == 0:
                        time.sleep(1)  # Wait 1 second before retry

            # BSE fallback if NSE failed
            if df.empty:
                logger.info(f"Trying BSE fallback for {ticker_sym}")
                ticker_sym_bo = ticker_sym.replace(".NS", ".BO")
                
                for attempt in range(2):
                    try:
                        logger.info(f"yfinance BSE attempt {attempt + 1}/2")
                        ticker_obj = yf.Ticker(ticker_sym_bo)
                        df = ticker_obj.history(period=period, interval=interval, auto_adjust=True)
                        if not df.empty:
                            logger.info(f"yfinance BSE returned {len(df)} rows for {ticker_sym_bo}")
                            result["ticker"] = ticker_sym_bo
                            break
                        else:
                            logger.warning(f"yfinance BSE returned empty dataframe (attempt {attempt + 1})")
                    except Exception as e:
                        logger.error(f"yfinance BSE attempt {attempt + 1} failed: {e}")
                        if attempt == 0:
                            time.sleep(1)
                    
            if df.empty:
                error_msg = f"No price data found for {symbol} from any source (Groww, yfinance NSE, yfinance BSE). This may be due to network restrictions or invalid ticker."
                logger.error(error_msg)
                result["error"] = error_msg
                return result

            # Flatten MultiIndex columns if present (yfinance sometimes returns them)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna()
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert("Asia/Kolkata")

        # ── Append indicators using ta ──────────────────────────────
        # Moving averages
        df["EMA_20"]  = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
        df["EMA_50"]  = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
        df["SMA_20"]  = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
        df["SMA_50"]  = ta.trend.SMAIndicator(df["Close"], window=50).sma_indicator()
        df["SMA_200"] = ta.trend.SMAIndicator(df["Close"], window=200).sma_indicator()

        # Momentum
        df["RSI_14"]  = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

        macd = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD"]        = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"]   = macd.macd_diff()

        # Volatility
        df["ATR_14"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()

        bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_mid"]   = bb.bollinger_mavg()
        df["BB_lower"] = bb.bollinger_lband()

        # Volume
        df["Volume_MA_20"] = df["Volume"].rolling(20).mean()

        # ── Serialise last 252 trading days (1 year) ──────────────────────
        df_out = df.copy()

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
                "date":   str(idx) if hasattr(idx, 'hour') else idx.strftime("%Y-%m-%d"),
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
