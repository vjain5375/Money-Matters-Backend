"""
router.py — FastAPI Router for Stock Analysis Endpoints
=========================================================
Registers all /stock/* routes into the main FastAPI app.

Routes:
    GET /stock/search          → NSE stock autocomplete
    GET /stock/fundamentals    → Fundamental metrics + Piotroski + Altman Z
    GET /stock/technical       → OHLCV + 18 indicators
    GET /stock/sentiment       → News from Indian RSS feeds + Gemini sentiment
    GET /stock/predict         → Ensemble Buy/Hold/Sell signal
    GET /stock/full            → All 4 data types in one call (for dashboard)
"""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException, Query, status

from stock.fundamentals  import get_fundamentals
from stock.technical     import get_technical_data
from stock.nse_search    import search_stocks
from stock.sentiment     import get_sentiment
from stock.predictor     import predict

logger = logging.getLogger("MoneyMattersAI.Stock.Router")

router = APIRouter(prefix="/stock", tags=["Stock Analysis"])

# Create a shared executor for parallel tasks
executor = ThreadPoolExecutor(max_workers=10)


def _nse_symbol(symbol: str) -> str:
    """Normalise ticker to NSE format."""
    s = symbol.upper().strip()
    if s.endswith(".NS") or s.endswith(".BO"):
        return s
    return f"{s}.NS"


# ── GET /stock/search ─────────────────────────────────────────────────────
@router.get("/search", summary="Search Indian Stocks")
async def search(q: str = Query(..., min_length=1, description="Search query")):
    """
    Autocomplete search for Indian NSE/BSE stocks.
    Returns list of {symbol, name, exchange, ticker}.
    """
    results = search_stocks(q, limit=10)
    return {"query": q, "results": results, "count": len(results)}


# ── GET /stock/fundamentals/{ticker} ──────────────────────────────────────
@router.get("/fundamentals/{ticker}", summary="Get Fundamental Metrics")
async def fundamentals(ticker: str):
    """
    Returns comprehensive fundamental data for an Indian stock.
    Includes P/E, P/B, ROE, Debt/Equity, Piotroski F-Score, Altman Z-Score.
    """
    symbol = _nse_symbol(ticker)
    logger.info(f"GET /stock/fundamentals/{symbol}")
    data = get_fundamentals(symbol)
    if data.get("error") and not data.get("company_name"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not fetch data for {ticker}: {data['error']}",
        )
    return data


# ── GET /stock/technical/{ticker} ─────────────────────────────────────────
@router.get("/technical/{ticker}", summary="Get Technical Indicators")
async def technical(
    ticker: str,
    period: str = Query("1y", description="Data period: 3mo, 6mo, 1y, 2y"),
):
    """
    Returns OHLCV candle data + 18 computed technical indicators
    (EMA/SMA, RSI, MACD, ATR, Bollinger Bands, etc.)
    """
    symbol = _nse_symbol(ticker)
    valid_periods = ["1mo", "3mo", "6mo", "1y", "2y"]
    if period not in valid_periods:
        period = "1y"
    logger.info(f"GET /stock/technical/{symbol}?period={period}")
    data = get_technical_data(symbol, period=period)
    if data.get("error"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Technical data failed for {ticker}: {data['error']}",
        )
    return data


# ── GET /stock/sentiment/{ticker} ─────────────────────────────────────────
@router.get("/sentiment/{ticker}", summary="Get News Sentiment")
async def sentiment(ticker: str, company_name: str = Query(None)):
    """
    Fetches latest news from Indian financial RSS feeds (ET, Moneycontrol,
    LiveMint, Business Standard) and scores sentiment via Gemini API.
    """
    symbol = _nse_symbol(ticker)
    logger.info(f"GET /stock/sentiment/{symbol}")
    data = get_sentiment(symbol, company_name=company_name)
    return data


# ── GET /stock/predict/{ticker} ───────────────────────────────────────────
@router.get("/predict/{ticker}", summary="Get Buy/Hold/Sell Signal")
async def predict_signal(ticker: str):
    """
    Ensemble predictor combining fundamental + technical + sentiment signals.
    Returns: signal (BUY/HOLD/SELL), confidence %, score breakdown, reasons.
    
    ⚠️ NOT financial advice — for educational reference only.
    """
    symbol = _nse_symbol(ticker)
    logger.info(f"GET /stock/predict/{symbol}")

    # Fetch all three data sources
    f_data = get_fundamentals(symbol)
    t_data = get_technical_data(symbol, period="6mo")
    company_name = f_data.get("company_name")
    s_data = get_sentiment(symbol, company_name=company_name)

    prediction = predict(f_data, t_data, s_data)
    prediction["ticker"] = symbol
    prediction["company_name"] = company_name
    return prediction


# ── GET /stock/full/{ticker} ──────────────────────────────────────────────
@router.get("/full/{ticker}", summary="All Data in One Call")
async def full_analysis(ticker: str, period: str = Query("1y"), company_name: str = None):
    """
    Combines fundamentals, technicals, and sentiment into one report.
    Fetches data in parallel to reduce latency.
    """
    symbol = _nse_symbol(ticker)
    loop = asyncio.get_event_loop()
    
    try:
        # Step 1: Fetch Fundamental and Technical data in parallel
        # They are independent and the main bottlenecks
        tasks = [
            loop.run_in_executor(executor, get_fundamentals, symbol),
            loop.run_in_executor(executor, get_technical_data, symbol, period)
        ]
        
        f_data, t_data = await asyncio.gather(*tasks)
        
        # Step 2: Fetch Sentiment (depends on company_name for better accuracy)
        # Note: We still fetch it even if fundamentals failed
        comp_name = company_name or f_data.get("company_name")
        s_data = await loop.run_in_executor(executor, get_sentiment, symbol, comp_name)
        
        # Step 3: Predict
        p_data = predict(f_data, t_data, s_data)
        
        # Step 4: Fallback — patch price & name from technicals if fundamentals failed
        if f_data.get("error") or f_data.get("current_price") is None:
            # Use latest close from technical candles as price fallback
            if t_data.get("candles"):
                last_close = t_data["candles"][-1].get("close")
                if last_close is not None:
                    f_data["current_price"] = last_close
            # Use ticker as company name fallback
            if not f_data.get("company_name"):
                f_data["company_name"] = symbol.replace(".NS", "").replace(".BO", "")
        
        return {
            "symbol": symbol,
            "company_name": f_data.get("company_name") or symbol,
            "ticker": symbol,
            "fundamentals": f_data,
            "technicals": t_data,
            "sentiment": s_data,
            "prediction": p_data
        }
    except Exception as e:
        logger.error(f"Full analysis failed for {symbol}: {str(e)}", exc_info=True)
        return {
            "symbol": symbol,
            "error": str(e),
            "message": "Failed to complete full stock analysis."
        }


# ── GET /stock/market-overview ────────────────────────────────────────────
@router.get("/market-overview", summary="Market Overview: Indices + Gainers/Losers")
async def market_overview():
    """
    Returns:
      - indices: Nifty 50, Sensex, Nifty Bank, Nifty IT
      - top_gainers: Top 5 gainers from popular NSE stocks (today)
      - top_losers: Top 5 losers from popular NSE stocks (today)
      - trending: Popular / most-searched stocks
    """
    import yfinance as yf
    import pandas as pd

    # Nifty indices + benchmark tickers
    INDEX_TICKERS = {
        "NIFTY 50":    "^NSEI",
        "SENSEX":      "^BSESN",
        "NIFTY Bank":  "^NSEBANK",
        "NIFTY IT":    "^CNXIT",
    }

    # Popular NSE stocks to compute gainers/losers from
    POPULAR = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "WIPRO.NS", "BAJAJFINSV.NS", "AXISBANK.NS", "LT.NS",
        "KOTAKBANK.NS", "MARUTI.NS", "TATAMOTORS.NS", "HCLTECH.NS", "ASIANPAINT.NS",
        "SUNPHARMA.NS", "TITAN.NS", "BHARTIARTL.NS", "NTPC.NS", "POWERGRID.NS",
        "ADANIENT.NS", "ULTRACEMCO.NS", "HINDUNILVR.NS", "ITC.NS", "ZOMATO.NS",
        "NESTLEIND.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "COALINDIA.NS", "ONGC.NS",
    ]

    TRENDING_DISPLAY = [
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries"},
        {"symbol": "TCS.NS",      "name": "TCS"},
        {"symbol": "INFY.NS",     "name": "Infosys"},
        {"symbol": "ZOMATO.NS",   "name": "Zomato"},
        {"symbol": "ADANIENT.NS", "name": "Adani Ent."},
        {"symbol": "TATAMOTORS.NS","name": "Tata Motors"},
        {"symbol": "SBIN.NS",     "name": "SBI"},
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank"},
    ]

    loop = asyncio.get_event_loop()

    def _fetch_indices():
        results = {}
        for name, ticker in INDEX_TICKERS.items():
            try:
                hist = yf.download(ticker, period="2d", interval="1d", progress=False, auto_adjust=True)
                if hist is None or hist.empty:
                    continue
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                hist = hist.dropna()
                if len(hist) >= 2:
                    prev = float(hist["Close"].iloc[-2])
                    curr = float(hist["Close"].iloc[-1])
                    chg = curr - prev
                    chg_pct = (chg / prev) * 100 if prev else 0
                    results[name] = {
                        "price": round(curr, 2),
                        "change": round(chg, 2),
                        "change_pct": round(chg_pct, 2),
                    }
                elif len(hist) == 1:
                    curr = float(hist["Close"].iloc[-1])
                    results[name] = {"price": round(curr, 2), "change": 0, "change_pct": 0}
            except Exception as e:
                logger.warning(f"Index fetch failed for {name}: {e}")
        return results

    def _fetch_movers():
        movers = []
        try:
            # Batch download all popular stocks — single request
            raw = yf.download(
                POPULAR, period="2d", interval="1d",
                progress=False, auto_adjust=True, group_by="ticker"
            )
            for ticker in POPULAR:
                try:
                    if ticker in raw.columns.get_level_values(0):
                        df = raw[ticker].dropna()
                    else:
                        continue
                    if len(df) < 2:
                        continue
                    prev = float(df["Close"].iloc[-2])
                    curr = float(df["Close"].iloc[-1])
                    if prev == 0:
                        continue
                    chg_pct = ((curr - prev) / prev) * 100
                    movers.append({
                        "symbol": ticker,
                        "name": ticker.replace(".NS", ""),
                        "price": round(curr, 2),
                        "change": round(curr - prev, 2),
                        "change_pct": round(chg_pct, 2),
                    })
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Batch movers fetch failed: {e}")
        return movers

    def _fetch_trending_prices():
        results = []
        try:
            tickers = [s["symbol"] for s in TRENDING_DISPLAY]
            raw = yf.download(
                tickers, period="2d", interval="1d",
                progress=False, auto_adjust=True, group_by="ticker"
            )
            for item in TRENDING_DISPLAY:
                t = item["symbol"]
                try:
                    if t in raw.columns.get_level_values(0):
                        df = raw[t].dropna()
                    else:
                        results.append({**item, "price": None, "change_pct": 0})
                        continue
                    if len(df) >= 2:
                        prev = float(df["Close"].iloc[-2])
                        curr = float(df["Close"].iloc[-1])
                        chg_pct = ((curr - prev) / prev) * 100 if prev else 0
                        results.append({
                            **item,
                            "price": round(curr, 2),
                            "change_pct": round(chg_pct, 2),
                        })
                    elif len(df) == 1:
                        results.append({**item, "price": round(float(df["Close"].iloc[-1]), 2), "change_pct": 0})
                except Exception:
                    results.append({**item, "price": None, "change_pct": 0})
        except Exception as e:
            logger.warning(f"Trending prices fetch failed: {e}")
        return results

    try:
        indices_task   = loop.run_in_executor(executor, _fetch_indices)
        movers_task    = loop.run_in_executor(executor, _fetch_movers)
        trending_task  = loop.run_in_executor(executor, _fetch_trending_prices)

        indices, movers, trending = await asyncio.gather(indices_task, movers_task, trending_task)

        movers_sorted = sorted(movers, key=lambda x: x["change_pct"], reverse=True)
        top_gainers = movers_sorted[:5]
        top_losers  = list(reversed(movers_sorted))[:5]

        return {
            "indices": indices,
            "top_gainers": top_gainers,
            "top_losers": top_losers,
            "trending": trending,
        }

    except Exception as e:
        logger.error(f"Market overview failed: {e}", exc_info=True)
        return {"indices": {}, "top_gainers": [], "top_losers": [], "trending": [], "error": str(e)}
