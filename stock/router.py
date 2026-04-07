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
import time
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

# Simple in-memory TTL cache for heavy endpoints
_cache: dict = {}
_CACHE_TTL = 300  # 5 minutes

def _cache_get(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < _CACHE_TTL:
        return entry["data"]
    return None

def _cache_set(key: str, data):
    _cache[key] = {"ts": time.time(), "data": data}


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
    Returns Nifty/Sensex indices, top gainers/losers & trending stocks.
    Results are cached for 5 minutes to avoid repeated slow fetches.
    """
    import yfinance as yf
    import pandas as pd

    # Return cached result if fresh
    cached = _cache_get("market_overview")
    if cached:
        return {**cached, "cached": True}

    # ── All tickers in one batch download ─────────────────────────────────
    INDEX_MAP = {
        "NIFTY 50":   "^NSEI",
        "SENSEX":     "^BSESN",
        "NIFTY Bank": "^NSEBANK",
        "NIFTY IT":   "^CNXIT",
    }

    POPULAR = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "WIPRO.NS", "BAJAJFINSV.NS", "AXISBANK.NS", "LT.NS",
        "KOTAKBANK.NS", "MARUTI.NS", "TATAMOTORS.NS", "HCLTECH.NS", "ASIANPAINT.NS",
        "SUNPHARMA.NS", "TITAN.NS", "BHARTIARTL.NS", "NTPC.NS", "POWERGRID.NS",
        "ADANIENT.NS", "ULTRACEMCO.NS", "HINDUNILVR.NS", "ITC.NS", "ZOMATO.NS",
        "TATASTEEL.NS", "JSWSTEEL.NS", "COALINDIA.NS", "ONGC.NS",
    ]

    TRENDING_META = [
        {"symbol": "RELIANCE.NS", "name": "Reliance"},
        {"symbol": "TCS.NS",      "name": "TCS"},
        {"symbol": "INFY.NS",     "name": "Infosys"},
        {"symbol": "ZOMATO.NS",   "name": "Zomato"},
        {"symbol": "ADANIENT.NS", "name": "Adani Ent."},
        {"symbol": "TATAMOTORS.NS","name": "Tata Motors"},
        {"symbol": "SBIN.NS",     "name": "SBI"},
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank"},
    ]

    def _fetch_all():
        # All tickers in ONE batch download — minimizes Yahoo API round trips
        index_tickers = list(INDEX_MAP.values())
        all_tickers = index_tickers + POPULAR  # indices + popular stocks

        try:
            raw = yf.download(
                all_tickers, period="5d", interval="1d",
                progress=False, auto_adjust=True, group_by="ticker"
            )
        except Exception as e:
            logger.error(f"Batch download failed: {e}")
            return {}, [], []

        def get_close_change(ticker):
            """Extract (current_price, pct_change) from raw batch data."""
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if ticker not in raw.columns.get_level_values(0):
                        return None, None
                    df = raw[ticker][["Close"]].dropna()
                else:
                    df = raw[["Close"]].dropna()
                if len(df) < 2:
                    return None, None
                prev = float(df["Close"].iloc[-2])
                curr = float(df["Close"].iloc[-1])
                pct = ((curr - prev) / prev * 100) if prev else 0
                return round(curr, 2), round(pct, 2)
            except Exception:
                return None, None

        # Build indices dict
        indices = {}
        for name, ticker in INDEX_MAP.items():
            price, pct = get_close_change(ticker)
            if price:
                curr = price
                prev_price = curr / (1 + pct / 100) if pct != -100 else curr
                indices[name] = {
                    "price": curr,
                    "change": round(curr - prev_price, 2),
                    "change_pct": pct,
                }

        # Build movers list
        movers = []
        for ticker in POPULAR:
            price, pct = get_close_change(ticker)
            if price is not None and pct is not None:
                movers.append({
                    "symbol": ticker,
                    "name": ticker.replace(".NS", ""),
                    "price": price,
                    "change_pct": pct,
                })

        movers_sorted = sorted(movers, key=lambda x: x["change_pct"], reverse=True)
        top_gainers = movers_sorted[:5]
        top_losers  = movers_sorted[-5:][::-1]

        # Build trending list
        trending = []
        for item in TRENDING_META:
            price, pct = get_close_change(item["symbol"])
            trending.append({
                "symbol": item["symbol"],
                "name": item["name"],
                "price": price,
                "change_pct": pct if pct is not None else 0,
            })

        return indices, top_gainers, top_losers, trending

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _fetch_all)

        if len(result) == 4:
            indices, top_gainers, top_losers, trending = result
        else:
            indices, top_gainers, top_losers, trending = {}, [], [], []

        response = {
            "indices": indices,
            "top_gainers": top_gainers,
            "top_losers": top_losers,
            "trending": trending,
        }
        _cache_set("market_overview", response)
        return {**response, "cached": False}

    except Exception as e:
        logger.error(f"Market overview failed: {e}", exc_info=True)
        return {"indices": {}, "top_gainers": [], "top_losers": [], "trending": [], "error": str(e)}
