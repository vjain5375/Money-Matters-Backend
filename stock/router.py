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
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse

from stock.fundamentals  import get_fundamentals
from stock.technical     import get_technical_data
from stock.nse_search    import search_stocks
from stock.sentiment     import get_sentiment
from stock.predictor     import predict

logger = logging.getLogger("MoneyMattersAI.Stock.Router")

router = APIRouter(prefix="/stock", tags=["Stock Analysis"])

# Create a shared executor for parallel tasks
executor = ThreadPoolExecutor(max_workers=10)

# ── In-memory TTL cache ───────────────────────────────────────────────────
_cache: dict = {}
_CACHE_TTL_MARKET   = 600   # 10 min — market overview (indices/gainers/losers)
_CACHE_TTL_FULL     = 180   # 3  min — full stock analysis per ticker
_CACHE_TTL_DEFAULT  = 300   # 5  min — fallback

def _cache_get(key: str, ttl: int = None):
    entry = _cache.get(key)
    effective_ttl = ttl if ttl is not None else _CACHE_TTL_DEFAULT
    if entry and (time.time() - entry["ts"]) < effective_ttl:
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
    
    # We return sentiment data as well to allow the frontend to lazy-load both
    # in a single network call, avoiding duplicate LLM costs.
    prediction["_sentiment_data"] = s_data
    
    return prediction


# ── GET /stock/full/{ticker} ──────────────────────────────────────────────
@router.get("/full/{ticker}", summary="All Data in One Call")
async def full_analysis(ticker: str, period: str = Query("1y"), company_name: str = None):
    """
    Combines fundamentals, technicals, and sentiment into one report.
    Fetches data in parallel to reduce latency. Results cached for 5 min.
    """
    symbol = _nse_symbol(ticker)

    # Check cache first — full analysis is expensive
    cache_key = f"full_{symbol}_{period}"
    cached = _cache_get(cache_key, _CACHE_TTL_FULL)
    if cached:
        return {**cached, "cached": True}

    loop = asyncio.get_event_loop()
    
    try:
        # Run ONLY fundamentals and technicals in parallel for lightning-fast UI load!
        # We skip sentiment and prediction here to reduce latency from 4s -> 0.8s.
        f_fut = loop.run_in_executor(executor, get_fundamentals, symbol)
        t_fut = loop.run_in_executor(executor, get_technical_data, symbol, period)

        f_data, t_data = await asyncio.gather(f_fut, t_fut)

        # Empty structures for the frontend to lazy-load later
        s_data = {}
        p_data = {}

        # Fallback — patch price & name from technicals if fundamentals failed
        if f_data.get("error") or f_data.get("current_price") is None:
            if t_data.get("candles"):
                last_close = t_data["candles"][-1].get("close")
                if last_close is not None:
                    f_data["current_price"] = last_close
            if not f_data.get("company_name"):
                f_data["company_name"] = symbol.replace(".NS", "").replace(".BO", "")
        
        response = {
            "symbol": symbol,
            "company_name": f_data.get("company_name") or symbol,
            "ticker": symbol,
            "fundamentals": f_data,
            "technicals": t_data,
            "sentiment": s_data,
            "prediction": p_data
        }
        _cache_set(cache_key, response)
        # Return with Cache-Control so Cloudflare/CDN/browser can cache too
        return JSONResponse(
            content={**response, "cached": False},
            headers={"Cache-Control": f"public, max-age={_CACHE_TTL_FULL}, stale-while-revalidate=60"}
        )

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
    Results are cached for 10 minutes to avoid repeated slow fetches.
    """
    import yfinance as yf
    import pandas as pd

    # Return cached result if fresh (10 min TTL)
    cached = _cache_get("market_overview_v4", _CACHE_TTL_MARKET)
    if cached:
        return JSONResponse(
            content={**cached, "cached": True},
            headers={"Cache-Control": f"public, max-age={_CACHE_TTL_MARKET}, stale-while-revalidate=60"}
        )


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
        "KOTAKBANK.NS", "MARUTI.NS", "M&M.NS", "HCLTECH.NS", "ASIANPAINT.NS",
        "SUNPHARMA.NS", "TITAN.NS", "BHARTIARTL.NS", "NTPC.NS", "POWERGRID.NS",
        "ADANIENT.NS", "ULTRACEMCO.NS", "HINDUNILVR.NS", "ITC.NS", "ETERNAL.NS",
        "TATASTEEL.NS", "JSWSTEEL.NS", "COALINDIA.NS", "ONGC.NS",
    ]

    TRENDING_META = [
        {"symbol": "RELIANCE.NS", "name": "Reliance"},
        {"symbol": "TCS.NS",      "name": "TCS"},
        {"symbol": "INFY.NS",     "name": "Infosys"},
        {"symbol": "ETERNAL.NS",  "name": "Eternal"},
        {"symbol": "ADANIENT.NS", "name": "Adani Ent."},
        {"symbol": "M&M.NS",      "name": "M&M"},
        {"symbol": "SBIN.NS",     "name": "SBI"},
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank"},
    ]

    COMMODITIES_MAP = {
        "Gold (10g)": "GC=F",
        "Silver (1kg)": "SI=F",
        "USD / INR": "INR=X"
    }

    def _fetch_all():
        import requests
        import concurrent.futures

        index_tickers = list(INDEX_MAP.values())
        commodity_tickers = list(COMMODITIES_MAP.values())
        hf_tickers = index_tickers + commodity_tickers

        try:
            raw = yf.download(hf_tickers, period="8d", interval="1d", progress=False, auto_adjust=True, group_by="ticker")
        except Exception as e:
            logger.error(f"Batch download failed: {e}")
            raw = None

        def get_close_change_yf(ticker):
            try:
                df = None
                if raw is not None:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if ticker in raw.columns.get_level_values(0):
                            df = raw[ticker][["Close"]].dropna()
                    else:
                        df = raw[["Close"]].dropna()
                
                if df is None or len(df) < 2:
                    return None, None, []
                    
                prev = float(df["Close"].iloc[-2])
                curr = float(df["Close"].iloc[-1])
                pct = ((curr - prev) / prev * 100) if prev else 0
                sparkline = [round(float(v), 2) for v in df["Close"].iloc[-7:].tolist()]
                return round(curr, 2), round(pct, 2), sparkline
            except Exception:
                return None, None, []

        indices = {}
        for name, ticker in INDEX_MAP.items():
            price, pct, sparkline = get_close_change_yf(ticker)
            if price:
                curr = price
                prev_price = curr / (1 + pct / 100) if pct != -100 else curr
                indices[name] = {
                    "price": curr,
                    "change": round(curr - prev_price, 2),
                    "change_pct": pct,
                    "sparkline_7d": sparkline,
                }

        commodities = {}
        for name, ticker in COMMODITIES_MAP.items():
            price, pct, sparkline = get_close_change_yf(ticker)
            if price:
                curr = price
                prev_price = curr / (1 + pct / 100) if pct != -100 else curr
                commodities[name] = {
                    "price": curr,
                    "change": round(curr - prev_price, 2) if pct is not None else 0,
                    "change_pct": pct if pct is not None else 0,
                    "sparkline_7d": sparkline,
                }

        # Override with pure Indian IBJA Rates!
        try:
            import bs4
            res = requests.get('https://ibjarates.com/', headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            soup = bs4.BeautifulSoup(res.text, 'html.parser')
            tds = soup.find_all('td')
            rows = [t.text.strip() for t in tds if t.text.strip().isdigit() and len(t.text.strip()) > 4]
            if len(rows) >= 6:
                gold_10g = float(rows[0])
                silver_1kg = float(rows[5])
                
                commodities["Gold (10g)"] = {
                    "price": gold_10g,
                    "change": commodities.get("Gold (10g)", {}).get("change", 0),
                    "change_pct": commodities.get("Gold (10g)", {}).get("change_pct", 0),
                    "sparkline_7d": commodities.get("Gold (10g)", {}).get("sparkline_7d", [])
                }
                
                commodities["Silver (1kg)"] = {
                    "price": silver_1kg,
                    "change": commodities.get("Silver (1kg)", {}).get("change", 0),
                    "change_pct": commodities.get("Silver (1kg)", {}).get("change_pct", 0),
                    "sparkline_7d": commodities.get("Silver (1kg)", {}).get("sparkline_7d", [])
                }
        except Exception as e:
            logger.error(f"IBJA fetch failed: {e}")
            inr = commodities.get("USD / INR", {}).get("price", 83.5)
            if "Gold (10g)" in commodities and commodities["Gold (10g)"]["price"] < 5000:
                commodities["Gold (10g)"]["price"] = (commodities["Gold (10g)"]["price"] * inr / 3.11034768) * 1.185
            if "Silver (1kg)" in commodities and commodities["Silver (1kg)"]["price"] < 100:
                commodities["Silver (1kg)"]["price"] = (commodities["Silver (1kg)"]["price"] * inr * 32.1507) * 1.185

        # Fetch Stocks concurrently from Groww
        def get_groww_price(ticker):
            symbol = ticker.replace('.NS', '').replace('.BO', '')
            url = f"https://groww.in/v1/api/stocks_data/v1/tr_live_prices/exchange/NSE/segment/CASH/{symbol}/latest"
            try:
                resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    if "ltp" in data and data["ltp"] is not None:
                        curr = float(data["ltp"])
                        pct = float(data.get("dayChangePerc", 0.0))
                        return ticker, round(curr, 2), round(pct, 2)
            except Exception:
                pass
            return ticker, None, None

        groww_results = {}
        unique_stocks = list(set(POPULAR + [t["symbol"] for t in TRENDING_META]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as exe:
            futures = [exe.submit(get_groww_price, t) for t in unique_stocks]
            for f in concurrent.futures.as_completed(futures):
                ticker, price, pct = f.result()
                groww_results[ticker] = (price, pct)

        movers = []
        for ticker in POPULAR:
            price, pct = groww_results.get(ticker, (None, None))
            if price is not None and pct is not None:
                movers.append({
                    "symbol": ticker,
                    "name": ticker.replace(".NS", ""),
                    "price": price,
                    "change_pct": pct,
                    "sparkline_7d": [],
                })

        movers_sorted = sorted(movers, key=lambda x: x["change_pct"], reverse=True)
        top_gainers = movers_sorted[:5]
        top_losers  = movers_sorted[-5:][::-1]

        trending = []
        for item in TRENDING_META:
            price, pct = groww_results.get(item["symbol"], (None, None))
            trending.append({
                "symbol": item["symbol"],
                "name": item["name"],
                "price": price,
                "change_pct": pct if pct is not None else 0,
                "sparkline_7d": [],
            })

        return indices, top_gainers, top_losers, trending, commodities

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _fetch_all)

        if len(result) == 5:
            indices, top_gainers, top_losers, trending, commodities = result
        else:
            indices, top_gainers, top_losers, trending, commodities = {}, [], [], [], {}

        response = {
            "indices": indices,
            "commodities": commodities,
            "top_gainers": top_gainers,
            "top_losers": top_losers,
            "trending": trending,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        _cache_set("market_overview_v4", response)
        return JSONResponse(
            content={**response, "cached": False},
            headers={"Cache-Control": f"public, max-age={_CACHE_TTL_MARKET}, stale-while-revalidate=120"}
        )

    except Exception as e:
        logger.error(f"Market overview failed: {e}", exc_info=True)
        return {"indices": {}, "commodities": {}, "top_gainers": [], "top_losers": [], "trending": [], "error": str(e)}

# ── GET /stock/chart/{ticker} ──────────────────────────────────────────────
@router.get("/chart/{ticker}", summary="Lightweight Chart Data")
async def chart_data(ticker: str, period: str = Query("1d")):
    symbol = _nse_symbol(ticker)
    cache_key = f"chart_{symbol}_{period}"
    cached = _cache_get(cache_key, 120)  # 2 min cache for charts
    if cached:
        return JSONResponse(content={**cached, "cached": True})
        
    data = get_technical_data(symbol, period=period)
    if data.get("error"):
        raise HTTPException(status_code=404, detail=data["error"])
        
    response = {"ticker": symbol, "period": period, "technicals": data}
    _cache_set(cache_key, response)
    return JSONResponse(content={**response, "cached": False})
