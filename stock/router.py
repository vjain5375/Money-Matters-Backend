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
from fastapi import APIRouter, HTTPException, Query, status

from stock.fundamentals  import get_fundamentals
from stock.technical     import get_technical_data
from stock.nse_search    import search_stocks
from stock.sentiment     import get_sentiment
from stock.predictor     import predict

logger = logging.getLogger("MoneyMattersAI.Stock.Router")

router = APIRouter(prefix="/stock", tags=["Stock Analysis"])


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
async def full_analysis(ticker: str, period: str = Query("1y")):
    """
    Fetches fundamentals + technical + sentiment + prediction in one request.
    Used by the dashboard for initial load.
    """
    symbol = _nse_symbol(ticker)
    logger.info(f"GET /stock/full/{symbol}")

    f_data = get_fundamentals(symbol)
    company_name = f_data.get("company_name")
    t_data = get_technical_data(symbol, period=period)
    s_data = get_sentiment(symbol, company_name=company_name)
    p_data = predict(f_data, t_data, s_data)

    return {
        "ticker":       symbol,
        "company_name": company_name,
        "fundamentals": f_data,
        "technical":    t_data,
        "sentiment":    s_data,
        "prediction":   p_data,
    }
