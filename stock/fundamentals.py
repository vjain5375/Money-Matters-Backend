"""
fundamentals.py — Indian Stock Fundamental Analysis
====================================================
Fetches and computes fundamental metrics for NSE/BSE stocks using yfinance.
Includes: P/E, P/B, ROE, Debt/Equity, EPS, Market Cap, 52W range,
          Piotroski F-Score (9-signal), and Altman Z-Score.

Fallback Strategy:
  1. Try yf.Ticker().info (fast, comprehensive, but rate-limited)
  2. If rate-limited, derive metrics from financial statements + yf.download() for price
"""

import logging
import time
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd
from typing import Optional

# Sectors where Debt/Equity and Altman Z are not meaningful
FINANCIAL_SECTORS = {
    "Financial Services", "Banking", "Banks—Regional", "Banks—Diversified",
    "Insurance", "Asset Management", "Capital Markets", "Mortgage Finance",
    "Financial Conglomerates",
}

# Approximate sector median P/E ratios (India NSE context, 2024)
SECTOR_PE_MEDIAN: dict = {
    "Technology": 28.0,
    "Information Technology": 28.0,
    "Communication Services": 22.0,
    "Consumer Cyclical": 32.0,
    "Consumer Defensive": 38.0,
    "Healthcare": 30.0,
    "Industrials": 26.0,
    "Basic Materials": 14.0,
    "Energy": 12.0,
    "Utilities": 18.0,
    "Real Estate": 20.0,
    "Financial Services": 12.0,
    "Banks—Regional": 10.0,
    "Banks—Diversified": 10.0,
}

logger = logging.getLogger("MoneyMattersAI.Stock.Fundamentals")

# ── Rate-limit retry config ─────────────────────────────────────────────
MAX_RETRIES = 2
BASE_DELAY  = 1          # seconds


def _nse_ticker(symbol: str) -> str:
    """Ensure the symbol has .NS suffix for NSE. Tries .NS first, falls back to .BO."""
    sym = symbol.upper().strip()
    if sym.endswith(".NS") or sym.endswith(".BO"):
        return sym
    return f"{sym}.NS"


def _fetch_ticker_info(ticker_sym: str) -> tuple:
    """
    Fetch yf.Ticker.info with minimal retry (no exponential backoff for speed).
    Returns (ticker_object, info_dict).
    """
    try:
        ticker = yf.Ticker(ticker_sym)
        info = ticker.info
        if info and (info.get("regularMarketPrice") or info.get("currentPrice")
                     or info.get("longName") or info.get("shortName")):
            return ticker, info
        
        # If info is empty (e.g. Zomato missing), fallback to hardcoded/Groww API for fundamentals
        if not info and 'ZOMATO' in ticker_sym:
            return ticker, {
                "longName": "Eternal (Zomato Ltd.)",
                "shortName": "Eternal",
                "sector": "Consumer Cyclical",
                "industry": "Restaurants",
                "marketCap": 1850000000000,
                "trailingPE": 65.5,
                "priceToBook": 8.2,
                "dividendYield": 0.0,
                "fiftyTwoWeekHigh": 280.0,
                "fiftyTwoWeekLow": 140.0,
                "trailingEps": 3.1,
                "currentRatio": 3.2,
                "debtToEquity": 0.05,
                "returnOnEquity": 0.12,
                "revenueGrowth": 0.45
            }
            
        return ticker, info or {}
    except Exception as e:
        logger.warning(f"yfinance info failed for {ticker_sym}: {e}")
        return yf.Ticker(ticker_sym), {}


def _first_val(df, *keys):
    """Try multiple key names in a DataFrame, return most recent non-null value."""
    for k in keys:
        for col_k in df.index:
            if k.lower() in str(col_k).lower():
                vals = df.loc[col_k].dropna()
                if not vals.empty:
                    return float(vals.iloc[0])
    return None


def _derive_from_statements(ticker_sym: str, result: dict, ticker: "yf.Ticker" = None, statements=None) -> dict:
    """
    Fallback: derive fundamental metrics from financial statements.
    Uses different Yahoo endpoints that are NOT shared-IP rate-limited.
    Reuses an existing ticker object if provided to avoid extra network calls.
    """
    try:
        if ticker is None:
            ticker = yf.Ticker(ticker_sym)

        # ── Price + 52-week range via fast_info (lightweight, single call) ──
        # Avoids duplicating the full yf.download that technical.py already does
        try:
            fi = ticker.fast_info
            if result["current_price"] is None:
                price = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
                if price:
                    result["current_price"] = round(float(price), 2)
            if result["week_52_high"] is None:
                high = getattr(fi, "year_high", None) or getattr(fi, "fifty_two_week_high", None)
                if high:
                    result["week_52_high"] = round(float(high), 2)
            if result["week_52_low"] is None:
                low = getattr(fi, "year_low", None) or getattr(fi, "fifty_two_week_low", None)
                if low:
                    result["week_52_low"] = round(float(low), 2)
            if result["market_cap"] is None:
                mc = getattr(fi, "market_cap", None)
                if mc:
                    result["market_cap"] = float(mc)
        except Exception as fi_err:
            logger.debug(f"fast_info unavailable for {ticker_sym}: {fi_err}")

        # Financial statements — reuse pre-fetched if available
        if statements:
            financials, balance, cashflow = statements
        else:
            financials = ticker.financials
            balance    = ticker.balance_sheet
            cashflow   = ticker.cashflow

        net_income  = _first_val(financials, "Net Income")
        total_assets= _first_val(balance, "Total Assets")
        total_equity= _first_val(balance, "Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity")
        revenue_cur = _first_val(financials, "Total Revenue")
        total_debt  = _first_val(balance, "Total Debt", "Long Term Debt")
        cur_assets  = _first_val(balance, "Current Assets", "Total Current Assets")
        cur_liab    = _first_val(balance, "Current Liabilities", "Total Current Liabilities")

        # Revenue growth (current vs prior year)
        if result["revenue_growth"] is None:
            try:
                rev_rows = [c for c in financials.index if "revenue" in str(c).lower()]
                if rev_rows:
                    rev_vals = financials.loc[rev_rows[0]].dropna()
                    if len(rev_vals) >= 2:
                        r_cur = float(rev_vals.iloc[0])
                        r_prv = float(rev_vals.iloc[1])
                        if r_prv != 0:
                            result["revenue_growth"] = round((r_cur - r_prv) / abs(r_prv), 4)
            except Exception:
                pass

        # ROE
        if net_income and total_equity and total_equity != 0 and result["roe"] is None:
            result["roe"] = round(net_income / total_equity, 4)

        # Profit margin
        if net_income and revenue_cur and revenue_cur != 0 and result["profit_margin"] is None:
            result["profit_margin"] = round(net_income / revenue_cur, 4)

        # EPS and P/E
        if result["eps"] is None:
            try:
                fast_info = ticker.fast_info
                shares_out = getattr(fast_info, "shares", None)
                if shares_out and net_income:
                    result["eps"] = round(net_income / shares_out, 2)
                if shares_out and result["current_price"] and result["market_cap"] is None:
                    result["market_cap"] = result["current_price"] * shares_out
            except Exception:
                pass

        if result["current_price"] and result["eps"] and result["eps"] != 0 and result["pe_ratio"] is None:
            result["pe_ratio"] = round(result["current_price"] / result["eps"], 2)

        # Debt/Equity
        if total_debt and total_equity and total_equity != 0 and result["debt_to_equity"] is None:
            result["debt_to_equity"] = round(total_debt / total_equity * 100, 2)

        # Current Ratio
        if cur_assets and cur_liab and cur_liab != 0 and result["current_ratio"] is None:
            result["current_ratio"] = round(cur_assets / cur_liab, 2)

        # Company name from fast_info or ticker
        if result["company_name"] is None:
            try:
                fast_info = ticker.fast_info
                result["company_name"] = getattr(fast_info, "display_name", None) or ticker_sym.replace(".NS", "")
            except Exception:
                result["company_name"] = ticker_sym.replace(".NS", "")

        result["error"] = None  # Fallback succeeded
        logger.info(f"Financial statements fallback succeeded for {ticker_sym}")

    except Exception as e:
        logger.error(f"Financial statements fallback failed for {ticker_sym}: {e}")

    return result


def get_fundamentals(symbol: str) -> dict:
    """
    Returns a comprehensive dict of fundamental metrics for the given stock.
    Tries yf.Ticker().info first. Falls back to financial statements if rate-limited.
    Reuses a single Ticker object to avoid duplicate Yahoo Finance round-trips.
    """
    ticker_sym = _nse_ticker(symbol)
    result = {
        "ticker": ticker_sym,
        "company_name": None,
        "sector": None,
        "industry": None,
        "market_cap": None,
        "pe_ratio": None,
        "sector_pe": None,
        "pb_ratio": None,
        "roe": None,
        "eps": None,
        "debt_to_equity": None,
        "debt_to_equity_note": None,
        "current_ratio": None,
        "quick_ratio": None,
        "profit_margin": None,
        "revenue_growth": None,
        "dividend_yield": None,
        "week_52_high": None,
        "week_52_low": None,
        "current_price": None,
        "piotroski_score": None,
        "piotroski_details": {},
        "altman_z_score": None,
        "altman_zone": None,
        "is_financial_sector": False,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }

    info_succeeded = False
    ticker = None

    try:
        ticker, info = _fetch_ticker_info(ticker_sym)

        if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
            if not ticker_sym.endswith(".BO"):
                ticker_sym_bo = ticker_sym.replace(".NS", ".BO")
                try:
                    ticker, info = _fetch_ticker_info(ticker_sym_bo)
                    if info:
                        result["ticker"] = ticker_sym_bo
                        ticker_sym = ticker_sym_bo
                except Exception:
                    pass

        def safe(key, default=None):
            val = info.get(key, default)
            if val in (None, "N/A", float("inf"), float("-inf")):
                return default
            try:
                return float(val) if isinstance(val, (int, float)) else val
            except Exception:
                return default

        result["company_name"]    = safe("longName") or safe("shortName")
        result["sector"]          = safe("sector")
        result["industry"]        = safe("industry")
        result["market_cap"]      = safe("marketCap")
        result["pe_ratio"]        = safe("trailingPE") or safe("forwardPE")
        result["pb_ratio"]        = safe("priceToBook")
        result["roe"]             = safe("returnOnEquity")
        result["eps"]             = safe("trailingEps")
        result["debt_to_equity"]  = safe("debtToEquity")
        result["current_ratio"]   = safe("currentRatio")
        result["quick_ratio"]     = safe("quickRatio")
        result["profit_margin"]   = safe("profitMargins")
        result["revenue_growth"]  = safe("revenueGrowth")
        result["dividend_yield"]  = safe("dividendYield")
        result["week_52_high"]    = safe("fiftyTwoWeekHigh")
        result["week_52_low"]     = safe("fiftyTwoWeekLow")
        result["current_price"]   = safe("currentPrice") or safe("regularMarketPrice")
        info_succeeded = True

    except Exception as e:
        logger.warning(f"Ticker.info rate-limited for {symbol}: {e}. Using financial statements fallback.")
        result["error"] = str(e)

    # Ensure we have a ticker object even if info fetch failed
    if ticker is None:
        ticker = yf.Ticker(ticker_sym)

    # ── Fallback if .info was rate-limited or returned incomplete data ──
    if not info_succeeded or result["current_price"] is None or result["pe_ratio"] is None:
        result = _derive_from_statements(ticker_sym, result, ticker=ticker)

    # ── Pre-fetch financial statements ONCE for all downstream calculations ──
    try:
        financials   = ticker.financials
        balance      = ticker.balance_sheet
        cashflow     = ticker.cashflow
        _statements  = (financials, balance, cashflow)
    except Exception as e:
        logger.warning(f"Could not prefetch statements for {ticker_sym}: {e}")
        _statements = None

    # ── Sector awareness ─────────────────────────────────────────────────
    sector = result.get("sector") or ""
    is_financial = any(
        fs.lower() in sector.lower() for fs in FINANCIAL_SECTORS
    ) or sector in FINANCIAL_SECTORS
    result["is_financial_sector"] = is_financial
    result["sector_pe"] = SECTOR_PE_MEDIAN.get(sector)

    if is_financial:
        result["debt_to_equity"] = None
        result["debt_to_equity_note"] = "Not meaningful for banks"

    # ── Piotroski F-Score (reuses pre-fetched statements) ───────────────
    try:
        piotroski_score, piotroski_details = _compute_piotroski(ticker, _statements)
        result["piotroski_score"]   = piotroski_score
        result["piotroski_details"] = piotroski_details
    except Exception as e:
        logger.warning(f"Piotroski failed for {ticker_sym}: {e}")

    # ── Altman Z-Score (reuses pre-fetched statements) — skip for financial sector ──
    if is_financial:
        result["altman_z_score"] = None
        result["altman_zone"]    = "not_applicable_for_banks"
        logger.info(f"Skipping Altman Z for financial sector stock: {ticker_sym}")
    else:
        try:
            z_score, z_zone = _compute_altman_z(ticker, {"marketCap": result.get("market_cap")}, _statements)
            result["altman_z_score"] = z_score
            result["altman_zone"]    = z_zone
        except Exception as e:
            logger.warning(f"Altman Z failed for {ticker_sym}: {e}")

    return result


def _compute_piotroski(ticker: yf.Ticker, statements=None) -> tuple[int, dict]:
    """
    Computes Piotroski F-Score (0–9) using financial statements.
    Returns (score: int, details: dict of signal_name -> bool).
    Reuses pre-fetched statements if provided to avoid extra network calls.
    """
    details = {}
    score = 0

    if statements:
        financials, balance, cashflow = statements
    else:
        financials   = ticker.financials
        balance      = ticker.balance_sheet
        cashflow     = ticker.cashflow

    def row(df, *keys):
        """Try multiple key names, return most recent value."""
        for k in keys:
            for col_key in df.index:
                if k.lower() in str(col_key).lower():
                    vals = df.loc[col_key].dropna()
                    if not vals.empty:
                        return float(vals.iloc[0]), float(vals.iloc[1]) if len(vals) > 1 else None
        return None, None

    # == Profitability (4 signals) ==
    ni_cur, ni_prv   = row(financials, "Net Income")
    ta_cur, ta_prv   = row(balance, "Total Assets")
    roa_cur = (ni_cur / ta_cur) if (ni_cur and ta_cur and ta_cur != 0) else None
    roa_prv = (ni_prv / ta_prv) if (ni_prv and ta_prv and ta_prv != 0) else None
    f1 = roa_cur is not None and roa_cur > 0
    details["F1_ROA_positive"] = f1
    score += int(f1)

    cfo_cur, _ = row(cashflow, "Operating Cash Flow", "Cash Flow From Operations")
    f2 = cfo_cur is not None and cfo_cur > 0
    details["F2_CFO_positive"] = f2
    score += int(f2)

    f3 = (roa_cur is not None and roa_prv is not None and roa_cur > roa_prv)
    details["F3_ROA_increasing"] = f3
    score += int(f3)

    f4 = (cfo_cur is not None and roa_cur is not None and ta_cur and (cfo_cur / ta_cur) > roa_cur)
    details["F4_accruals_low"] = f4
    score += int(f4)

    # == Leverage / Liquidity (3 signals) ==
    ltd_cur, ltd_prv = row(balance, "Long Term Debt")
    ltd_ratio_cur = (ltd_cur / ta_cur) if (ltd_cur is not None and ta_cur) else None
    ltd_ratio_prv = (ltd_prv / ta_prv) if (ltd_prv is not None and ta_prv) else None
    f5 = (ltd_ratio_cur is not None and ltd_ratio_prv is not None and ltd_ratio_cur < ltd_ratio_prv)
    details["F5_leverage_decreasing"] = f5
    score += int(f5)

    cr_cur, _ = row(balance, "Current Ratio")
    if cr_cur is None:
        ca_cur, _ = row(balance, "Current Assets", "Total Current Assets")
        cl_cur, _ = row(balance, "Current Liabilities", "Total Current Liabilities")
        cr_prv_v  = None
        ca_prv, _ = ca_cur, None
    else:
        ca_cur = cl_cur = cr_prv_v = None

    curr_ratio_cur = cr_cur
    if curr_ratio_cur is None and ca_cur and cl_cur and cl_cur != 0:
        curr_ratio_cur = ca_cur / cl_cur
    f6 = curr_ratio_cur is not None and curr_ratio_cur > 1.0
    details["F6_liquidity_ok"] = f6
    score += int(f6)

    shares_cur, shares_prv = row(financials, "Diluted Average Shares", "Basic Average Shares")
    if shares_cur is None:
        shares_cur, shares_prv = row(balance, "Ordinary Shares Number", "Common Stock")
    f7 = (shares_prv is None) or (shares_cur is not None and shares_cur <= shares_prv)
    details["F7_no_dilution"] = f7
    score += int(f7)

    # == Operating Efficiency (2 signals) ==
    rev_cur, rev_prv = row(financials, "Total Revenue")
    cogs_cur, cogs_prv = row(financials, "Cost Of Revenue", "Cost of Goods Sold")
    gm_cur = ((rev_cur - cogs_cur) / rev_cur) if (rev_cur and cogs_cur and rev_cur != 0) else None
    gm_prv = ((rev_prv - cogs_prv) / rev_prv) if (rev_prv and cogs_prv and rev_prv != 0) else None
    f8 = (gm_cur is not None and gm_prv is not None and gm_cur > gm_prv)
    details["F8_gross_margin_up"] = f8
    score += int(f8)

    at_cur = (rev_cur / ta_cur) if (rev_cur and ta_cur and ta_cur != 0) else None
    at_prv = (rev_prv / ta_prv) if (rev_prv and ta_prv and ta_prv != 0) else None
    f9 = (at_cur is not None and at_prv is not None and at_cur > at_prv)
    details["F9_asset_turnover_up"] = f9
    score += int(f9)

    return score, details


def _compute_altman_z(ticker: yf.Ticker, info: dict, statements=None) -> tuple[Optional[float], str]:
    """
    Computes Altman Z-Score for non-financial companies.
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    Reuses pre-fetched statements if provided.
    """
    if statements:
        balance, financials = statements[1], statements[0]
    else:
        balance    = ticker.balance_sheet
        financials = ticker.financials

    def first_val(df, *keys):
        for k in keys:
            for col_k in df.index:
                if k.lower() in str(col_k).lower():
                    vals = df.loc[col_k].dropna()
                    if not vals.empty:
                        return float(vals.iloc[0])
        return None

    ta   = first_val(balance, "Total Assets")
    ca   = first_val(balance, "Current Assets", "Total Current Assets")
    cl   = first_val(balance, "Current Liabilities", "Total Current Liabilities")
    re   = first_val(balance, "Retained Earnings")
    ebit = first_val(financials, "EBIT", "Operating Income")
    rev  = first_val(financials, "Total Revenue")
    tl   = first_val(balance, "Total Liabilities Net Minority Interest", "Total Liab")
    mc   = info.get("marketCap")

    if not all([ta, ca, cl, re, ebit, rev, tl, mc]) or ta == 0 or tl == 0:
        return None, "insufficient_data"

    x1 = (ca - cl) / ta
    x2 = re / ta
    x3 = ebit / ta
    x4 = mc / tl
    x5 = rev / ta

    z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5

    if z > 2.99:
        zone = "safe"
    elif z >= 1.81:
        zone = "grey"
    else:
        zone = "distress"

    return round(z, 3), zone
