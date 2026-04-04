"""
fundamentals.py — Indian Stock Fundamental Analysis
====================================================
Fetches and computes fundamental metrics for NSE/BSE stocks using yfinance.
Includes: P/E, P/B, ROE, Debt/Equity, EPS, Market Cap, 52W range,
          Piotroski F-Score (9-signal), and Altman Z-Score.
"""

import logging
from typing import Optional
import yfinance as yf
import pandas as pd

logger = logging.getLogger("MoneyMattersAI.Stock.Fundamentals")


def _nse_ticker(symbol: str) -> str:
    """Ensure the symbol has .NS suffix for NSE. Tries .NS first, falls back to .BO."""
    sym = symbol.upper().strip()
    if sym.endswith(".NS") or sym.endswith(".BO"):
        return sym
    return f"{sym}.NS"


def get_fundamentals(symbol: str) -> dict:
    """
    Returns a comprehensive dict of fundamental metrics for the given stock.

    Args:
        symbol: NSE/BSE ticker, e.g. 'RELIANCE', 'TCS.NS', 'HDFCBANK.BO'

    Returns:
        dict with keys: ticker, company_name, sector, industry, market_cap,
        pe_ratio, pb_ratio, roe, eps, debt_to_equity, current_ratio, quick_ratio,
        profit_margin, revenue_growth, dividend_yield, week_52_high, week_52_low,
        current_price, piotroski_score, piotroski_details, altman_z_score,
        altman_zone, error (None if success)
    """
    ticker_sym = _nse_ticker(symbol)
    result = {
        "ticker": ticker_sym,
        "company_name": None,
        "sector": None,
        "industry": None,
        "market_cap": None,
        "pe_ratio": None,
        "pb_ratio": None,
        "roe": None,
        "eps": None,
        "debt_to_equity": None,
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
        "error": None,
    }

    try:
        ticker = yf.Ticker(ticker_sym)
        info = ticker.info

        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            # Try BSE fallback
            if not ticker_sym.endswith(".BO"):
                ticker_sym_bo = ticker_sym.replace(".NS", ".BO")
                ticker = yf.Ticker(ticker_sym_bo)
                info = ticker.info
                if info:
                    result["ticker"] = ticker_sym_bo

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

        # --- Piotroski F-Score (9 binary signals) ---
        try:
            piotroski_score, piotroski_details = _compute_piotroski(ticker)
            result["piotroski_score"]   = piotroski_score
            result["piotroski_details"] = piotroski_details
        except Exception as e:
            logger.warning(f"Piotroski failed for {ticker_sym}: {e}")
            result["piotroski_score"]   = None
            result["piotroski_details"] = {}

        # --- Altman Z-Score ---
        try:
            z_score, z_zone = _compute_altman_z(ticker, info)
            result["altman_z_score"] = z_score
            result["altman_zone"]    = z_zone
        except Exception as e:
            logger.warning(f"Altman Z failed for {ticker_sym}: {e}")
            result["altman_z_score"] = None
            result["altman_zone"]    = None

    except Exception as e:
        logger.error(f"Fundamentals error for {symbol}: {e}")
        result["error"] = str(e)

    return result


def _compute_piotroski(ticker: yf.Ticker) -> tuple[int, dict]:
    """
    Computes Piotroski F-Score (0–9) using financial statements.
    Returns (score: int, details: dict of signal_name -> bool).
    """
    details = {}
    score = 0

    financials   = ticker.financials      # income statement (annual)
    balance      = ticker.balance_sheet   # balance sheet (annual)
    cashflow     = ticker.cashflow        # cash flow (annual)

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
    # F1: ROA > 0 (Net income / Total assets > 0)
    ni_cur, ni_prv   = row(financials, "Net Income")
    ta_cur, ta_prv   = row(balance, "Total Assets")
    roa_cur = (ni_cur / ta_cur) if (ni_cur and ta_cur and ta_cur != 0) else None
    roa_prv = (ni_prv / ta_prv) if (ni_prv and ta_prv and ta_prv != 0) else None
    f1 = roa_cur is not None and roa_cur > 0
    details["F1_ROA_positive"] = f1
    score += int(f1)

    # F2: Operating Cash Flow > 0
    cfo_cur, _ = row(cashflow, "Operating Cash Flow", "Cash Flow From Operations")
    f2 = cfo_cur is not None and cfo_cur > 0
    details["F2_CFO_positive"] = f2
    score += int(f2)

    # F3: Increasing ROA (year-over-year)
    f3 = (roa_cur is not None and roa_prv is not None and roa_cur > roa_prv)
    details["F3_ROA_increasing"] = f3
    score += int(f3)

    # F4: Accruals — CFO > ROA (cash earnings quality)
    f4 = (cfo_cur is not None and roa_cur is not None and ta_cur and (cfo_cur / ta_cur) > roa_cur)
    details["F4_accruals_low"] = f4
    score += int(f4)

    # == Leverage / Liquidity (3 signals) ==
    # F5: Decreasing long-term debt ratio
    ltd_cur, ltd_prv = row(balance, "Long Term Debt")
    ltd_ratio_cur = (ltd_cur / ta_cur) if (ltd_cur is not None and ta_cur) else None
    ltd_ratio_prv = (ltd_prv / ta_prv) if (ltd_prv is not None and ta_prv) else None
    f5 = (ltd_ratio_cur is not None and ltd_ratio_prv is not None and ltd_ratio_cur < ltd_ratio_prv)
    details["F5_leverage_decreasing"] = f5
    score += int(f5)

    # F6: Increasing current ratio
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
    f6 = curr_ratio_cur is not None and curr_ratio_cur > 1.0  # simplified signal
    details["F6_liquidity_ok"] = f6
    score += int(f6)

    # F7: No new share issuance (shares unchanged or decreased)
    shares_cur, shares_prv = row(financials, "Diluted Average Shares", "Basic Average Shares")
    if shares_cur is None:
        shares_cur, shares_prv = row(balance, "Ordinary Shares Number", "Common Stock")
    f7 = (shares_prv is None) or (shares_cur is not None and shares_cur <= shares_prv)
    details["F7_no_dilution"] = f7
    score += int(f7)

    # == Operating Efficiency (2 signals) ==
    # F8: Improving gross margin
    rev_cur, rev_prv = row(financials, "Total Revenue")
    cogs_cur, cogs_prv = row(financials, "Cost Of Revenue", "Cost of Goods Sold")
    gm_cur = ((rev_cur - cogs_cur) / rev_cur) if (rev_cur and cogs_cur and rev_cur != 0) else None
    gm_prv = ((rev_prv - cogs_prv) / rev_prv) if (rev_prv and cogs_prv and rev_prv != 0) else None
    f8 = (gm_cur is not None and gm_prv is not None and gm_cur > gm_prv)
    details["F8_gross_margin_up"] = f8
    score += int(f8)

    # F9: Improving asset turnover
    at_cur = (rev_cur / ta_cur) if (rev_cur and ta_cur and ta_cur != 0) else None
    at_prv = (rev_prv / ta_prv) if (rev_prv and ta_prv and ta_prv != 0) else None
    f9 = (at_cur is not None and at_prv is not None and at_cur > at_prv)
    details["F9_asset_turnover_up"] = f9
    score += int(f9)

    return score, details


def _compute_altman_z(ticker: yf.Ticker, info: dict) -> tuple[Optional[float], str]:
    """
    Computes Altman Z-Score for non-financial companies.
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    Zones:
        Z > 2.99  → Safe (green)
        1.81–2.99 → Grey zone
        Z < 1.81  → Distress (red)
    """
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

    x1 = (ca - cl) / ta          # Working capital / Total assets
    x2 = re / ta                  # Retained earnings / Total assets
    x3 = ebit / ta                # EBIT / Total assets
    x4 = mc / tl                  # Market cap / Total liabilities
    x5 = rev / ta                 # Revenue / Total assets

    z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5

    if z > 2.99:
        zone = "safe"
    elif z >= 1.81:
        zone = "grey"
    else:
        zone = "distress"

    return round(z, 3), zone
