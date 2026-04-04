"""
nse_search.py — NSE/BSE Stock Search & Autocomplete
=====================================================
Provides ticker symbol autocomplete using NSE India's public search API
and a curated local list of the top 100 Indian stocks as a reliable fallback.
"""

import logging
import requests
from typing import List

logger = logging.getLogger("MoneyMattersAI.Stock.NSESearch")

# ── Top 100 Indian stocks (local fallback) ─────────────────────────────────
TOP_INDIAN_STOCKS = [
    {"symbol": "RELIANCE",   "name": "Reliance Industries Ltd",         "exchange": "NSE"},
    {"symbol": "TCS",        "name": "Tata Consultancy Services Ltd",   "exchange": "NSE"},
    {"symbol": "HDFCBANK",   "name": "HDFC Bank Ltd",                   "exchange": "NSE"},
    {"symbol": "INFY",       "name": "Infosys Ltd",                     "exchange": "NSE"},
    {"symbol": "HINDUNILVR", "name": "Hindustan Unilever Ltd",          "exchange": "NSE"},
    {"symbol": "ICICIBANK",  "name": "ICICI Bank Ltd",                  "exchange": "NSE"},
    {"symbol": "KOTAKBANK",  "name": "Kotak Mahindra Bank Ltd",         "exchange": "NSE"},
    {"symbol": "LT",         "name": "Larsen & Toubro Ltd",             "exchange": "NSE"},
    {"symbol": "SBIN",       "name": "State Bank of India",             "exchange": "NSE"},
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel Ltd",               "exchange": "NSE"},
    {"symbol": "ITC",        "name": "ITC Ltd",                         "exchange": "NSE"},
    {"symbol": "ASIANPAINT", "name": "Asian Paints Ltd",                "exchange": "NSE"},
    {"symbol": "AXISBANK",   "name": "Axis Bank Ltd",                   "exchange": "NSE"},
    {"symbol": "MARUTI",     "name": "Maruti Suzuki India Ltd",         "exchange": "NSE"},
    {"symbol": "WIPRO",      "name": "Wipro Ltd",                       "exchange": "NSE"},
    {"symbol": "HCLTECH",    "name": "HCL Technologies Ltd",            "exchange": "NSE"},
    {"symbol": "TITAN",      "name": "Titan Company Ltd",               "exchange": "NSE"},
    {"symbol": "BAJFINANCE", "name": "Bajaj Finance Ltd",               "exchange": "NSE"},
    {"symbol": "ULTRACEMCO", "name": "UltraTech Cement Ltd",            "exchange": "NSE"},
    {"symbol": "SUNPHARMA",  "name": "Sun Pharmaceutical Industries",   "exchange": "NSE"},
    {"symbol": "NESTLEIND",  "name": "Nestle India Ltd",                "exchange": "NSE"},
    {"symbol": "ADANIENT",   "name": "Adani Enterprises Ltd",           "exchange": "NSE"},
    {"symbol": "ADANIPORTS", "name": "Adani Ports & SEZ Ltd",           "exchange": "NSE"},
    {"symbol": "TATAMOTOR",  "name": "Tata Motors Ltd",                 "exchange": "NSE"},
    {"symbol": "TATASTEEL",  "name": "Tata Steel Ltd",                  "exchange": "NSE"},
    {"symbol": "NTPC",       "name": "NTPC Ltd",                        "exchange": "NSE"},
    {"symbol": "POWERGRID",  "name": "Power Grid Corporation",          "exchange": "NSE"},
    {"symbol": "ONGC",       "name": "Oil and Natural Gas Corporation", "exchange": "NSE"},
    {"symbol": "COALINDIA",  "name": "Coal India Ltd",                  "exchange": "NSE"},
    {"symbol": "JSWSTEEL",   "name": "JSW Steel Ltd",                   "exchange": "NSE"},
    {"symbol": "TECHM",      "name": "Tech Mahindra Ltd",               "exchange": "NSE"},
    {"symbol": "GRASIM",     "name": "Grasim Industries Ltd",           "exchange": "NSE"},
    {"symbol": "BAJAJFINSV", "name": "Bajaj Finserv Ltd",               "exchange": "NSE"},
    {"symbol": "DRREDDY",    "name": "Dr. Reddy's Laboratories",        "exchange": "NSE"},
    {"symbol": "CIPLA",      "name": "Cipla Ltd",                       "exchange": "NSE"},
    {"symbol": "DIVISLAB",   "name": "Divi's Laboratories Ltd",         "exchange": "NSE"},
    {"symbol": "EICHERMOT",  "name": "Eicher Motors Ltd",               "exchange": "NSE"},
    {"symbol": "BAJAJ-AUTO", "name": "Bajaj Auto Ltd",                  "exchange": "NSE"},
    {"symbol": "HEROMOTOCO", "name": "Hero MotoCorp Ltd",               "exchange": "NSE"},
    {"symbol": "INDUSINDBK", "name": "IndusInd Bank Ltd",               "exchange": "NSE"},
    {"symbol": "M&M",        "name": "Mahindra & Mahindra Ltd",         "exchange": "NSE"},
    {"symbol": "BRITANNIA",  "name": "Britannia Industries Ltd",        "exchange": "NSE"},
    {"symbol": "PIDILITIND", "name": "Pidilite Industries Ltd",         "exchange": "NSE"},
    {"symbol": "SHREECEM",   "name": "Shree Cement Ltd",                "exchange": "NSE"},
    {"symbol": "APOLLOHOSP", "name": "Apollo Hospitals Enterprise",     "exchange": "NSE"},
    {"symbol": "ADANIGREEN", "name": "Adani Green Energy Ltd",          "exchange": "NSE"},
    {"symbol": "ADANITRANS", "name": "Adani Transmission Ltd",          "exchange": "NSE"},
    {"symbol": "BEL",        "name": "Bharat Electronics Ltd",          "exchange": "NSE"},
    {"symbol": "BPCL",       "name": "Bharat Petroleum Corp Ltd",       "exchange": "NSE"},
    {"symbol": "TATACONSUM", "name": "Tata Consumer Products Ltd",      "exchange": "NSE"},
    {"symbol": "HINDALCO",   "name": "Hindalco Industries Ltd",         "exchange": "NSE"},
    {"symbol": "VEDL",       "name": "Vedanta Ltd",                     "exchange": "NSE"},
    {"symbol": "GODREJCP",   "name": "Godrej Consumer Products Ltd",    "exchange": "NSE"},
    {"symbol": "DABUR",      "name": "Dabur India Ltd",                 "exchange": "NSE"},
    {"symbol": "MARICO",     "name": "Marico Ltd",                      "exchange": "NSE"},
    {"symbol": "HAVELLS",    "name": "Havells India Ltd",               "exchange": "NSE"},
    {"symbol": "MUTHOOTFIN", "name": "Muthoot Finance Ltd",             "exchange": "NSE"},
    {"symbol": "CHOLAFIN",   "name": "Cholamandalam Investment & Fin",  "exchange": "NSE"},
    {"symbol": "HDFCLIFE",   "name": "HDFC Life Insurance Co Ltd",      "exchange": "NSE"},
    {"symbol": "SBILIFE",    "name": "SBI Life Insurance Co Ltd",       "exchange": "NSE"},
    {"symbol": "ICICIPRULI", "name": "ICICI Prudential Life Insurance", "exchange": "NSE"},
    {"symbol": "ICICIGI",    "name": "ICICI Lombard General Insurance", "exchange": "NSE"},
    {"symbol": "NAUKRI",     "name": "Info Edge (India) Ltd",           "exchange": "NSE"},
    {"symbol": "PAGEIND",    "name": "Page Industries Ltd",             "exchange": "NSE"},
    {"symbol": "TORNTPHARM", "name": "Torrent Pharmaceuticals Ltd",     "exchange": "NSE"},
    {"symbol": "LUPIN",      "name": "Lupin Ltd",                       "exchange": "NSE"},
    {"symbol": "BIOCON",     "name": "Biocon Ltd",                      "exchange": "NSE"},
    {"symbol": "ALKEM",      "name": "Alkem Laboratories Ltd",          "exchange": "NSE"},
    {"symbol": "AMBUJACEM",  "name": "Ambuja Cements Ltd",              "exchange": "NSE"},
    {"symbol": "ACC",        "name": "ACC Ltd",                         "exchange": "NSE"},
    {"symbol": "DMART",      "name": "Avenue Supermarts Ltd (D-Mart)",  "exchange": "NSE"},
    {"symbol": "ZOMATO",     "name": "Zomato Ltd",                      "exchange": "NSE"},
    {"symbol": "PAYTM",      "name": "One97 Communications (Paytm)",    "exchange": "NSE"},
    {"symbol": "NYKAA",      "name": "FSN E-Commerce Ventures (Nykaa)","exchange": "NSE"},
    {"symbol": "POLICYBZR",  "name": "PB Fintech (PolicyBazaar)",       "exchange": "NSE"},
    {"symbol": "IRCTC",      "name": "IRCTC Ltd",                       "exchange": "NSE"},
    {"symbol": "RAILVIKAS",  "name": "Rail Vikas Nigam Ltd",            "exchange": "NSE"},
    {"symbol": "HAL",        "name": "Hindustan Aeronautics Ltd (HAL)", "exchange": "NSE"},
    {"symbol": "IOC",        "name": "Indian Oil Corporation Ltd",      "exchange": "NSE"},
    {"symbol": "GAIL",       "name": "GAIL (India) Ltd",                "exchange": "NSE"},
    {"symbol": "PEL",        "name": "Piramal Enterprises Ltd",         "exchange": "NSE"},
    {"symbol": "TRENT",      "name": "Trent Ltd (TATA Retail)",         "exchange": "NSE"},
    {"symbol": "PERSISTENT", "name": "Persistent Systems Ltd",          "exchange": "NSE"},
    {"symbol": "MPHASIS",    "name": "Mphasis Ltd",                     "exchange": "NSE"},
    {"symbol": "COFORGE",    "name": "Coforge Ltd",                     "exchange": "NSE"},
    {"symbol": "LTIM",       "name": "LTIMindtree Ltd",                 "exchange": "NSE"},
    {"symbol": "DIXON",      "name": "Dixon Technologies India Ltd",    "exchange": "NSE"},
    {"symbol": "TATAPOWER",  "name": "Tata Power Company Ltd",          "exchange": "NSE"},
    {"symbol": "NHPC",       "name": "NHPC Ltd",                        "exchange": "NSE"},
    {"symbol": "SJVN",       "name": "SJVN Ltd",                        "exchange": "NSE"},
    {"symbol": "RVNL",       "name": "Rail Vikas Nigam Ltd",            "exchange": "NSE"},
    {"symbol": "CANBK",      "name": "Canara Bank",                     "exchange": "NSE"},
    {"symbol": "PNB",        "name": "Punjab National Bank",            "exchange": "NSE"},
    {"symbol": "BANKBARODA", "name": "Bank of Baroda",                  "exchange": "NSE"},
    {"symbol": "FEDERALBNK", "name": "Federal Bank Ltd",                "exchange": "NSE"},
    {"symbol": "IDFCFIRSTB", "name": "IDFC First Bank Ltd",             "exchange": "NSE"},
    {"symbol": "BANDHANBNK", "name": "Bandhan Bank Ltd",                "exchange": "NSE"},
    {"symbol": "RBLBANK",    "name": "RBL Bank Ltd",                    "exchange": "NSE"},
    {"symbol": "YESBANK",    "name": "Yes Bank Ltd",                    "exchange": "NSE"},
]


def search_stocks(query: str, limit: int = 10) -> List[dict]:
    """
    Search for Indian stocks by name or ticker symbol.
    First tries local curated list (instant), then NSE API if needed.

    Args:
        query: Search string (e.g. 'reliance', 'TCS', 'hdfc')
        limit: Max results to return

    Returns:
        List of dicts: {symbol, name, exchange, ticker (with .NS suffix)}
    """
    q = query.strip().lower()
    if not q or len(q) < 1:
        return []

    # ── Local fuzzy search (always works, no network needed) ──────────────
    matches = []
    for stock in TOP_INDIAN_STOCKS:
        sym_match  = stock["symbol"].lower().startswith(q)
        name_match = q in stock["name"].lower()
        if sym_match or name_match:
            matches.append({
                "symbol":   stock["symbol"],
                "name":     stock["name"],
                "exchange": stock["exchange"],
                "ticker":   f"{stock['symbol']}.NS",
            })

    if matches:
        return matches[:limit]

    # ── Yahoo Finance Fallback (Supports NSE + BSE) ───────────────────────
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        
        # Yahoo Finance captures both BSE (.BO) and NSE (.NS)
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount={limit}&newsCount=0"
        resp = requests.get(url, headers=headers, timeout=5)
        
        if resp.status_code == 200:
            data = resp.json()
            quotes = data.get("quotes", [])
            
            # Filter specifically for Indian exchanges to keep it clean
            for idx, q_info in enumerate(quotes):
                exch = q_info.get("exchange", "")
                if exch in ("BSE", "NSI", "NSE"):
                    # Extract raw symbol without exchange suffix for display
                    raw_sym = q_info.get("symbol", "").split(".")[0]
                    matches.append({
                        "symbol": raw_sym,
                        "name": q_info.get("shortname", q_info.get("longname", raw_sym)),
                        "exchange": exch,
                        "ticker": q_info.get("symbol") # already has .NS or .BO
                    })
            
            # deduplicate by ticker just in case
            unique_matches = []
            seen = set()
            for m in matches:
                if m["ticker"] not in seen:
                    unique_matches.append(m)
                    seen.add(m["ticker"])
            
            return unique_matches[:limit]
            
    except Exception as e:
        logger.warning(f"Yahoo Search API failed: {e}")

    return matches[:limit]
