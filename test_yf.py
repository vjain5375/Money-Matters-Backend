import yfinance as yf
import pandas as pd

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

all_tickers = list(INDEX_MAP.values()) + POPULAR

print("Downloading...")
raw = yf.download(
    all_tickers, period="8d", interval="1d",
    progress=False, auto_adjust=True, group_by="ticker"
)
print("Columns:", raw.columns)
print("Is MultiIndex:", isinstance(raw.columns, pd.MultiIndex))

for t in ["ZOMATO.NS", "TATAMOTORS.NS", "RELIANCE.NS"]:
    print(f"\n--- {t} ---")
    if isinstance(raw.columns, pd.MultiIndex):
        if t in raw.columns.get_level_values(0):
            df = raw[t]
            print(df.tail(2))
        else:
            print("Not in columns level 0")
    else:
        print("Not MultiIndex")
