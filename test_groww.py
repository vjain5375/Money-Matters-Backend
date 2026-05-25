import requests
import concurrent.futures

POPULAR = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "WIPRO", "BAJAJFINSV", "AXISBANK", "LT",
    "KOTAKBANK", "MARUTI", "TATAMOTORS", "HCLTECH", "ASIANPAINT",
    "SUNPHARMA", "TITAN", "BHARTIARTL", "NTPC", "POWERGRID",
    "ADANIENT", "ULTRACEMCO", "HINDUNILVR", "ITC", "ZOMATO",
    "TATASTEEL", "JSWSTEEL", "COALINDIA", "ONGC",
]

def fetch_groww(symbol):
    try:
        url = f'https://groww.in/v1/api/stocks_data/v1/tr_live_prices/exchange/NSE/segment/CASH/{symbol}/latest'
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=2)
        d = r.json()
        return symbol, d.get('ltp'), d.get('dayChangePerc')
    except:
        return symbol, None, None

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(fetch_groww, POPULAR))

for sym, ltp, pct in results:
    if ltp is None:
        print(f"FAILED: {sym}")
