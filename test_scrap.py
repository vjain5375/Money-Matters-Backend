import requests
import re
from bs4 import BeautifulSoup

url = 'https://www.google.com/finance/quote/ZOMATO:NSE'
resp = requests.get(url, timeout=3)
match = re.search(r'data-last-price="([0-9.]+)"', resp.text)
if match:
    print("Price:", match.group(1))

# Also try to extract the change percentage
soup = BeautifulSoup(resp.text, 'html.parser')
# Google finance change is usually next to price
print("Found price from regex:", match.group(1) if match else "No")
