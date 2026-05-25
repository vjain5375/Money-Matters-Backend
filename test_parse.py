import re

with open('gf.html', encoding='utf-8') as f:
    html = f.read()

# Look for data-last-price
m1 = re.search(r'data-last-price="([^"]+)"', html)
print("data-last-price:", m1.group(1) if m1 else "Not found")

# Google Finance class for price is often YMlKec fxKbKc
m2 = re.search(r'class="[^"]*YMlKec[^"]*"[^>]*>₹?([0-9,]+\.[0-9]{2})', html)
print("YMlKec class price:", m2.group(1) if m2 else "Not found")

# Look for the div with data-source
m3 = re.search(r'data-source="ZOMATO"[^>]*>([^<]+)', html)
print("data-source ZOMATO text:", m3.group(1) if m3 else "Not found")
