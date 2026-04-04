"""
sentiment.py — Indian Stock News Sentiment Analysis
====================================================
Fetches news from Indian financial RSS feeds (Economic Times, Moneycontrol,
LiveMint, Business Standard) and scores sentiment using Groq (LLaMA 3).

Requires: pip install groq feedparser
Env var:  GROQ_API_KEY=your_key_here   (free at console.groq.com)

Fallback: if no API key, uses keyword heuristics (no external calls).
"""

import os
import logging
import json
import re
from typing import List
import feedparser
import requests

logger = logging.getLogger("MoneyMattersAI.Stock.Sentiment")

# ── Indian Financial News RSS Feeds ───────────────────────────────────────
RSS_FEEDS = [
    {
        "name": "Economic Times Markets",
        "url": "https://economictimes.indiatimes.com/markets/stocks/rss.cms",
    },
    {
        "name": "Moneycontrol",
        "url": "https://www.moneycontrol.com/rss/buzzingstocks.xml",
    },
    {
        "name": "LiveMint Markets",
        "url": "https://www.livemint.com/rss/markets",
    },
    {
        "name": "Business Standard",
        "url": "https://www.business-standard.com/rss/markets-106.rss",
    },
]

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama3-8b-8192"   # free, fast, smart


def get_sentiment(symbol: str, company_name: str = None) -> dict:
    """
    Fetches relevant news for an Indian stock and scores sentiment via Groq.

    Args:
        symbol:       NSE ticker (e.g. 'RELIANCE.NS')
        company_name: Full name to improve relevance matching (e.g. 'Reliance Industries')

    Returns:
        dict with: ticker, articles (list), overall_score (-1 to +1),
                   overall_label, articles_found, error
    """
    result = {
        "ticker":         symbol,
        "company_name":   company_name,
        "articles":       [],
        "overall_score":  None,
        "overall_label":  None,
        "articles_found": 0,
        "error":          None,
    }

    bare_symbol = symbol.replace(".NS", "").replace(".BO", "").upper()
    search_terms = [bare_symbol.lower()]
    if company_name:
        words = company_name.lower().replace(" ltd", "").replace(" limited", "").strip().split()
        if words:
            search_terms.append(words[0])
            if len(words) >= 2:
                search_terms.append(" ".join(words[:2]))

    # ── Collect articles from RSS feeds ───────────────────────────────────
    raw_articles = []
    for feed_info in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_info["url"])
            for entry in feed.entries[:30]:
                title    = getattr(entry, "title", "") or ""
                summary  = getattr(entry, "summary", "") or ""
                link     = getattr(entry, "link", "") or ""
                pub_date = getattr(entry, "published", "") or ""

                combined = (title + " " + summary).lower()
                if any(term in combined for term in search_terms):
                    raw_articles.append({
                        "title":   title.strip(),
                        "summary": re.sub(r"<[^>]+>", "", summary).strip()[:300],
                        "url":     link,
                        "source":  feed_info["name"],
                        "date":    pub_date,
                        "sentiment":        None,
                        "sentiment_reason": None,
                    })
        except Exception as e:
            logger.warning(f"RSS fetch failed for {feed_info['name']}: {e}")

    # Deduplicate by title, keep latest 10
    seen_titles = set()
    articles = []
    for a in raw_articles:
        t = a["title"].lower()[:60]
        if t not in seen_titles:
            seen_titles.add(t)
            articles.append(a)
        if len(articles) >= 10:
            break

    result["articles_found"] = len(articles)

    if not articles:
        result["overall_label"] = "neutral"
        result["overall_score"] = 0.0
        result["error"] = "No recent news found for this stock."
        return result

    # ── Score sentiment ────────────────────────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        articles = _groq_sentiment(articles, bare_symbol, company_name, groq_key)
    else:
        logger.info("No GROQ_API_KEY found — using keyword heuristics")
        articles = _heuristic_sentiment(articles, bare_symbol)

    # ── Aggregate score ───────────────────────────────────────────────────
    label_to_score = {"POSITIVE": 1.0, "NEGATIVE": -1.0, "NEUTRAL": 0.0}
    scored = [a for a in articles if a.get("sentiment") in label_to_score]
    if scored:
        avg = sum(label_to_score[a["sentiment"]] for a in scored) / len(scored)
        result["overall_score"] = round(avg, 3)
        if avg > 0.2:
            result["overall_label"] = "positive"
        elif avg < -0.2:
            result["overall_label"] = "negative"
        else:
            result["overall_label"] = "neutral"
    else:
        result["overall_score"] = 0.0
        result["overall_label"] = "neutral"

    result["articles"] = articles
    return result


def _groq_sentiment(articles: List[dict], symbol: str, company_name: str, api_key: str) -> List[dict]:
    """Score each article's sentiment using Groq (LLaMA 3) — free tier."""
    headlines = "\n".join([
        f'{i+1}. "{a["title"]}"' for i, a in enumerate(articles)
    ])
    company_ref = company_name or symbol

    prompt = f"""You are a financial news sentiment analyst for Indian stock markets.
Rate each of these {len(articles)} news headlines about {company_ref} stock from an Indian investor's perspective.

Respond with ONLY a valid JSON array, no extra text, no markdown:
[
  {{"n": 1, "sentiment": "POSITIVE", "reason": "brief reason"}},
  {{"n": 2, "sentiment": "NEGATIVE", "reason": "brief reason"}},
  ...
]

Sentiment must be exactly one of: POSITIVE, NEGATIVE, NEUTRAL

Headlines:
{headlines}"""

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 600,
            },
            timeout=15,
        )

        if resp.status_code != 200:
            logger.warning(f"Groq returned {resp.status_code}: {resp.text[:200]}")
            return _heuristic_sentiment(articles, symbol)

        raw = resp.json()
        text = raw["choices"][0]["message"]["content"].strip()

        # Extract JSON array (handle if model wraps in markdown)
        json_match = re.search(r'\[.*?\]', text, re.DOTALL)
        if not json_match:
            logger.warning(f"Could not parse Groq JSON response: {text[:200]}")
            return _heuristic_sentiment(articles, symbol)

        ratings = json.loads(json_match.group())
        for rating in ratings:
            idx = rating.get("n", 0) - 1
            if 0 <= idx < len(articles):
                articles[idx]["sentiment"]        = rating.get("sentiment", "NEUTRAL").upper()
                articles[idx]["sentiment_reason"] = rating.get("reason", "")

    except Exception as e:
        logger.warning(f"Groq sentiment failed: {e}")
        return _heuristic_sentiment(articles, symbol)

    # Fill any unscored articles
    for a in articles:
        if not a.get("sentiment"):
            a["sentiment"] = "NEUTRAL"
            a["sentiment_reason"] = "Could not score"

    return articles


def _heuristic_sentiment(articles: List[dict], symbol: str) -> List[dict]:
    """Keyword-based fallback when Groq is unavailable."""
    positive_kw = [
        "surge", "rally", "gain", "record", "profit", "beat", "high",
        "rise", "upgrade", "buy", "strong", "bullish", "growth", "breakout",
        "positive", "outperform", "upside", "acquisition", "dividend",
        "revenue", "expansion", "deal", "partnership", "wins", "awarded",
    ]
    negative_kw = [
        "fall", "drop", "loss", "crash", "plunge", "sell", "weak", "bearish",
        "decline", "downgrade", "miss", "cut", "low", "concern", "risk",
        "warning", "penalty", "lawsuit", "fraud", "negative", "underperform",
        "debt", "default", "probe", "investigation", "tumbles", "slips",
    ]
    for a in articles:
        text = (a["title"] + " " + a.get("summary", "")).lower()
        pos  = sum(1 for w in positive_kw if w in text)
        neg  = sum(1 for w in negative_kw if w in text)
        if pos > neg:
            a["sentiment"]        = "POSITIVE"
            a["sentiment_reason"] = "Positive keywords detected (heuristic)"
        elif neg > pos:
            a["sentiment"]        = "NEGATIVE"
            a["sentiment_reason"] = "Negative keywords detected (heuristic)"
        else:
            a["sentiment"]        = "NEUTRAL"
            a["sentiment_reason"] = "Mixed or neutral language"
    return articles
