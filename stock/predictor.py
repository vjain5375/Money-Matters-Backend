"""
predictor.py — Buy / Hold / Sell Ensemble Predictor
=====================================================
Combines Fundamental Score + Technical Score + Sentiment Score
into a final weighted signal for Indian stocks.

No GPU / ML training required — rule-based ensemble.
"""

import logging
from typing import Optional

logger = logging.getLogger("MoneyMattersAI.Stock.Predictor")


def predict(fundamentals: dict, technical: dict, sentiment: dict) -> dict:
    """
    Ensemble predictor combining 3 signal sources.

    Weights:
        Fundamentals  40%
        Technical     35%
        Sentiment     25%

    Returns:
        dict with: signal (BUY/HOLD/SELL), confidence (%), final_score (0-100),
                   fundamental_score, technical_score, sentiment_score,
                   reasons (list of strings), disclaimer
    """
    result = {
        "signal":            "HOLD",
        "confidence":        0,
        "final_score":       50,
        "fundamental_score": 50,
        "technical_score":   50,
        "sentiment_score":   50,
        "reasons":           [],
        "score_breakdown":   {},
        "disclaimer":        "⚠️ This is not financial advice. Use this purely for educational reference.",
    }

    reasons = []
    fs = _fundamental_score(fundamentals, reasons)
    ts = _technical_score(technical, reasons)
    ss = _sentiment_score(sentiment, reasons)

    final = round(0.40 * fs + 0.35 * ts + 0.25 * ss, 1)

    if final >= 65:
        signal = "BUY"
        confidence = min(95, round(50 + (final - 65) * 2.0))
    elif final <= 38:
        signal = "SELL"
        confidence = min(95, round(50 + (38 - final) * 2.0))
    else:
        signal = "HOLD"
        confidence = round(50 + abs(final - 51) * 0.8)

    result["signal"]            = signal
    result["confidence"]        = confidence
    result["final_score"]       = final
    result["fundamental_score"] = round(fs, 1)
    result["technical_score"]   = round(ts, 1)
    result["sentiment_score"]   = round(ss, 1)
    result["reasons"]           = reasons
    result["score_breakdown"]   = {
        "Fundamental (40%)": round(fs, 1),
        "Technical (35%)":   round(ts, 1),
        "Sentiment (25%)":   round(ss, 1),
    }

    return result


def _fundamental_score(f: dict, reasons: list) -> float:
    """Score fundamentals 0–100."""
    if not f or f.get("error"):
        reasons.append("⚠️ Fundamental data unavailable — using neutral score")
        return 50.0

    score = 50.0   # start neutral

    # --- Piotroski F-Score (0–9) — highest weight fundamental signal ---
    pf = f.get("piotroski_score")
    if pf is not None:
        pf_score = (pf / 9) * 100
        # Weight: 25 pts of final fundamental score
        contribution = (pf_score - 50) * 0.45
        score += contribution
        if pf >= 7:
            reasons.append(f"✅ Piotroski F-Score: {pf}/9 — Strong fundamentals")
        elif pf >= 4:
            reasons.append(f"➡️ Piotroski F-Score: {pf}/9 — Average financials")
        else:
            reasons.append(f"🔴 Piotroski F-Score: {pf}/9 — Weak financials")

    # --- Altman Z-Score ---
    az = f.get("altman_z_score")
    az_zone = f.get("altman_zone", "")
    if az is not None:
        if az_zone == "safe":
            score += 12
            reasons.append(f"✅ Altman Z-Score: {az:.2f} — Financially safe zone")
        elif az_zone == "grey":
            score += 2
            reasons.append(f"➡️ Altman Z-Score: {az:.2f} — Grey zone (monitor)")
        elif az_zone == "distress":
            score -= 18
            reasons.append(f"🔴 Altman Z-Score: {az:.2f} — Distress zone")

    # --- P/E Ratio (sector-agnostic heuristic) ---
    pe = f.get("pe_ratio")
    if pe is not None and pe > 0:
        if pe < 10:
            score += 8
            reasons.append(f"✅ P/E: {pe:.1f} — Attractively valued")
        elif pe < 25:
            score += 4
            reasons.append(f"➡️ P/E: {pe:.1f} — Fairly valued")
        elif pe < 50:
            score -= 2
            reasons.append(f"⚠️ P/E: {pe:.1f} — Slightly expensive")
        else:
            score -= 8
            reasons.append(f"🔴 P/E: {pe:.1f} — High valuation risk")

    # --- ROE ---
    roe = f.get("roe")
    if roe is not None:
        pct = roe * 100
        if pct > 20:
            score += 6
            reasons.append(f"✅ ROE: {pct:.1f}% — Excellent return on equity")
        elif pct > 10:
            score += 2
        elif pct < 0:
            score -= 8
            reasons.append(f"🔴 ROE: {pct:.1f}% — Negative returns")

    # --- Debt/Equity ---
    de = f.get("debt_to_equity")
    if de is not None:
        if de < 0.3:
            score += 5
        elif de > 2.0:
            score -= 8
            reasons.append(f"🔴 Debt/Equity: {de:.2f} — High leverage")

    # --- Profit Margin ---
    pm = f.get("profit_margin")
    if pm is not None:
        pct = pm * 100
        if pct > 15:
            score += 5
        elif pct < 0:
            score -= 10
            reasons.append(f"🔴 Profit Margin: {pct:.1f}% — Unprofitable")

    return max(0.0, min(100.0, score))


def _technical_score(t: dict, reasons: list) -> float:
    """Score technical indicators 0–100."""
    if not t or t.get("error"):
        reasons.append("⚠️ Technical data unavailable — using neutral score")
        return 50.0

    score = 50.0
    latest = t.get("latest", {})
    signals = t.get("signals", [])

    # --- RSI ---
    rsi = latest.get("RSI_14")
    if rsi is not None:
        if rsi < 30:
            score += 15
            reasons.append(f"✅ RSI: {rsi:.1f} — Oversold (potential buy signal)")
        elif rsi < 45:
            score += 5
        elif rsi > 70:
            score -= 15
            reasons.append(f"🔴 RSI: {rsi:.1f} — Overbought (caution)")
        elif rsi > 60:
            score -= 3

    # --- MACD ---
    macd      = latest.get("MACD")
    macd_sig  = latest.get("MACD_signal")
    macd_hist = latest.get("MACD_hist")
    if macd is not None and macd_sig is not None:
        if macd > macd_sig:
            score += 10
            reasons.append("✅ MACD: Above signal line — bullish momentum")
        else:
            score -= 10
            reasons.append("🔴 MACD: Below signal line — bearish momentum")

    if macd_hist is not None:
        if macd_hist > 0:
            score += 3   # strengthening momentum
        else:
            score -= 3

    # --- Trend (Price vs EMA/SMA) ---
    price = latest.get("close")
    ema20 = latest.get("EMA_20")
    ema50 = latest.get("EMA_50")
    sma200 = latest.get("SMA_200")

    if price and ema20 and ema50:
        if price > ema20 > ema50:
            score += 12
            reasons.append("✅ Trend: Price above EMA20 & EMA50 — strong uptrend")
        elif price < ema20 < ema50:
            score -= 12
            reasons.append("🔴 Trend: Price below EMA20 & EMA50 — downtrend")
        elif price > ema20:
            score += 4
        else:
            score -= 4

    # Golden/Death Cross relative to SMA200
    if price and sma200:
        if price > sma200:
            score += 5
            reasons.append("✅ Price above 200-day SMA — long-term bullish")
        else:
            score -= 5
            reasons.append("⚠️ Price below 200-day SMA — long-term caution")

    # --- Bollinger Band position ---
    bb_upper = latest.get("BB_upper")
    bb_lower = latest.get("BB_lower")
    if price and bb_upper and bb_lower:
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_pos = (price - bb_lower) / bb_range  # 0=lower band, 1=upper band
            if bb_pos < 0.2:
                score += 8   # near lower band — potential bounce
            elif bb_pos > 0.85:
                score -= 6   # near upper band — overbought

    return max(0.0, min(100.0, score))


def _sentiment_score(s: dict, reasons: list) -> float:
    """Score news sentiment 0–100."""
    if not s or s.get("error") == "No recent news found for this stock.":
        reasons.append("➡️ Sentiment: No recent news — neutral by default")
        return 50.0

    raw_score  = s.get("overall_score", 0.0) or 0.0
    label      = s.get("overall_label", "neutral")
    num_articles = s.get("articles_found", 0)

    # Convert -1..+1 → 0..100
    score = 50.0 + (raw_score * 30.0)

    if label == "positive":
        reasons.append(f"✅ Sentiment: Mostly positive news ({num_articles} articles scanned)")
    elif label == "negative":
        reasons.append(f"🔴 Sentiment: Mostly negative news ({num_articles} articles scanned)")
    else:
        reasons.append(f"➡️ Sentiment: Neutral/mixed news ({num_articles} articles scanned)")

    # Low article count → dilute toward neutral
    if num_articles < 3:
        score = 0.6 * score + 0.4 * 50.0

    return max(0.0, min(100.0, score))
