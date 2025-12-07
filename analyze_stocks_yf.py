import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import json

# ====== Helper ======
def r(v):
    return round(v, 2) if isinstance(v, (int, float)) else None

def r_pct(v):
    return round(v, 2) if isinstance(v, (int, float)) else None

def safe_float(v):
    return float(v) if isinstance(v, (int, float, np.floating)) else None

def to_percent(v):
    v = safe_float(v)
    return round(v * 100, 2) if v is not None else None

def none_if_nan(v):
    if v is None:
        return None
    try:
        if np.isnan(v):
            return None
    except TypeError:
        pass
    return v

def recommend(peg, upside):
    peg = safe_float(peg)
    upside = safe_float(upside)

    if peg is not None and upside is not None:
        if peg <= 1 and upside >= 20:
            return "BUY"
        if peg <= 1.5 and upside >= 10:
            return "ACCUMULATE"
        if upside < 0:
            return "AVOID"
    return "HOLD"

def format_yf_recommendation(key):
    if not key:
        return "NONE"
    return key.replace("_", " ").upper()

# ====== Technical Indicators & Scoring Engine ======
# All scores are designed so that:
#   - TechnicalScore roughly ranges from about -25 to +30
#   - FundamentalScore (Valuation + Growth + Quality) roughly ranges from about -25 to +30
#   - Upside / Volume / RSI provide additional fine‑tuning
# This keeps the unified DecisionScore stable around a 0–100 scale.

# ======================= RSI Score =======================
# RSI14 – measures overbought / oversold
#   < 30  : strongly oversold  → positive score (potential buy zone)
#   30–40 : mildly oversold    → small positive score
#   40–60 : neutral             → very small bias
#   60–70 : mildly overbought   → small negative score
#   > 70  : strongly overbought → stronger negative score
def score_rsi(rsi):
    if rsi is None:
        return 0
    if rsi < 30:
        return 8
    if rsi < 40:
        return 4
    if rsi < 60:
        return 2
    if rsi < 70:
        return -4
    return -8


# ======================= MACD Score =======================
# MACD evaluates trend strength / momentum:
#   macd > signal and hist > 0  : strong uptrend   → +8
#   macd > signal               : mild uptrend     → +4
#   macd < signal and hist < 0  : strong downtrend → -6
#   otherwise                   : neutral
def score_macd(macd, signal, hist):
    if macd is None or signal is None or hist is None:
        return 0
    if macd > signal and hist > 0:
        return 8
    if macd > signal:
        return 4
    if macd < signal and hist < 0:
        return -6
    return 0


# ======================= Trend Score (SMA/EMA) =======================
# Trend score looks at medium‑term structure:
#   Price > SMA20 > SMA50 and EMA20 > EMA50 -> strong uptrend   → +10
#   Price > SMA20 and EMA20 > EMA50        -> healthy uptrend   → +6
#   Price > SMA50                          -> still constructive→ +3
#   Clear downtrend (SMA20 < SMA50, EMA20 < EMA50, price < SMA20) → -6
#   Otherwise                                                 → 0
def score_trend(price, sma20, sma50, ema20, ema50):
    if None in (price, sma20, sma50, ema20, ema50):
        return 0

    # Strong bullish structure
    if price > sma20 > sma50 and ema20 > ema50:
        return 10

    # Decent bullish structure
    if price > sma20 and ema20 > ema50:
        return 6

    # Price above long/medium‑term average is still constructive
    if price > sma50:
        return 3

    # Clear downtrend: both MAs pointing lower and price under short MA
    if sma20 < sma50 and ema20 < ema50 and price < sma20:
        return -6

    return 0


# ======================= Bollinger Bands Score =======================
# Position of price inside Bollinger Bands (20, 2):
#   Price <= lower band : strongly oversold  → +4
#   Price >= upper band : strongly overbought→ -4
#   Price > middle band : slightly bullish   → +1
#   Price < middle band : slightly bearish   → -1
def score_bb(price, bb_mid, bb_low, bb_up):
    if None in (price, bb_mid, bb_low, bb_up):
        return 0

    score = 0

    if price <= bb_low:
        score += 4
    if price >= bb_up:
        score -= 4

    if price > bb_mid:
        score += 1
    else:
        score -= 1

    return score


# ======================= Risk Score (ATR/Price) =======================
# Relative volatility (daily range vs price):
#   ATR/Price < 2%  : calm / easier to hold  → +3
#   2–4%            : normal                 → 0
#   > 4%            : highly volatile        → -4
def score_risk(atr14, price):
    if atr14 is None or price is None or price == 0:
        return 0
    risk = atr14 / price
    if risk < 0.02:
        return 3
    if risk < 0.04:
        return 0
    return -4


# ======================= Volume Ratio Score =======================
# VolumeRatio = current volume / 20‑day average volume
#   > 2.0 : very strong money inflow / breakout   → +6
#   1.5–2 : strong confirmation of move          → +4
#   1.0–1.5 : healthy participation              → +2
#   0.7–1.0 : normal / neutral                   → 0
#   < 0.7 : weak participation                   → -3
def score_volume(vr):
    if vr is None:
        return 0
    if vr > 2.0:
        return 6
    if vr > 1.5:
        return 4
    if vr > 1.0:
        return 2
    if vr >= 0.7:
        return 0
    return -3


# ======================= Upside Score =======================
# Upside (%) between current price and analyst average target:
#   > 40% : very attractive upside       → +18
#   30–40%: strong upside                → +14
#   20–30%: good upside                  → +10
#   10–20%: moderate upside              → +6
#   0–10% : slightly positive            → +3
#   < 0   : priced above target (stretched) → -6
def score_upside(u):
    if u is None:
        return 0
    if u > 40:
        return 18
    if u > 30:
        return 14
    if u > 20:
        return 10
    if u > 10:
        return 6
    if u > 0:
        return 3
    return -6


# ======================= Valuation Score (PEG, PE) =======================
# Valuation is kept more moderate to avoid over‑penalizing growth stocks.
# PEG:
#   <= 1.0 : very attractive vs growth   → +10
#   1.0–1.5: reasonable                  → +6
#   1.5–2.5: acceptable                  → +2
#   > 3.5  : clearly expensive           → -6
#
# Forward PE:
#   < 20   : cheap / reasonable          → +6
#   20–30 : fair                         → +3
#   > 45  : rich                         → -6
#
# Trailing PE:
#   > 50  : very expensive on past earnings → -4
def score_valuation(peg, pe_fwd, pe_trailing):
    peg = safe_float(peg)
    pe_fwd = safe_float(pe_fwd)
    pe_trailing = safe_float(pe_trailing)
    score = 0

    if peg is not None:
        if peg <= 1.0:
            score += 10
        elif peg <= 1.5:
            score += 6
        elif peg <= 2.5:
            score += 2
        elif peg > 3.5:
            score -= 6

    if pe_fwd is not None:
        if pe_fwd < 20:
            score += 6
        elif pe_fwd <= 30:
            score += 3
        elif pe_fwd > 45:
            score -= 6

    if pe_trailing is not None and pe_trailing > 50:
        score -= 4

    return score


# ======================= Growth Score (Revenue, Earnings) =======================
# revenueGrowth / earningsGrowth are decimals (0.20 = 20% YoY).
# Revenue growth:
#   >= 20% : strong top‑line growth      → +8
#   10–20% : good                        → +4
#   0–10%  : mildly positive             → +1
#   < 0    : shrinking revenue           → -6
#
# Earnings growth:
#   >= 25% : very strong earnings growth → +8
#   12–25% : good                        → +4
#   0–12%  : mildly positive             → +1
#   < 0    : shrinking earnings          → -8
def score_growth(rev_g, earn_g):
    rev_g = safe_float(rev_g)
    earn_g = safe_float(earn_g)
    score = 0

    if rev_g is not None:
        if rev_g >= 0.20:
            score += 8
        elif rev_g >= 0.10:
            score += 4
        elif rev_g >= 0:
            score += 1
        else:
            score -= 6

    if earn_g is not None:
        if earn_g >= 0.25:
            score += 8
        elif earn_g >= 0.12:
            score += 4
        elif earn_g >= 0:
            score += 1
        else:
            score -= 8

    return score


# ======================= Quality Score (ROE, Margins) =======================
# ROE (returnOnEquity, decimal):
#   >= 25% : excellent capital efficiency → +8
#   15–25% : good                         → +4
#   8–15%  : acceptable                   → +1
#   < 5%   : weak                         → -6
#
# Gross margin:
#   >= 55% : very strong moat             → +4
#   40–55% : healthy                      → +2
#   25–40% : neutral                      → 0
#   < 25%  : structurally weak            → -4
#
# Profit margin:
#   >= 22% : very profitable              → +4
#   12–22% : good                         → +2
#   0–12%  : low but positive             → 0
#   < 0    : loss‑making                  → -6
def score_quality(roe, gross_m, profit_m):
    roe = safe_float(roe)
    gross_m = safe_float(gross_m)
    profit_m = safe_float(profit_m)
    score = 0

    if roe is not None:
        if roe >= 0.25:
            score += 8
        elif roe >= 0.15:
            score += 4
        elif roe >= 0.08:
            score += 1
        elif roe < 0.05:
            score -= 6

    if gross_m is not None:
        if gross_m >= 0.55:
            score += 4
        elif gross_m >= 0.40:
            score += 2
        elif gross_m < 0.25:
            score -= 4

    if profit_m is not None:
        if profit_m >= 0.22:
            score += 4
        elif profit_m >= 0.12:
            score += 2
        elif profit_m < 0:
            score -= 6

    return score


# ======================= Unified Decision Score =======================
# Goal: create a 0–100 DecisionScore that blends:
#   - Technicals (price action & indicators)
#   - Fundamentals (valuation, growth, quality)
#   - Analyst upside (target price vs current)
#   - Volume & RSI (timing / money flow)
#
# Weighting for Balanced Style (Version B):
#   25%  TechnicalScore
#   30%  FundamentalScore (Valuation + Growth + Quality)
#   35%  UpsideScore (analyst targets)
#   10%  Volume + RSI (5% each)
#
# Typical raw range is roughly [-30, +40].
# We clamp to [-30, +40] and then rescale into [0, 100].
def unified_decision_score(tech, val, growth, quality, upside, rsi, vol_ratio):
    # Normalize technical and fundamental ranges to avoid extremes dominating.
    tech_norm = max(min(tech, 30), -25)
    fundamental = max(min(val + growth + quality, 30), -25)

    upside_score = score_upside(upside)
    vol_score = score_volume(vol_ratio)
    rsi_score = score_rsi(rsi)

    decision_raw = (
        0.25 * tech_norm +
        0.30 * fundamental +
        0.35 * upside_score +
        0.05 * vol_score +
        0.05 * rsi_score
    )

    # Convert to 0–100 band.
    raw_min, raw_max = -30.0, 40.0
    decision_clamped = max(min(decision_raw, raw_max), raw_min)
    decision_norm = (decision_clamped - raw_min) / (raw_max - raw_min) * 100.0

    return decision_norm


# ======================= Technical Signal (only TechnicalScore) =======================
def get_signal(score):
    if score is None:
        return ""
    if score >= 75:
        return "STRONG BUY"
    elif score >= 65:
        return "BUY"
    elif score >= 50:
        return "WATCH"
    elif score >= 40:
        return "SELL"
    else:
        return "STRONG SELL"


# ======================= Final Decision Signal (based on DecisionScore 0–100) =======================
# DecisionScore mapping:
#   >= 80 : STRONG BUY  – strong fundamentals + technicals + upside
#   >= 65 : BUY         – positive skew overall
#   >= 50 : WATCH       – hold / monitor, not a clear edge
#   >= 40 : SELL        – reduce exposure / be cautious
#   <  40 : STRONG SELL – avoid / exit unless special thesis
def final_decision_signal_v2(score):
    if score is None:
        return ""
    if score >= 80:
        return "STRONG BUY"
    elif score >= 65:
        return "BUY"
    elif score >= 50:
        return "WATCH"
    elif score >= 40:
        return "SELL"
    else:
        return "STRONG SELL"


def get_stock(symbol: str):
    try:
        yf_t = yf.Ticker(symbol)
        info = yf_t.info

        # === FUNDAMENTAL / VALUATION INFO ===
        pe_trailing = info.get("trailingPE")
        pe_forward = info.get("forwardPE")
        peg = info.get("trailingPegRatio")

        market_cap = info.get("marketCap")
        dividend_yield = info.get("dividendYield")
        beta = info.get("beta")

        revenue_growth = info.get("revenueGrowth")
        earnings_growth = info.get("earningsGrowth")
        gross_margins = info.get("grossMargins")
        profit_margins = info.get("profitMargins")
        roe = info.get("returnOnEquity")

        free_cashflow = info.get("freeCashflow")

        df = yf_t.history(period="6mo")
        if df is None or df.empty:
            return {"symbol": symbol, "error": "No price history"}

        price = info.get("regularMarketPrice")
        target_avg = info.get("targetMeanPrice")

        raw_upside = ((target_avg - price) / price) * 100 if price and target_avg else None
        upside = r_pct(raw_upside)

        # === TECHNICAL INDICATORS ===
        close = df["Close"]
        volume = df["Volume"]

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi14 = 100 - (100 / (1 + rs.iloc[-1]))

        # SMA / EMA
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        ema20 = close.ewm(span=20).mean().iloc[-1]
        ema50 = close.ewm(span=50).mean().iloc[-1]

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_val = ema12 - ema26
        macd = macd_val.iloc[-1]
        signal = macd_val.ewm(span=9).mean().iloc[-1]
        hist = macd - signal

        # ATR14
        high = df["High"]
        low = df["Low"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]

        # Bollinger Bands (20,2)
        bb_mid = close.rolling(20).mean().iloc[-1]
        std = close.rolling(20).std().iloc[-1]
        bb_up = bb_mid + 2 * std
        bb_low = bb_mid - 2 * std

        # Volume Ratio
        vol_avg20 = volume.rolling(20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_avg20 if vol_avg20 else None

        # === CALCULATE TECHNICAL SCORE ===
        tech_score = (
            score_rsi(rsi14) +
            score_macd(macd, signal, hist) +
            score_trend(price, sma20, sma50, ema20, ema50) +
            score_bb(price, bb_mid, bb_low, bb_up) +
            score_risk(atr14, price) +
            score_volume(vol_ratio) +
            score_upside(upside)
        )

        # === FUNDAMENTAL SCORES ===
        val_score = score_valuation(peg, pe_forward, pe_trailing)
        growth_score = score_growth(revenue_growth, earnings_growth)
        quality_score = score_quality(roe, gross_margins, profit_margins)

        total_score = tech_score + val_score + growth_score + quality_score
        # === DECISION ENGINE 2.0 ===
        decision_score = unified_decision_score(
            tech_score,
            val_score,
            growth_score,
            quality_score,
            upside,
            rsi14,
            vol_ratio
        )

        decision_signal = final_decision_signal_v2(decision_score)

        return {
            "symbol": symbol,
            "company": info.get("shortName"),

            # === PRICE & CHANGE ===
            "price": round(price, 2) if price is not None else None,
            "change": r_pct(info.get("regularMarketChange")),
            "changePercent": r_pct(info.get("regularMarketChangePercent")),
            "dayHigh": round(info.get("regularMarketDayHigh"), 2) if info.get("regularMarketDayHigh") else None,
            "dayLow": round(info.get("regularMarketDayLow"), 2) if info.get("regularMarketDayLow") else None,
            "avg50days": round(info.get("fiftyDayAverage"), 2) if info.get("fiftyDayAverage") else None,

            # === TARGET PRICE / ANALYST VIEW ===
            "targetLow": round(info.get("targetLowPrice"), 2) if info.get("targetLowPrice") else None,
            "targetAvg": round(target_avg, 2) if target_avg else None,
            "targetHigh": round(info.get("targetHighPrice"), 2) if info.get("targetHighPrice") else None,
            "upside": round(upside, 2) if upside is not None else None,
            "yfRecommendation": info.get("recommendationKey"),
            "numberAnalyst": info.get("numberOfAnalystOpinions"),

            # === VALUATION METRICS ===
            "trailingPE": r(pe_trailing),
            "forwardPE": r(pe_forward),
            "PEG": r(peg),
            "marketCap": int(market_cap) if market_cap else None,
            "dividendYieldPct": to_percent(dividend_yield),
            "beta": r(beta),

            # === FUNDAMENTAL GROWTH / QUALITY ===
            "revenueGrowthPct": to_percent(revenue_growth),
            "earningsGrowthPct": to_percent(earnings_growth),
            "grossMarginsPct": to_percent(gross_margins),
            "profitMarginsPct": to_percent(profit_margins),
            "returnOnEquityPct": to_percent(roe),
            "freeCashflow": int(free_cashflow) if free_cashflow else None,

            # === SIMPLE RECOMMENDATION (PEG + UPSIDE) ===
            "recommendation": recommend(peg, upside),

            # === TECHNICAL SCORE / SIGNAL ===
            "TechnicalScore": tech_score,
            "techSignal": get_signal(tech_score),

            # === FUNDAMENTAL SCORES ===
            "ValuationScore": val_score,
            "GrowthScore": growth_score,
            "QualityScore": quality_score,
            "TotalScore": total_score,
            "DecisionScore": round(decision_score, 2),
            "DecisionSignal": decision_signal,

            # === TECHNICAL OUTPUT (DETAIL) ===
            "RSI14": round(rsi14, 2),
            "SMA20": round(sma20, 2),
            "SMA50": round(sma50, 2),
            "EMA20": round(ema20, 2),
            "EMA50": round(ema50, 2),

            "MACD": round(macd, 2),
            "MACD_Hist": round(hist, 2),

            "ATR14": round(atr14, 2),

            "BB_Middle": round(bb_mid, 2),
            "BB_Upper": round(bb_up, 2),
            "BB_Lower": round(bb_low, 2),

            "Volume": int(volume.iloc[-1]) if volume.iloc[-1] else None,
            "VolumeAvg20": int(vol_avg20) if vol_avg20 else None,
            "VolumeRatio": round(vol_ratio, 2) if vol_ratio else None,
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

# ====== MAIN (multi-threaded) ======
def main(symbols):
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_stock, s) for s in symbols]
        for f in as_completed(futures):
            results.append(f.result())
    print(json.dumps(results))

if __name__ == "__main__":
    # ✅ Nhận danh sách dạng: "AAPL,MSFT,NVDA,TSLA"
    raw = sys.argv[1]
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    main(symbols)
