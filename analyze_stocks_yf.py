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

# ====== Technical Indicators ======

# ======================= RSI (Relative Strength Index) =======================
# RSI14 – đo sức mạnh xu hướng dựa trên mức tăng/giảm trung bình 14 ngày.
# - RSI < 30: Oversold → khả năng bật mạnh → cộng điểm cao.
# - 30–70: Trung tính.
# - RSI > 70: Overbought → rủi ro điều chỉnh → trừ điểm.
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ======================= MACD (Moving Average Convergence Divergence) =======================
# MACD đo độ chênh giữa EMA12 và EMA26.
# - MACD > Signal và Histogram > 0 → xu hướng tăng mạnh → +15 điểm.
# - MACD > Signal → tăng yếu → +10 điểm.
# - MACD < Signal và Histogram < 0 → giảm mạnh → -10 điểm.
def calc_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

# ======================= ATR14 (Average True Range) =======================
# ATR đo mức độ biến động (volatility).
# Risk = ATR14 / Price:
# - <0.01 → biến động thấp → +10 điểm.
# - <0.02 → +5 điểm.
# - <0.03 → 0 điểm.
# - >=0.03 → rất rủi ro → -10 điểm.
def calc_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# ======================= Bollinger Bands (20, 2) =======================
# Dải BB giúp phát hiện overbought/oversold:
# - Price gần Lower Band → oversold → +10 điểm.
# - Price vượt Upper Band → overbought → -10 điểm.
# - Price > Mid (SMA20) → xu hướng tăng → +5 điểm.
# - Price < Mid → xu hướng giảm → -5 điểm.
def calc_bbands(series, period=20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    return sma, upper, lower

def score_rsi(rsi):
    # Chấm điểm RSI theo mức độ overbought/oversold
    if rsi is None: return 0
    if rsi < 25: return 20
    if rsi < 30: return 15
    if rsi < 40: return 10
    if rsi < 55: return 5
    if rsi < 70: return 0
    if rsi < 80: return -5
    return -10

def score_macd(macd, signal, hist):
    if macd is None or signal is None or hist is None: return 0
    if macd > signal and hist > 0: return 15
    if macd > signal: return 10
    # MACD dưới Signal và Histogram âm → xu hướng giảm mạnh
    if macd < signal and hist < 0: return -10
    return 0

# ======================= Xu hướng SMA/EMA =======================
# Trend mạnh khi:
# Price > SMA20 > SMA50 và EMA20 > EMA50 → Uptrend mạnh → điểm cao nhất.
# Nếu chỉ vượt một phần → cộng ít điểm.
# Nếu SMA50 > SMA20 hoặc giá dưới đường trung bình → xu hướng yếu → trừ điểm.
def score_trend(price, sma20, sma50, ema20, ema50):
    if None in (price, sma20, sma50, ema20, ema50): return 0
    score = 0
    if price > sma20 and sma20 > sma50: score += 10
    elif price > sma20: score += 5
    elif sma20 > sma50: score += 5
    else: score -= 10

    if ema20 > ema50: score += 10
    elif price > ema20: score += 5
    else: score -= 10
    return score

# ======================= Bollinger Score =======================
# Đánh giá vị trí giá trong dải BB:
# - Dưới Mid → bearish → -5
# - Trên Mid → bullish → +5
# - Chạm Lower Band → rất oversold → +10
# - Chạm Upper Band → rất overbought → -10
def score_bb(price, bb_mid, bb_low, bb_up):
    if None in (price, bb_mid, bb_low, bb_up): return 0
    score = 0
    if price > bb_mid: score += 5
    else: score -= 5

    if price <= bb_low: score += 10
    if price >= bb_up: score -= 10
    return score

# ======================= Risk Score (ATR/Price) =======================
# Tỷ lệ biến động/giá càng thấp → càng an toàn.
def score_risk(atr14, price):
    if atr14 is None or price is None or price == 0: return 0
    risk = atr14 / price
    if risk < 0.01: return 10
    if risk < 0.02: return 5
    if risk < 0.03: return 0
    return -10

# ======================= Volume Ratio Score =======================
# Volume hiện tại / Volume trung bình 20 ngày:
# >1.5 → breakout mạnh có volume → +5
# >1.0 → xác nhận xu hướng → +3
# <0.7 → yếu → -5
def score_volume(vr):
    if vr is None: return 0
    if vr > 1.5: return 5
    if vr > 1.0: return 3
    if vr < 0.7: return -5
    return 0

# ======================= Upside Score =======================
# Upside tính dựa trên TargetMeanPrice:
# >30% → +20, >20% → +15, >10% → +10, >0 → +5, <0 → -5.
def score_upside(u):
    if u is None: return 0
    if u > 30: return 20
    if u > 20: return 15
    if u > 10: return 10
    if u > 0: return 5
    return -5

# ======================= Valuation Score =======================
# PEG <1 → undervalued → +20
# PEG <1.5 → hơi rẻ → +10
# PEG >2.5 → đắt → -10
# Forward PE <25 → rẻ → +10
# Forward PE >40 → đắt → -10
# Trailing PE >45 → quá đắt → -5
def score_valuation(peg, pe_fwd, pe_trailing):
    peg = safe_float(peg)
    pe_fwd = safe_float(pe_fwd)
    pe_trailing = safe_float(pe_trailing)
    score = 0

    # PEG: <=1 rất tốt, <=1.5 chấp nhận, >=2.5 đắt
    if peg is not None:
        if peg <= 1:
            score += 20
        elif peg <= 1.5:
            score += 10
        elif peg >= 2.5:
            score -= 10

    # Forward PE: <=25 tốt, >=40 đắt
    if pe_fwd is not None:
        if pe_fwd <= 25:
            score += 10
        elif pe_fwd >= 40:
            score -= 10

    # Trailing PE quá cao cũng trừ điểm
    if pe_trailing is not None and pe_trailing >= 45:
        score -= 5

    return score


# ======================= Growth Score =======================
# Dựa trên tăng trưởng doanh thu & lợi nhuận:
# Revenue >15% → +10, >8% → +5, <0 → -10
# Earnings >20% → +10, >10% → +5, <0 → -10
def score_growth(rev_g, earn_g):
    rev_g = safe_float(rev_g)
    earn_g = safe_float(earn_g)
    score = 0

    # revenueGrowth / earningsGrowth là dạng 0.x (tương đương x%)
    if rev_g is not None:
        if rev_g >= 0.15:
            score += 10
        elif rev_g >= 0.08:
            score += 5
        elif rev_g <= 0:
            score -= 10

    if earn_g is not None:
        if earn_g >= 0.20:
            score += 10
        elif earn_g >= 0.10:
            score += 5
        elif earn_g <= 0:
            score -= 10

    return score


# ======================= Quality Score =======================
# ROE:
# >40% → excellent → +10
# >20% → strong → +5
# <10% → yếu → -5
#
# Gross Margin:
# >50% → top-tier → +5
# >35% → healthy → +3
# <20% → yếu → -5
#
# Profit Margin:
# >25% → excellent → +5
# >15% → good → +3
# <5% → rất yếu → -5
def score_quality(roe, gross_m, profit_m):
    roe = safe_float(roe)
    gross_m = safe_float(gross_m)
    profit_m = safe_float(profit_m)
    score = 0

    # ROE, grossMargins, profitMargins dạng 0.x
    if roe is not None:
        if roe >= 0.4:
            score += 10
        elif roe >= 0.2:
            score += 5
        elif roe <= 0.1:
            score -= 5

    if gross_m is not None:
        if gross_m >= 0.5:
            score += 5
        elif gross_m >= 0.35:
            score += 3
        elif gross_m <= 0.2:
            score -= 5

    if profit_m is not None:
        if profit_m >= 0.25:
            score += 5
        elif profit_m >= 0.15:
            score += 3
        elif profit_m <= 0.05:
            score -= 5

    return score


# ======================= Technical Signal =======================
# Chuyển điểm TechnicalScore → tín hiệu ngắn hạn:
# >=75 → STRONG BUY
# >=65 → BUY
# >=50 → WATCH
# >=40 → SELL
# <40 → STRONG SELL
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

# ======================= Final Decision Signal =======================
# Dựa trên toàn bộ yếu tố: Technical + Valuation + Growth + Quality + Upside
# >=85 và Upside>=20% → STRONG BUY
# >=70 và Upside>=10% → BUY
# >=55 → WATCH
# >=40 → SELL
# thấp hơn → STRONG SELL
def final_decision_signal(total_score, upside, yf_rec_key):
    upside = safe_float(upside)
    key = (yf_rec_key or "").lower()

    if total_score is None:
        return ""

    if total_score >= 85 and upside is not None and upside >= 20:
        return "STRONG BUY"
    if total_score >= 70 and upside is not None and upside >= 10:
        return "BUY"
    if total_score >= 55:
        return "WATCH"
    if total_score >= 40:
        return "SELL"
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
        decision_signal = final_decision_signal(total_score, upside, info.get("recommendationKey"))

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
