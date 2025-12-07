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

# ======================= RSI Score =======================
# RSI14 – đo mức độ overbought / oversold
# - Rất thấp (oversold mạnh) → ưu tiên mua (điểm cao)
# - Trung tính → điểm nhỏ
# - Rất cao (overbought mạnh) → rủi ro điều chỉnh (trừ điểm)
def score_rsi(rsi):
    if rsi is None:
        return 0
    if rsi < 25:
        return 18      # oversold rất mạnh, cơ hội bắt đáy
    if rsi < 30:
        return 12
    if rsi < 40:
        return 6
    if rsi < 60:
        return 3       # vùng trung tính
    if rsi < 70:
        return -3      # hơi cao, cẩn trọng
    if rsi < 80:
        return -8
    return -15         # rất cao, dễ bị xả


# ======================= MACD Score =======================
# MACD đánh giá xu hướng/động lượng:
# - MACD > Signal và Histogram > 0 → uptrend mạnh
# - MACD > Signal → uptrend vừa
# - MACD < Signal, Hist < 0 → downtrend mạnh
def score_macd(macd, signal, hist):
    if macd is None or signal is None or hist is None:
        return 0
    if macd > signal and hist > 0:
        return 15
    if macd > signal:
        return 10
    if macd < signal and hist < 0:
        return -10
    return 0


# ======================= Trend Score (SMA/EMA) =======================
# Đánh giá xu hướng trung hạn:
# - Price > SMA20 > SMA50 và EMA20 > EMA50 → Uptrend mạnh.
# - Nếu chỉ thỏa một phần → cộng ít điểm.
# - Ngược lại → trừ điểm nhẹ (không quá nặng tay).
def score_trend(price, sma20, sma50, ema20, ema50):
    if None in (price, sma20, sma50, ema20, ema50):
        return 0

    score = 0

    # Cấu trúc SMA (trung hạn)
    if price > sma20 and sma20 > sma50:
        score += 12     # uptrend mạnh
    elif price > sma20:
        score += 6      # giá nằm trên SMA20, hơi tích cực
    elif sma20 > sma50:
        score += 4      # SMA20 > SMA50, xu hướng dài trung hạn vẫn tốt
    else:
        score -= 5      # xu hướng yếu

    # Cấu trúc EMA (ngắn hạn)
    if ema20 > ema50:
        score += 8
    elif price > ema20:
        score += 4
    else:
        score -= 5

    return score


# ======================= Bollinger Bands Score =======================
# Đánh giá vị trí giá trong dải BB:
# - Trên mid → hơi bullish
# - Dưới mid → hơi bearish
# - Chạm Lower band → oversold
# - Chạm Upper band → overbought
def score_bb(price, bb_mid, bb_low, bb_up):
    if None in (price, bb_mid, bb_low, bb_up):
        return 0

    score = 0

    # Giá so với đường giữa
    if price > bb_mid:
        score += 3
    else:
        score -= 3

    # Gần/đụng Lower/Upper band
    if price <= bb_low:
        score += 5      # rất oversold
    if price >= bb_up:
        score -= 5      # rất overbought

    return score


# ======================= Risk Score (ATR/Price) =======================
# Độ biến động tương đối:
# - ATR/Price thấp → cổ phiếu "êm", ít rung lắc → dễ nắm giữ → cộng điểm.
# - ATR/Price cao → biến động mạnh → trừ điểm nhưng không quá nặng.
def score_risk(atr14, price):
    if atr14 is None or price is None or price == 0:
        return 0
    risk = atr14 / price
    if risk < 0.01:
        return 8
    if risk < 0.02:
        return 4
    if risk < 0.03:
        return 0
    return -8


# ======================= Volume Ratio Score =======================
# VolumeRatio = Volume hiện tại / Volume trung bình 20 ngày
# - >1.5 → dòng tiền vào mạnh (breakout) → +6
# - >1.0 → dòng tiền xác nhận xu hướng → +3
# - <0.7 → volume yếu, tín hiệu kém tin cậy → -5
def score_volume(vr):
    if vr is None:
        return 0
    if vr > 1.5:
        return 6
    if vr > 1.0:
        return 3
    if vr < 0.7:
        return -5
    return 0


# ======================= Upside Score =======================
# Upside dựa trên TargetMeanPrice (% so với giá hiện tại)
# - >30% → tiềm năng rất tốt
# - >20% → tốt
# - >10% → tạm ổn
# - >0 → hơi tốt
# - <0 → bị định giá cao hơn target → trừ điểm nhưng không quá nặng
def score_upside(u):
    if u is None:
        return 0
    if u > 30:
        return 22
    if u > 20:
        return 16
    if u > 10:
        return 10
    if u > 0:
        return 5
    return -8


# ======================= Valuation Score (PEG, PE) =======================
# Đánh giá định giá:
# - PEG <=1 → rất rẻ so với tăng trưởng → +20
# - PEG <=1.5 → chấp nhận được → +10
# - PEG >=2.5 → khá đắt → -10
# - Forward PE <=25 → hợp lý → +8; >=40 → đắt → -8
# - Trailing PE >=45 → quá đắt → -5
def score_valuation(peg, pe_fwd, pe_trailing):
    peg = safe_float(peg)
    pe_fwd = safe_float(pe_fwd)
    pe_trailing = safe_float(pe_trailing)
    score = 0

    if peg is not None:
        if peg <= 1:
            score += 20
        elif peg <= 1.5:
            score += 10
        elif peg >= 2.5:
            score -= 10

    if pe_fwd is not None:
        if pe_fwd <= 25:
            score += 8
        elif pe_fwd >= 40:
            score -= 8

    if pe_trailing is not None and pe_trailing >= 45:
        score -= 5

    return score


# ======================= Growth Score (Revenue, Earnings) =======================
# revenueGrowth / earningsGrowth là dạng 0.x (tương đương x%)
# - Revenue >=15% → +10, >=8% → +5, <=0 → -10
# - Earnings >=20% → +10, >=10% → +5, <=0 → -10
def score_growth(rev_g, earn_g):
    rev_g = safe_float(rev_g)
    earn_g = safe_float(earn_g)
    score = 0

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


# ======================= Quality Score (ROE, Margin) =======================
# ROE (returnOnEquity):
#   >=40% → +10, >=20% → +5, <=10% → -5
# Gross Margin:
#   >=50% → +5, >=35% → +3, <=20% → -5
# Profit Margin:
#   >=25% → +5, >=15% → +3, <=5% → -5
def score_quality(roe, gross_m, profit_m):
    roe = safe_float(roe)
    gross_m = safe_float(gross_m)
    profit_m = safe_float(profit_m)
    score = 0

    if roe is not None:
        if roe >= 0.40:
            score += 10
        elif roe >= 0.20:
            score += 5
        elif roe <= 0.10:
            score -= 5

    if gross_m is not None:
        if gross_m >= 0.50:
            score += 5
        elif gross_m >= 0.35:
            score += 3
        elif gross_m <= 0.20:
            score -= 5

    if profit_m is not None:
        if profit_m >= 0.25:
            score += 5
        elif profit_m >= 0.15:
            score += 3
        elif profit_m <= 0.05:
            score -= 5

    return score


# ======================= Unified Decision Score =======================
# Mục tiêu: tạo DecisionScore ~ 0–100
# - ~50: trung tính (HOLD / WATCH)
# - >65: nên BUY
# - >80: STRONG BUY
# TechScore thường dao động khoảng [-40, +100]
# FundamentalScore (Val + Growth + Quality) thường khoảng [-40, +100]
def unified_decision_score(tech, val, growth, quality, upside, rsi, vol_ratio):
    # Chuẩn hoá các thành phần về biên an toàn
    tech_norm = max(min(tech, 80), -40)
    fundamental = max(min(val + growth + quality, 80), -40)

    upside_score = score_upside(upside)
    vol_score = score_volume(vol_ratio)
    rsi_score = score_rsi(rsi)

    # Trọng số:
    #  - 35% kỹ thuật (tech_norm)
    #  - 35% nền tảng (fundamental)
    #  - 20% upside theo analyst
    #  - 5% volume (xác nhận dòng tiền)
    #  - 5% RSI (thời điểm vào lệnh)
    decision_raw = (
        0.35 * tech_norm +
        0.35 * fundamental +
        0.20 * upside_score +
        0.05 * vol_score +
        0.05 * rsi_score
    )

    # Quy đổi về thang 0–100:
    # Giả định decision_raw ~ [-60, +100]
    #  -60  → 0
    #  +100 → 100
    raw_min, raw_max = -60.0, 100.0
    decision_clamped = max(min(decision_raw, raw_max), raw_min)
    decision_norm = (decision_clamped - raw_min) / (raw_max - raw_min) * 100.0

    return decision_norm


# ======================= Technical Signal (chỉ dùng TechScore) =======================
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


# ======================= Final Decision Signal (dùng DecisionScore 0–100) =======================
# DecisionScore:
#   >=80 → STRONG BUY
#   >=65 → BUY
#   >=50 → WATCH (giữ / theo dõi)
#   >=40 → SELL (cân nhắc thoát bớt)
#   <40  → STRONG SELL (tránh / thoát)
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
