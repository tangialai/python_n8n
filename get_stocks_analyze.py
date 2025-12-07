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

def recommend(peg, upside):
    if isinstance(peg, (int, float)) and peg < 1 and isinstance(upside, (int, float)) and upside > 20:
        return "BUY"
    elif isinstance(upside, (int, float)) and upside < 0:
        return "AVOID"
    else:
        return "HOLD"

def format_yf_recommendation(key):
    if not key:
        return "NONE"
    return key.replace("_", " ").upper()

# ====== Technical Indicators ======
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calc_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def calc_bbands(series, period=20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    return sma, upper, lower

def score_rsi(rsi):
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
    if macd < signal and hist < 0: return -10
    return 0

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

def score_bb(price, bb_mid, bb_low, bb_up):
    if None in (price, bb_mid, bb_low, bb_up): return 0
    score = 0
    if price > bb_mid: score += 5
    else: score -= 5

    if price <= bb_low: score += 10
    if price >= bb_up: score -= 10
    return score

def score_risk(atr14, price):
    if atr14 is None or price is None or price == 0: return 0
    risk = atr14 / price
    if risk < 0.01: return 10
    if risk < 0.02: return 5
    if risk < 0.03: return 0
    return -10

def score_volume(vr):
    if vr is None: return 0
    if vr > 1.5: return 5
    if vr > 1.0: return 3
    if vr < 0.7: return -5
    return 0

def score_upside(u):
    if u is None: return 0
    if u > 30: return 20
    if u > 20: return 15
    if u > 10: return 10
    if u > 0: return 5
    return -5

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

def get_stock(symbol: str):
    try:
        yf_t = yf.Ticker(symbol)
        info = yf_t.info

        df = yf_t.history(period="6mo")

        price = info.get("regularMarketPrice")
        target_avg = info.get("targetMeanPrice")
        peg = info.get("trailingPegRatio")

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

        return {
    "symbol": symbol,
    "company": info.get("shortName"),

    # === PRICE & CHANGE ===
    "price": round(price, 2) if price is not None else None,
    "change": r_pct(info.get("regularMarketChange")),
    "changePercent": r_pct(info.get("regularMarketChangePercent")),
    "dayHigh": round(info.get("regularMarketDayHigh"), 2) if info.get("regularMarketDayHigh") else None,
    "dayLow": round(info.get("regularMarketDayLow"), 2) if info.get("regularMarketDayLow") else None,

    # === TARGET PRICE ===
    "targetLow": round(info.get("targetLowPrice"), 2) if info.get("targetLowPrice") else None,
    "targetAvg": round(target_avg, 2) if target_avg else None,
    "targetHigh": round(info.get("targetHighPrice"), 2) if info.get("targetHighPrice") else None,
    "upside": round(upside, 2) if upside is not None else None,

    # === RECOMMENDATION ===
    "recommendation": recommend(peg, upside),

    # === FINAL SCORE ===
    "TechnicalScore": tech_score,
    "techSignal": get_signal(tech_score),

    # === TECHNICAL OUTPUT ===
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

