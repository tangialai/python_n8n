import yfinance as yf
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def get_stock(symbol: str):
    try:
        t = yf.Ticker(symbol)
        fast = t.fast_info
        info = t.get_info()

        price = fast.get("lastPrice")
        target_avg = info.get("targetMeanPrice")
        peg = info.get("trailingPegRatio")

        raw_upside = ((target_avg - price) / price) * 100 if price and target_avg and price > 0 else None
        upside = r_pct(raw_upside)

        return {
            "symbol": symbol,
            "company": info.get("shortName"),
            "price": r(price),
            "change": r(fast.get("regularMarketChange")),
            "changePercent": r_pct(fast.get("regularMarketChangePercent")),
            "dayHigh": r(fast.get("dayHigh")),
            "dayLow": r(fast.get("dayLow")),
            "targetLow": r(info.get("targetLowPrice")),
            "targetAvg": r(target_avg),
            "targetHigh": r(info.get("targetHighPrice")),
            "upside": upside,
            "recommendation": recommend(peg, upside),
            "yfRecommendation": format_yf_recommendation(info.get("recommendationKey")),
        }

    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

def main(symbols):
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(get_stock, s) for s in symbols]
        for f in as_completed(futures):
            results.append(f.result())

    print(json.dumps(results, ensure_ascii=False))

if __name__ == "__main__":
    # ✅ Nhận danh sách dạng: "AAPL,MSFT,NVDA,TSLA"
    raw = sys.argv[1]
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    main(symbols)
