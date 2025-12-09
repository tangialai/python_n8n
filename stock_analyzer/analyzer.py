"""
analyzer.py

Đầu não của package:

- Gọi Yahoo Finance để lấy dữ liệu.
- Tính chỉ báo kỹ thuật.
- Trích chỉ số cơ bản.
- Chấm điểm và đưa ra quyết định.

Hàm public:
- analyze_single_stock(symbol)
- analyze_many(symbols)
"""

from typing import Dict, Any, List

from .data_fetcher import fetch_price_history, fetch_snapshot
from .indicators import (
    compute_trend_indicators,
    compute_momentum_indicators,
    compute_volume_indicators,
    compute_volatility_indicators,
)
from .fundamentals import extract_fundamentals
from .scoring import score_technical, score_fundamentals, make_decision


def analyze_single_stock(symbol: str) -> Dict[str, Any]:
    """
    Phân tích ĐẦY ĐỦ 1 mã cổ phiếu.

    Trả về dict gồm:
    - snapshot: thông tin giá hiện tại + raw info từ Yahoo.
    - technical: indicators kỹ thuật (trend, momentum, volume, volatility).
    - fundamentals: chỉ số cơ bản.
    - scoring: điểm kỹ thuật, cơ bản, tổng & quyết định.
    """
    try:
        snapshot = fetch_snapshot(symbol)
        price = snapshot.get("price")
        volume = snapshot.get("volume")

        df = fetch_price_history(symbol, period="1y", interval="1d")
        if df.empty or price is None:
            return {"symbol": symbol, "error": "No price data from Yahoo Finance."}

        # ----- Tính indicators -----
        trend = compute_trend_indicators(df, price=price)
        momentum = compute_momentum_indicators(df)
        vol_ind = compute_volatility_indicators(df, price=price)
        vol_flow = compute_volume_indicators(df, current_volume=volume)

        # ----- Fundamentals -----
        fundamentals = extract_fundamentals(snapshot)

        # ----- Scoring -----
        tech_score = score_technical(snapshot, trend, momentum, vol_flow, vol_ind)
        fund_score = score_fundamentals(fundamentals)
        total_score = max(0.0, tech_score["total"] + fund_score["total"])
        decision = make_decision(total_score)

        return {
            "symbol": symbol,
            "snapshot": snapshot,
            "technical": {
                "trend": trend,
                "momentum": momentum,
                "volume": vol_flow,
                "volatility": vol_ind,
            },
            "fundamentals": fundamentals,
            "scoring": {
                "technical": tech_score,
                "fundamental": fund_score,
                "total_score": total_score,
                "decision": decision,
            },
        }

    except Exception as exc:
        return {"symbol": symbol, "error": str(exc)}


def analyze_many(symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Phân tích nhiều mã cùng lúc.

    symbols: list ["AAPL", "MSFT", ...]
    """
    results: List[Dict[str, Any]] = []
    for s in symbols:
        s_clean = s.strip().upper()
        if not s_clean:
            continue
        results.append(analyze_single_stock(s_clean))
    return results