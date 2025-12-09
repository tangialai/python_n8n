"""
fundamentals.py

Trích các CHỈ SỐ CƠ BẢN (fundamentals) từ ticker.info:

- Định giá:
  + trailingPE, forwardPE, pegRatio

- Chất lượng doanh nghiệp:
  + returnOnEquity (ROE), profitMargins

- Tăng trưởng:
  + revenueGrowth, earningsGrowth

- Analyst:
  + targetMeanPrice, targetHighPrice, targetLowPrice, recommendationKey

Sau đó tính thêm:
- upside_pct: % tăng từ giá hiện tại lên targetMeanPrice.
"""

from typing import Dict, Any

from .utils import safe_float, percent_change


def extract_fundamentals(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    info = snapshot.get("info_raw") or {}

    trailing_pe = safe_float(info.get("trailingPE"))
    forward_pe = safe_float(info.get("forwardPE"))
    peg_ratio = safe_float(info.get("pegRatio"))

    roe = safe_float(info.get("returnOnEquity"))
    profit_margin = safe_float(info.get("profitMargins"))

    revenue_growth = safe_float(info.get("revenueGrowth"))
    earnings_growth = safe_float(info.get("earningsGrowth"))

    target_mean = safe_float(info.get("targetMeanPrice"))
    target_high = safe_float(info.get("targetHighPrice"))
    target_low = safe_float(info.get("targetLowPrice"))
    recommendation_key = info.get("recommendationKey")

    price = snapshot.get("price")
    upside_pct = percent_change(target_mean, price) if price and target_mean else None

    return {
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "peg_ratio": peg_ratio,
        "roe": roe,
        "profit_margin": profit_margin,
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "target_mean": target_mean,
        "target_high": target_high,
        "target_low": target_low,
        "upside_pct": upside_pct,
        "recommendation_key": recommendation_key,
    }