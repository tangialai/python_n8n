"""
scoring.py

ĐỊNH NGHĨA CÁCH CHẤM ĐIỂM & QUYẾT ĐỊNH.

Ý tưởng:
- Điểm tổng ~ 10.
- 6 điểm cho KỸ THUẬT (Trend + Momentum + Volume + Volatility).
- 4 điểm cho CƠ BẢN (Định giá + Chất lượng + Tăng trưởng + Upside + Analyst).

Toàn bộ ngưỡng & trọng số lấy từ config.py để dễ chỉnh.
"""

from typing import Dict, Any

from .config import (
    TECHNICAL_MAX,
    FUNDAMENTAL_MAX,
    DECISION_THRESHOLDS,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    ATR_LOW,
    ATR_HIGH,
    UPSIDE_CAP,
    PE_RULES,
    PEG_RULES,
    ROE_LEVELS,
    MARGIN_LEVELS,
    GROWTH_LEVELS,
    ANALYST_RECO_POINTS,
)
from .utils import clamp


def _apply_range_rules(value, rules):
    """
    Hàm tiện ích: cho value & danh sách rules (min, max, score),
    trả về tổng điểm.
    """
    if value is None:
        return 0.0
    total = 0.0
    for min_v, max_v, score in rules:
        if min_v <= value < max_v:
            total += score
    return total


def score_technical(
    snapshot: Dict[str, Any],
    trend: Dict[str, Any],
    momentum: Dict[str, Any],
    volume: Dict[str, Any],
    vol: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Chấm điểm KỸ THUẬT.

    Chi tiết:
    - Trend: SMA50, SMA200, 52w position.
    - Momentum: RSI, MACD.
    - Volume: RVOL.
    - Volatility: ATR%.
    """
    price = snapshot.get("price")

    score_trend = 0.0
    score_mom = 0.0
    score_vol = 0.0
    score_volatility = 0.0

    # ===== Trend =====
    sma50 = trend.get("sma50")
    sma200 = trend.get("sma200")
    pos_52w = trend.get("pos_52w")

    if price and sma50 and price > sma50:
        score_trend += 1.0
    if price and sma200 and price > sma200:
        score_trend += 1.5

    if pos_52w is not None:
        # giá nằm trong khoảng 30%–80% dải 52w: không quá đỉnh, không quá đáy
        if 0.3 <= pos_52w <= 0.8:
            score_trend += 0.5
        elif pos_52w < 0.1:
            # sát đáy 52w: có thể là cơ hội (nhưng rủi ro) → cộng nhẹ
            score_trend += 0.2

    # ===== Momentum =====
    rsi = momentum.get("rsi")
    macd = momentum.get("macd")
    macd_signal = momentum.get("macd_signal")

    if rsi:
        if RSI_OVERSOLD - 10 <= rsi <= RSI_OVERSOLD:
            # gần vùng quá bán → giá "hơi rẻ"
            score_mom += 0.5
        if RSI_OVERSOLD < rsi < RSI_OVERBOUGHT:
            # vùng khỏe & ổn định
            score_mom += 0.5
        if rsi > RSI_OVERBOUGHT + 5:
            # quá mua rõ rệt → trừ điểm
            score_mom -= 0.5

    if macd is not None and macd_signal is not None:
        if macd > macd_signal and macd > 0:
            score_mom += 1.0
        elif macd < macd_signal and macd < 0:
            score_mom -= 0.5

    # ===== Volume =====
    rvol = volume.get("rvol")
    if rvol:
        if rvol >= 1.5:
            score_vol += 1.0
        elif 1.0 <= rvol < 1.5:
            score_vol += 0.5
        elif rvol < 0.7:
            score_vol -= 0.2

    # ===== Volatility (ATR%) =====
    atr_pct = vol.get("atr_pct")
    if atr_pct:
        if ATR_LOW <= atr_pct <= ATR_HIGH:
            score_volatility += 0.5
        elif atr_pct > ATR_HIGH * 1.5:
            score_volatility -= 0.5

    raw_total = score_trend + score_mom + score_vol + score_volatility
    total = clamp(raw_total, -TECHNICAL_MAX, TECHNICAL_MAX)

    return {
        "total": total,
        "details": {
            "trend": score_trend,
            "momentum": score_mom,
            "volume": score_vol,
            "volatility": score_volatility,
        },
    }


def score_fundamentals(fund: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chấm điểm CƠ BẢN:
    - Định giá: PE, PEG.
    - Chất lượng: ROE, lợi nhuận biên.
    - Tăng trưởng: doanh thu, lợi nhuận.
    - Upside: analyst target vs giá hiện tại.
    - Analyst: recommendationKey.
    """
    score_valuation = 0.0
    score_quality = 0.0
    score_growth = 0.0
    score_upside = 0.0
    score_analyst = 0.0

    pe = fund.get("forward_pe") or fund.get("trailing_pe")
    peg = fund.get("peg_ratio")
    roe = fund.get("roe")
    margin = fund.get("profit_margin")
    rev_g = fund.get("revenue_growth")
    earn_g = fund.get("earnings_growth")
    upside = fund.get("upside_pct")
    reco = (fund.get("recommendation_key") or "").lower()

    # ===== Định giá (PE, PEG) =====
    score_valuation += _apply_range_rules(pe, PE_RULES)
    score_valuation += _apply_range_rules(peg, PEG_RULES)

    # ===== Chất lượng (ROE, Biên LN) =====
    if roe:
        for level, pts in ROE_LEVELS:
            if roe >= level:
                score_quality += pts
                break

    if margin:
        for level, pts in MARGIN_LEVELS:
            if margin >= level:
                score_quality += pts
                break

    # ===== Tăng trưởng =====
    if rev_g:
        for level, pts in GROWTH_LEVELS:
            if rev_g >= level:
                score_growth += pts
                break

    if earn_g:
        for level, pts in GROWTH_LEVELS:
            if earn_g >= level:
                score_growth += pts
                break

    # ===== Upside =====
    if upside:
        from_cap = clamp(upside, 0.0, UPSIDE_CAP)
        score_upside += (from_cap / UPSIDE_CAP) * 0.8  # tối đa 0.8 điểm

    # ===== Analyst =====
    for key, pts in ANALYST_RECO_POINTS.items():
        if key in reco:
            score_analyst += pts
            break

    raw_total = (
        score_valuation + score_quality + score_growth + score_upside + score_analyst
    )
    total = clamp(raw_total, -FUNDAMENTAL_MAX, FUNDAMENTAL_MAX)

    return {
        "total": total,
        "details": {
            "valuation": score_valuation,
            "quality": score_quality,
            "growth": score_growth,
            "upside": score_upside,
            "analyst": score_analyst,
        },
    }


def make_decision(total_score: float) -> str:
    """
    Chuyển tổng điểm (technical + fundamental) thành tín hiệu:

    - STRONG_BUY
    - BUY
    - HOLD
    - SELL
    - STRONG_SELL
    """
    if total_score >= DECISION_THRESHOLDS["STRONG_BUY"]:
        return "STRONG_BUY"
    if total_score >= DECISION_THRESHOLDS["BUY"]:
        return "BUY"
    if total_score >= DECISION_THRESHOLDS["HOLD"]:
        return "HOLD"
    if total_score >= DECISION_THRESHOLDS["SELL"]:
        return "SELL"
    return "STRONG_SELL"