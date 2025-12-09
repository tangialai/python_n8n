"""
indicators.py

Các hàm tính CHỈ BÁO KỸ THUẬT từ chuỗi giá lịch sử:

- Trend (xu hướng):
  + SMA50, SMA200: trung & dài hạn.
  + Vị trí giá trong 52-week range: gần đáy hay gần đỉnh.

- Momentum (động lượng):
  + RSI(14): sức mạnh xu hướng, tránh mua lúc quá mua.
  + MACD 12–26–9: xu hướng trung hạn & điểm đảo chiều.

- Volume (dòng tiền):
  + Volume trung bình 20 phiên.
  + RVOL: Volume hiện tại / Volume TB 20 phiên.

- Volatility (biến động):
  + ATR(14): biên độ dao động trung bình.
  + ATR% = ATR / Price * 100.
"""

from typing import Dict, Any

import numpy as np
import pandas as pd

from .utils import safe_float


def compute_trend_indicators(df: pd.DataFrame, price: float) -> Dict[str, Any]:
    closes = df["Close"]

    sma50 = closes.rolling(50).mean()
    sma200 = closes.rolling(200).mean()

    sma50_last = safe_float(sma50.iloc[-1]) if len(sma50) >= 50 else None
    sma200_last = safe_float(sma200.iloc[-1]) if len(sma200) >= 200 else None

    # 52-week range ~ 252 phiên giao dịch
    lookback = min(252, len(closes))
    last_window = closes.tail(lookback)
    high_52w = safe_float(last_window.max())
    low_52w = safe_float(last_window.min())

    if high_52w is not None and low_52w is not None and high_52w != low_52w and price:
        pos_52w = (price - low_52w) / (high_52w - low_52w)  # 0 = sát đáy, 1 = sát đỉnh
    else:
        pos_52w = None

    return {
        "sma50": sma50_last,
        "sma200": sma200_last,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pos_52w": pos_52w,
    }


def compute_momentum_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    closes = df["Close"]

    # ===== RSI 14 ngày =====
    delta = closes.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi_last = safe_float(rsi.iloc[-1])

    # ===== MACD 12–26–9 =====
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal

    macd_last = safe_float(macd.iloc[-1])
    signal_last = safe_float(signal.iloc[-1])
    hist_last = safe_float(hist.iloc[-1])

    return {
        "rsi": rsi_last,
        "macd": macd_last,
        "macd_signal": signal_last,
        "macd_hist": hist_last,
    }


def compute_volume_indicators(df: pd.DataFrame, current_volume: float) -> Dict[str, Any]:
    vol = df["Volume"]
    avg_vol_20 = safe_float(vol.tail(20).mean()) if len(vol) >= 5 else None

    if avg_vol_20 and current_volume:
        rvol = current_volume / avg_vol_20
    else:
        rvol = None

    return {
        "avg_volume_20": avg_vol_20,
        "rvol": rvol,
    }


def compute_volatility_indicators(df: pd.DataFrame, price: float) -> Dict[str, Any]:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()
    atr_last = safe_float(atr.iloc[-1])

    if atr_last is not None and price:
        atr_pct = (atr_last / price) * 100.0
    else:
        atr_pct = None

    return {
        "atr14": atr_last,
        "atr_pct": atr_pct,
    }