"""
data_fetcher.py

Nhiệm vụ:
- Lấy lịch sử giá (OHLCV) để tính chỉ báo kỹ thuật.
- Lấy snapshot thông tin cơ bản & giá hiện tại từ Yahoo Finance
  thông qua yf.Ticker(symbol).info (yf.info).

Thư viện: yfinance
"""

from typing import Dict, Any

import yfinance as yf
import pandas as pd

from .utils import safe_float


def fetch_price_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Lấy lịch sử giá.

    period: thời gian look-back (ví dụ: "6mo", "1y", "2y"...).
    interval: "1d" để phân tích daily cho trung/dài hạn.

    Trả về DataFrame có các cột: Open, High, Low, Close, Volume.
    """
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    df = df.sort_index()
    return df


def fetch_snapshot(symbol: str) -> Dict[str, Any]:
    """
    Lấy snapshot thông tin hiện tại của mã cổ phiếu từ Yahoo Finance.

    Sử dụng:
    - ticker.info (yf.info): dict chứa rất nhiều trường cơ bản & định giá.

    Các trường quan trọng được trích sẵn:
    - price, dayHigh, dayLow, previousClose, volume, marketCap, ...
    - cùng với toàn bộ dict gốc trong key "info_raw" để sau này tái sử dụng.
    """
    ticker = yf.Ticker(symbol)

    try:
        info_raw: Dict[str, Any] = ticker.info  # yf.info
    except Exception:
        # Trong một số bản yfinance mới .info có thể lỗi,
        # khi đó có thể dùng ticker.get_info() thay thế.
        try:
            info_raw = ticker.get_info()
        except Exception:
            info_raw = {}
    company = info_raw.get("shortName")
    change = round2(safe_float(info_raw.get("regularMarketChange")))
    change_pct = round2(safe_float(info_raw.get("regularMarketChangePercent")))
    price = safe_float(info_raw.get("currentPrice") or info_raw.get("regularMarketPrice"))
    day_high = safe_float(info_raw.get("dayHigh"))
    day_low = safe_float(info_raw.get("dayLow"))
    previous_close = safe_float(
        info_raw.get("previousClose") or info_raw.get("regularMarketPreviousClose")
    )
    volume = safe_float(info_raw.get("volume"))
    market_cap = safe_float(info_raw.get("marketCap"))
    avg_volume_10d = safe_float(info_raw.get("averageDailyVolume10Day"))
    avg_volume_3m = safe_float(info_raw.get("averageVolume"))
    avg_50days = safe_float(info_raw.get("fiftyDayAverage"))

    snapshot: Dict[str, Any] = {
        "symbol": symbol,
        "company": company,
        "change": change,
        "change_pct": change_pct,
        "price": price,
        "day_high": day_high,
        "day_low": day_low,
        "previous_close": previous_close,
        "volume": volume,
        "market_cap": market_cap,
        "avg_volume_10d": avg_volume_10d,
        "avg_volume_3m": avg_volume_3m,
        "avg_50days": avg_50days,
        "info_raw": info_raw,
    }
    return snapshot
