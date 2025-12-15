"""
utils.py

Các hàm tiện ích dùng chung trong toàn bộ package.
"""

from typing import Optional


def safe_float(value):
    """
    Ép kiểu float an toàn.
    - Nếu value là pandas Series 1 phần tử → lấy giá trị đầu.
    - Nếu None hoặc lỗi → trả về None.
    """
    try:
        if value is None:
            return None

        # Nếu là Series (pandas) thì lấy phần tử đầu tiên
        if hasattr(value, "iloc"):
            if len(value) == 0:
                return None
            return float(value.iloc[0])

        return float(value)

    except Exception:
        return None

def round2(v):
    return round(v, 2) if isinstance(v, (int, float)) else None

def percent_change(current: float, base: float) -> Optional[float]:
    """
    Tính phần trăm thay đổi: (current - base) / base * 100.
    Dùng để tính % tăng từ giá hiện tại lên target price.
    """
    if current is None or base is None:
        return None
    if base == 0:
        return None
    return (current - base) / base * 100.0


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Giới hạn value trong [min_value, max_value]."""
    return max(min_value, min(max_value, value))
