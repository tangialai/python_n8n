"""
stock_analyzer

Package phân tích cổ phiếu trung & dài hạn dựa trên dữ liệu Yahoo Finance.

Luồng xử lý:
1. Lấy dữ liệu giá & thông tin cơ bản từ Yahoo Finance (Ticker.info).
2. Tính các chỉ báo kỹ thuật: SMA, EMA, RSI, MACD, ATR, 52-week range, RVOL...
3. Tính các chỉ số cơ bản: P/E, PEG, ROE, biên lợi nhuận, tăng trưởng, analyst target...
4. Chấm điểm dựa trên cấu hình trong config.py.
5. Trả về tổng điểm & tín hiệu: STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL.
"""

__version__ = "0.1.0"