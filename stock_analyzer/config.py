"""
config.py

TẤT CẢ CÁC THAM SỐ CHẤM ĐIỂM NÊN THAY ĐỔI Ở ĐÂY.

Mục tiêu:
- Khi muốn chỉnh chiến lược (ví dụ tăng trọng số fundamentals,
  thay đổi ngưỡng RSI, PE, Upside...) chỉ cần sửa file này.
- Các module khác (indicators, scoring, analyzer) hầu như không phải đổi.
"""

# -----------------------------
# 1. Trọng số tổng cho từng nhóm
# -----------------------------

# Tổng điểm kỹ thuật tối đa
TECHNICAL_MAX = 6.0

# Tổng điểm cơ bản tối đa
FUNDAMENTAL_MAX = 4.0

# -----------------------------
# 2. Ngưỡng quyết định mua / bán
# -----------------------------

DECISION_THRESHOLDS = {
    # Điểm >= 8: rất đẹp cho trung/dài hạn
    "STRONG_BUY": 8.0,
    # 6–<8: có thể giải ngân dần
    "BUY": 6.0,
    # 4–<6: nắm giữ / chờ thêm tín hiệu
    "HOLD": 4.0,
    # 2–<4: nên giảm tỷ trọng
    "SELL": 2.0,
    # <2: tránh xa / thoát dần
    "STRONG_SELL": 0.0,
}

# -----------------------------
# 3. Tham số cho RSI / ATR
# -----------------------------

# Vùng RSI cho trung & dài hạn
RSI_OVERSOLD = 35    # dưới mức này thường là "hơi rẻ"
RSI_OVERBOUGHT = 70  # trên mức này là "quá mua"

# Biên độ biến động ATR% đánh giá rủi ro
ATR_LOW = 1.0        # ATR% < 1%: biến động rất thấp
ATR_HIGH = 5.0       # ATR% > 5%: biến động cao

# -----------------------------
# 4. Chuẩn hóa Upside
# -----------------------------

# Giới hạn tối đa Upside (để scale 0–1)
UPSIDE_CAP = 100.0   # 100% upside = điểm tối đa cho Upside

# -----------------------------
# 5. Các rule đơn giản cho định giá
# -----------------------------

PE_RULES = [
    # (min, max, score)
    (10, 25, 0.8),   # vùng hợp lý
    (0, 10, 0.4),    # rẻ (nhưng có thể là value trap)
    (25, 40, 0.3),   # hơi cao
    (40, 9999, -0.3) # quá cao, trừ điểm
]

PEG_RULES = [
    (0.5, 1.5, 0.7),  # PEG hợp lý
    (0.0, 0.5, 0.4),  # PEG quá thấp
    (1.5, 2.0, 0.2),  # hơi cao
    (2.0, 9999, -0.2)
]

ROE_LEVELS = [(0.15, 0.8), (0.10, 0.5)]
MARGIN_LEVELS = [(0.15, 0.7), (0.05, 0.3)]

GROWTH_LEVELS = [
    (0.15, 0.7),  # tăng trưởng tốt
    (0.05, 0.4),  # tăng trưởng vừa phải
]

# Điểm thưởng theo đánh giá analyst recommendationKey
ANALYST_RECO_POINTS = {
    "strong_buy": 0.6,
    "buy": 0.4,
    "hold": 0.1,
    "sell": -0.4,
    "underperform": -0.4,
}