"""
cli.py

Dùng để chạy từ command line hoặc trong n8n.

Ví dụ CLI:
    python3 -m stock_analyzer AAPL,MSFT,NVDA

Trong n8n (Execute Command node):
    Command: python3 -m stock_analyzer {{$json["symbols"]}}

Output: JSON chứa danh sách kết quả cho từng symbol.
"""

import sys
import json
from typing import List

from .analyzer import analyze_many


def _parse_symbols(argv: List[str]) -> List[str]:
    if len(argv) < 2:
        print("Usage: python3 -m stock_analyzer <symbol1,symbol2,...>")
        sys.exit(1)

    raw = argv[1]
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not symbols:
        print("No symbol provided.")
        sys.exit(1)
    return symbols


def main() -> None:
    symbols = _parse_symbols(sys.argv)
    data = analyze_many(symbols)
    # In JSON ra stdout để n8n đọc được
    print(json.dumps(data, indent=2, ensure_ascii=False))