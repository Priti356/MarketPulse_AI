"""
Tests for Ticker Intelligence: Position Extractor and Support/Resistance.
"""

import pytest
import pandas as pd
from src.intelligence.position_extractor import PositionExtractor
from src.intelligence.support_resistance import TechnicalAnalyzer


def test_position_extraction_explicit():
    extractor = PositionExtractor()

    # Buy
    assert extractor.extract_position("We recommend a strong buy rating on Tencent.")["position"] == "Buy"
    assert extractor.extract_position("強烈建議買入0700！")["position"] == "Buy"
    assert extractor.extract_position("삼성전자 적극 매수 추천합니다.")["position"] == "Buy"

    # Sell
    assert extractor.extract_position("Suggest sell position before next week earnings.")["position"] == "Sell"
    assert extractor.extract_position("建議沽出減持斬倉。")["position"] == "Sell"
    assert extractor.extract_position("비중 축소 전량 매도 권고.")["position"] == "Sell"

    # Hold
    assert extractor.extract_position("Maintaining hold position for now.")["position"] == "Hold"
    assert extractor.extract_position("堅決持有，靜待分拆。")["position"] == "Hold"

    # Not stated
    assert extractor.extract_position("Today is a sunny day in Hong Kong.")["position"] == "Not stated"


def test_support_resistance_pivots():
    analyzer = TechnicalAnalyzer()
    df = pd.DataFrame({
        "High": [105.0, 108.0, 110.0],
        "Low": [98.0, 101.0, 103.0],
        "Close": [102.0, 106.0, 108.0],
        "Volume": [1000, 1500, 2000]
    })
    levels = analyzer.calculate_support_resistance(df)
    assert levels["support"] < levels["resistance"]
    assert levels["support"] > 0
    assert levels["resistance"] > 0
    assert "pivot" in levels

