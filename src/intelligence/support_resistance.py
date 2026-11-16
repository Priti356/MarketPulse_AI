"""
Support, Resistance, and Technical Market Direction Calculator.
Employs standard Classical Floor Pivots, 20-day Rolling Extrema, and Volume-Weighted clusters.
"""

import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Calculates verifiable support/resistance levels and market trend indicators."""

    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Derive support and resistance from recent OHLCV price action.
        Uses Classical Floor Pivots and 20-day Rolling Extrema.
        """
        if df.empty:
            return {
                "support": 0.0,
                "resistance": 0.0,
                "pivot": 0.0,
                "s1": 0.0,
                "r1": 0.0,
                "s2": 0.0,
                "r2": 0.0,
                "rolling_min_20d": 0.0,
                "rolling_max_20d": 0.0,
                "method": "Insufficient data"
            }

        # Latest completed trading bar
        last_bar = df.iloc[-1]
        high = float(last_bar["High"])
        low = float(last_bar["Low"])
        close = float(last_bar["Close"])

        # Classical Floor Pivot calculation
        pivot = (high + low + close) / 3.0
        r1 = (2.0 * pivot) - low
        s1 = (2.0 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)

        # 20-day Rolling Extrema
        window = min(20, len(df))
        rolling_slice = df.iloc[-window:]
        rolling_max = float(rolling_slice["High"].max())
        rolling_min = float(rolling_slice["Low"].min())

        # Synthesize primary support & resistance
        # Support is anchored by the closest significant structural floor below close
        primary_support = round(s1 if s1 < close else (rolling_min if rolling_min < close else close * 0.98), 2)
        primary_resistance = round(r1 if r1 > close else (rolling_max if rolling_max > close else close * 1.02), 2)

        # Round values for currency appropriateness
        is_krw = close > 1000.0
        decimals = 0 if is_krw else 2

        return {
            "support": round(primary_support, decimals),
            "resistance": round(primary_resistance, decimals),
            "pivot": round(pivot, decimals),
            "s1": round(s1, decimals),
            "r1": round(r1, decimals),
            "s2": round(s2, decimals),
            "r2": round(r2, decimals),
            "rolling_min_20d": round(rolling_min, decimals),
            "rolling_max_20d": round(rolling_max, decimals),
            "method": "Classical Floor Pivots & 20D Extrema"
        }

    @staticmethod
    def determine_market_direction(df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        """
        Evaluate trend direction using EMA(10), EMA(30), and RSI(14).
        Classifies into 'Bullish', 'Bearish', or 'Neutral / Consolidating'.
        """
        if df.empty or len(df) < 15:
            return "Neutral", {"rsi": 50.0, "ema_10": 0.0, "ema_30": 0.0}

        close = df["Close"]
        ema_10 = close.ewm(span=10, adjust=False).mean().iloc[-1]
        ema_30 = close.ewm(span=30, adjust=False).mean().iloc[-1]
        current_close = close.iloc[-1]

        # Calculate standard RSI (14 periods)
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(window=14, min_periods=5).mean().iloc[-1]
        avg_loss = loss.rolling(window=14, min_periods=5).mean().iloc[-1]

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = round(float(100.0 - (100.0 / (1.0 + rs))), 2)

        metrics = {
            "rsi": rsi,
            "ema_10": round(float(ema_10), 2),
            "ema_30": round(float(ema_30), 2),
            "current_close": round(float(current_close), 2)
        }

        # Trend Decision Logic
        if current_close > ema_10 and ema_10 >= ema_30 and rsi >= 48.0:
            direction = "Bullish"
        elif current_close < ema_10 and ema_10 <= ema_30 and rsi <= 52.0:
            direction = "Bearish"
        else:
            direction = "Neutral"

        return direction, metrics
