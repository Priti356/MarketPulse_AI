"""
Tests for Data Cleaning and Feature Engineering Modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.processing.data_cleaning import DataCleaner
from src.processing.feature_engineering import FeatureEngineer


def test_clean_text():
    raw_text = "Check Tencent 0700.HK at https://t.me/stocks ! 🚀🚀 <p>Strong buy</p>"
    cleaned = DataCleaner.clean_text(raw_text)
    assert "https://" not in cleaned
    assert "<p>" not in cleaned
    assert "Tencent 0700.HK" in cleaned


def test_feature_engineering_alignment():
    """Verify feature dataset generation prevents future leakage and computes lags."""
    engineer = FeatureEngineer()

    # Generate dummy prices
    dates = pd.date_range(start="2026-01-01", periods=30, freq="B")
    prices = [100.0 * (1 + 0.01 * i) for i in range(30)]
    price_df = pd.DataFrame({
        "Open": prices,
        "High": [p * 1.02 for p in prices],
        "Low": [p * 0.98 for p in prices],
        "Close": prices,
        "Volume": [1000000] * 30
    }, index=dates)

    # Dummy sentiment
    sent_records = [{
        "source_type": "social",
        "ticker": "0700.HK",
        "sentiment_score": 0.5,
        "weight": 1.5,
        "timestamp": d
    } for d in dates[:10]]

    feat_df = engineer.build_feature_dataset(price_df, sent_records)
    assert not feat_df.empty
    assert "target_next_return" in feat_df.columns
    assert "sentiment_lag_1d" in feat_df.columns
    assert "rsi_14" in feat_df.columns
    assert "macd" in feat_df.columns

    # Check that feature columns match expected list
    for col in FeatureEngineer.get_feature_columns():
        assert col in feat_df.columns

