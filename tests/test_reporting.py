"""
Tests for Report Generation (CSV and PDF).
"""

import os
import pytest
from src.reporting.report_generator import ReportGenerator


def test_csv_and_pdf_generation(tmp_path):
    generator = ReportGenerator(reports_dir=str(tmp_path))

    sample_intel = [
        {
            "ticker": "0700.HK",
            "asset_name": "Tencent Holdings",
            "exchange": "HKEX",
            "latest_price": 415.0,
            "predicted_price": 425.0,
            "sentiment_direction": "Bullish",
            "investor_position": "Buy",
            "position_quote": "Strong cloud growth",
            "support_level": 405.0,
            "resistance_level": 430.0,
            "total_mentions": 120,
            "sentiment_score": 0.45,
            "pos_pct": 65.0,
            "neg_pct": 10.0,
            "neu_pct": 25.0,
        },
        {
            "ticker": "005930.KS",
            "asset_name": "Samsung Electronics",
            "exchange": "KRX",
            "latest_price": 58000.0,
            "predicted_price": 60500.0,
            "sentiment_direction": "Bullish",
            "investor_position": "Buy",
            "position_quote": "HBM3E memory surge",
            "support_level": 56000.0,
            "resistance_level": 62000.0,
            "total_mentions": 250,
            "sentiment_score": 0.52,
            "pos_pct": 70.0,
            "neg_pct": 10.0,
            "neu_pct": 20.0,
        }
    ]

    csv_file = generator.generate_csv_report(sample_intel)
    assert os.path.exists(csv_file)
    assert os.path.getsize(csv_file) > 100

    pdf_file = generator.generate_pdf_report(sample_intel)
    assert os.path.exists(pdf_file)
    assert os.path.getsize(pdf_file) > 1000

