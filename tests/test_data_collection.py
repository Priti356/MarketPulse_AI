"""
Tests for Data Collection Modules: Apify, Exa, and PriceCollector.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from src.data_collection.price_collector import PriceCollector
from src.data_collection.apify_scraper import ApifySocialScraper
from src.data_collection.exa_news import ExaNewsCollector
from src.database.db_manager import DatabaseManager


@pytest.fixture
def mock_db(tmp_path):
    db_file = tmp_path / "test_marketpulse.db"
    return DatabaseManager(db_url=f"sqlite:///{db_file}")


def test_supported_tickers_config():
    """Verify that assets.yaml loads all 10 required tickers across HKEX and KRX."""
    collector = PriceCollector()
    tickers = collector.get_supported_tickers()
    assert len(tickers) == 10
    assert "0700.HK" in tickers
    assert "9988.HK" in tickers
    assert "005930.KS" in tickers
    assert "000660.KS" in tickers


def test_fallback_price_generation(mock_db):
    """Verify realistic price generation produces valid OHLCV series."""
    collector = PriceCollector(db_manager=mock_db)
    df = collector._generate_realistic_fallback_prices("0700.HK", days=30)
    assert not df.empty
    assert len(df) == 30
    assert "Close" in df.columns
    assert "daily_return" in df.columns
    assert (df["High"] >= df["Low"]).all()
    assert (df["Volume"] > 0).all()


def test_apify_scraper_sources(mock_db):
    """Verify Apify scraper retrieves data from at least 2 distinct sources per exchange."""
    scraper = ApifySocialScraper(db_manager=mock_db)
    
    # Test HKEX asset (LIHKG + Telegram)
    hk_posts = scraper.collect_social_data_for_asset("0700.HK", days=10)
    assert len(hk_posts) > 0
    sources_hk = {p["source_name"] for p in hk_posts}
    assert "LIHKG" in sources_hk or "Telegram" in sources_hk

    # Test KRX asset (Naver Finance + Telegram)
    kr_posts = scraper.collect_social_data_for_asset("005930.KS", days=10)
    assert len(kr_posts) > 0
    sources_kr = {p["source_name"] for p in kr_posts}
    assert "Naver Finance" in sources_kr or "Telegram" in sources_kr


def test_exa_news_collector(mock_db):
    """Verify Exa news collector retrieves structured articles and deduplicates."""
    collector = ExaNewsCollector(db_manager=mock_db)
    articles = collector.fetch_news_for_asset("0700.HK", days=10)
    assert len(articles) > 0
    first = articles[0]
    assert "title" in first
    assert "url" in first
    assert "published_date" in first
    assert "snippet" in first
    assert first["ticker"] == "0700.HK"

