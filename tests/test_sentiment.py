"""
Tests for Financial Sentiment Models and Engagement Weighting.
"""

import pytest
from src.sentiment.finbert_model import FinancialSentimentModel
from src.sentiment.sentiment_analyzer import SentimentAnalyzer
from src.database.db_manager import DatabaseManager


@pytest.fixture
def mock_db(tmp_path):
    db_file = tmp_path / "test_sent.db"
    return DatabaseManager(db_url=f"sqlite:///{db_file}")


def test_language_detection():
    model = FinancialSentimentModel(use_hf_if_available=False)
    assert model.detect_language("Tencent beats quarterly earnings estimates with strong cloud growth") == "en"
    assert model.detect_language("騰訊業績大好，建議買入加倉！") == "zh"
    assert model.detect_language("삼성전자 5만전자 반도체 턴어라운드 강력 매수 추천") == "ko"


def test_english_sentiment_scoring():
    model = FinancialSentimentModel(use_hf_if_available=False)
    pos_res = model.predict("Tencent surged on record profits and dividend expansion outperforming targets.")
    assert pos_res["sentiment_score"] > 0
    assert pos_res["sentiment_label"] == "Positive"
    assert pos_res["prob_positive"] > pos_res["prob_negative"]

    neg_res = model.predict("Company reports heavy deficit and earnings plunge with severe regulatory probe.")
    assert neg_res["sentiment_score"] < 0
    assert neg_res["sentiment_label"] == "Negative"


def test_chinese_traditional_sentiment():
    model = FinancialSentimentModel(use_hf_if_available=False)
    pos_res = model.predict("阿里雲業務增長強勁，南向資金淨流入，強烈建議買入！")
    assert pos_res["sentiment_score"] > 0
    assert pos_res["sentiment_label"] == "Positive"

    neg_res = model.predict("大市疲軟，高位見頂，面臨龐大沽壓，建議減持沽出斬倉。")
    assert neg_res["sentiment_score"] < 0
    assert neg_res["sentiment_label"] == "Negative"


def test_korean_sentiment():
    model = FinancialSentimentModel(use_hf_if_available=False)
    pos_res = model.predict("SK하이닉스 HBM 영업이익 사상 최고치 경신! 강력 매수 추천.")
    assert pos_res["sentiment_score"] > 0
    assert pos_res["sentiment_label"] == "Positive"

    neg_res = model.predict("실적 부진 및 어닝 쇼크 우려로 단기 급락 위험, 비중 축소 매도 권고.")
    assert neg_res["sentiment_score"] < 0
    assert neg_res["sentiment_label"] == "Negative"


def test_engagement_weighting(mock_db):
    analyzer = SentimentAnalyzer(db_manager=mock_db)
    base_weight = analyzer.calculate_engagement_weight(likes=0, reposts=0, views=0, followers=0)
    assert base_weight == 1.0

    viral_weight = analyzer.calculate_engagement_weight(likes=500, reposts=100, views=50000, followers=20000)
    assert 1.0 < viral_weight <= 4.0

