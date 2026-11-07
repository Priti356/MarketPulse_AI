"""
MarketPulse AI - End-to-End Pipeline Runner.
Executes data collection, multilingual FinBERT sentiment analysis,
technical feature engineering, ML model training, and executive reporting for all 10 assets.
"""

import os
import sys
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MarketPulsePipeline")

from src.database.db_manager import DatabaseManager
from src.data_collection.price_collector import PriceCollector
from src.data_collection.apify_scraper import ApifySocialScraper
from src.data_collection.exa_news import ExaNewsCollector
from src.sentiment.sentiment_analyzer import SentimentAnalyzer
from src.intelligence.position_extractor import PositionExtractor
from src.intelligence.support_resistance import TechnicalAnalyzer
from src.processing.feature_engineering import FeatureEngineer
from src.prediction.predict_price import PricePredictor
from src.reporting.report_generator import ReportGenerator


def run_full_pipeline():
    logger.info("=" * 70)
    logger.info("MARKETPULSE AI: EXECUTING END-TO-END INTELLIGENCE PIPELINE")
    logger.info("=" * 70)

    db = DatabaseManager()
    price_collector = PriceCollector(db_manager=db)
    social_scraper = ApifySocialScraper(db_manager=db)
    news_collector = ExaNewsCollector(db_manager=db)
    sentiment_analyzer = SentimentAnalyzer(db_manager=db)
    position_extractor = PositionExtractor()
    technical_analyzer = TechnicalAnalyzer()
    feature_engineer = FeatureEngineer()
    price_predictor = PricePredictor(db_manager=db)
    report_generator = ReportGenerator()

    tickers = price_collector.get_supported_tickers()
    logger.info(f"Loaded {len(tickers)} assets from configuration: {tickers}")

    for idx, ticker in enumerate(tickers, 1):
        meta = price_collector.tickers_meta[ticker]
        name = meta["name"]
        exchange = meta["exchange"]
        logger.info(f"\n--- [{idx}/{len(tickers)}] Processing {ticker} ({name} - {exchange}) ---")

        # 1. Historical Prices
        logger.info(f"1. Fetching historical prices for {ticker}...")
        price_df = price_collector.fetch_historical_prices(ticker, period="6mo", force_refresh=True)
        latest_price = float(price_df["Close"].iloc[-1])
        logger.info(f"   Latest Close Price: {latest_price:,.2f} {meta['currency']}")

        # 2. Social Discussions (at least 2 distinct sources per exchange)
        logger.info(f"2. Collecting social discussions across {exchange} sources...")
        social_posts = social_scraper.collect_social_data_for_asset(ticker, days=10)
        logger.info(f"   Collected {len(social_posts)} posts from {set(p['source_name'] for p in social_posts)}")

        # 3. Financial News (Exa API)
        logger.info(f"3. Collecting financial news articles...")
        news_articles = news_collector.fetch_news_for_asset(ticker, days=10)
        logger.info(f"   Retrieved {len(news_articles)} news items")

        # 4. Multilingual Sentiment Scoring & Engagement Weighting
        logger.info(f"4. Analyzing sentiment with language-aware FinBERT engine...")
        scored_social = sentiment_analyzer.analyze_social_posts(social_posts)
        scored_news = sentiment_analyzer.analyze_news_articles(news_articles)
        sent_agg = sentiment_analyzer.aggregate_ticker_sentiment(scored_social, scored_news)
        logger.info(f"   Sentiment Score: {sent_agg['sentiment_score']:+.3f} | Direction: {sent_agg['sentiment_direction']}")
        logger.info(f"   Breakdown: +{sent_agg['pos_pct']}% / ={sent_agg['neu_pct']}% / -{sent_agg['neg_pct']}%")

        # 5. Position & Technical Intelligence Extraction
        logger.info(f"5. Extracting investor positioning and support/resistance...")
        investor_pos, pos_quote = position_extractor.aggregate_asset_position(social_posts)
        tech_levels = technical_analyzer.calculate_support_resistance(price_df)
        market_dir, _ = technical_analyzer.determine_market_direction(price_df)
        logger.info(f"   Investor Position: '{investor_pos}' (Quote: '{pos_quote[:50]}...')")
        logger.info(f"   Support S1: {tech_levels['support']:,.2f} | Resistance R1: {tech_levels['resistance']:,.2f}")

        # 6. Feature Engineering & Machine Learning Prediction
        logger.info(f"6. Building temporal features and training ML prediction models...")
        all_sentiment = scored_social + scored_news
        feature_df = feature_engineer.build_feature_dataset(price_df, all_sentiment)
        pred_res = price_predictor.predict_next_price(ticker, feature_df)
        pred_price = pred_res["predicted_price"]
        exp_pct = pred_res["expected_change_pct"]
        logger.info(f"   Optimal ML Model: {pred_res['model_name']}")
        logger.info(f"   Forecast: Current={latest_price:,.2f} -> Next Session={pred_price:,.2f} ({exp_pct:+.2f}%)")
        logger.info(f"   Test Directional Accuracy: {pred_res['test_metrics'].get('directional_accuracy', 0)}%")

        # 7. Upsert to Database
        db.upsert_ticker_intelligence({
            "ticker": ticker,
            "asset_name": name,
            "exchange": exchange,
            "latest_price": latest_price,
            "total_mentions": sent_agg["total_mentions"],
            "sentiment_score": sent_agg["sentiment_score"],
            "sentiment_direction": market_dir,
            "pos_pct": sent_agg["pos_pct"],
            "neg_pct": sent_agg["neg_pct"],
            "neu_pct": sent_agg["neu_pct"],
            "investor_position": investor_pos,
            "position_quote": pos_quote,
            "support_level": tech_levels["support"],
            "resistance_level": tech_levels["resistance"],
            "recent_news": news_articles[:3],
            "predicted_price": pred_price
        })

    # Generate Reports
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING EXPORTABLE CSV AND PDF INTELLIGENCE DOSSIERS")
    logger.info("=" * 70)
    all_intel = db.get_all_ticker_intelligence()
    csv_path = report_generator.generate_csv_report(all_intel)
    pdf_path = report_generator.generate_pdf_report(all_intel)
    logger.info(f"CSV Report generated at: {csv_path}")
    logger.info(f"PDF Report generated at: {pdf_path}")
    logger.info("\nSUCCESS: End-to-end pipeline completed cleanly for all 10 assets!")


if __name__ == "__main__":
    run_full_pipeline()

