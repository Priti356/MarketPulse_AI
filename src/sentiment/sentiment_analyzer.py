"""
Sentiment Analyzer for MarketPulse AI.
Orchestrates multilingual FinBERT scoring, engagement weighting, and ticker-level aggregation.
"""

import math
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.sentiment.finbert_model import FinancialSentimentModel
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Computes engagement-weighted sentiment across social posts and financial news."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
        self.model = FinancialSentimentModel()

    def calculate_engagement_weight(
        self,
        likes: int = 0,
        reposts: int = 0,
        views: int = 0,
        followers: int = 0
    ) -> float:
        """
        Computes capped reach factor to prevent single viral bot or outlier from dominating:
        weight = 1.0 + min(log10(1 + likes + 2*reposts + 0.1*views + 0.005*followers), 3.0)
        Range: [1.0, 4.0]
        """
        raw_engagement = (
            max(0, int(likes or 0)) +
            2.0 * max(0, int(reposts or 0)) +
            0.1 * max(0, int(views or 0)) +
            0.005 * max(0, int(followers or 0))
        )
        log_term = math.log10(1.0 + raw_engagement)
        capped_weight = 1.0 + min(log_term, 3.0)
        return round(float(capped_weight), 3)

    def analyze_social_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score each post with multilingual sentiment model and calculate engagement weight."""
        scored_records = []
        for p in posts:
            text = p.get("post_text", "")
            sentiment = self.model.predict(text)

            weight = self.calculate_engagement_weight(
                likes=p.get("likes", 0),
                reposts=p.get("reposts", 0),
                views=p.get("views", 0),
                followers=p.get("author_followers", 0)
            )

            weighted_score = round(sentiment["sentiment_score"] * weight, 4)

            scored = {
                "source_type": "social",
                "source_id": p.get("post_id"),
                "ticker": p["ticker"],
                "cleaned_text": text,
                "sentiment_label": sentiment["sentiment_label"],
                "sentiment_score": sentiment["sentiment_score"],
                "prob_positive": sentiment["prob_positive"],
                "prob_negative": sentiment["prob_negative"],
                "prob_neutral": sentiment["prob_neutral"],
                "weighted_score": weighted_score,
                "weight": weight,
                "timestamp": p.get("post_timestamp"),
                "source_name": p.get("source_name", "Social")
            }
            scored_records.append(scored)

        # Save to database
        self.db.save_sentiment_records(scored_records)
        return scored_records

    def analyze_news_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score each financial news article."""
        scored_records = []
        for a in articles:
            # Combine title and snippet for holistic context
            content = f"{a.get('title', '')}. {a.get('snippet', '')}"
            sentiment = self.model.predict(content)

            # News articles carry a reliable institutional base weight of 1.5
            weight = 1.5
            weighted_score = round(sentiment["sentiment_score"] * weight, 4)

            scored = {
                "source_type": "news",
                "source_id": a.get("url"),
                "ticker": a["ticker"],
                "cleaned_text": content,
                "sentiment_label": sentiment["sentiment_label"],
                "sentiment_score": sentiment["sentiment_score"],
                "prob_positive": sentiment["prob_positive"],
                "prob_negative": sentiment["prob_negative"],
                "prob_neutral": sentiment["prob_neutral"],
                "weighted_score": weighted_score,
                "weight": weight,
                "timestamp": a.get("published_date"),
                "source_name": a.get("source", "Exa News")
            }
            scored_records.append(scored)

        self.db.save_sentiment_records(scored_records)
        return scored_records

    def aggregate_ticker_sentiment(
        self,
        social_records: List[Dict[str, Any]],
        news_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate sentiment statistics across all posts and news for a ticker:
        - sentiment_score (-1.0 to 1.0)
        - sentiment_direction ('Bullish', 'Bearish', 'Neutral')
        - positive, negative, neutral percentages
        - weighted sentiment score
        - total social mentions
        """
        all_records = list(social_records) + list(news_records)
        if not all_records:
            return {
                "total_mentions": 0,
                "sentiment_score": 0.0,
                "weighted_sentiment": 0.0,
                "sentiment_direction": "Neutral",
                "pos_pct": 0.0,
                "neg_pct": 0.0,
                "neu_pct": 100.0,
                "counts": {"Positive": 0, "Negative": 0, "Neutral": 0}
            }

        total_weight = sum(r.get("weight", 1.0) for r in all_records)
        weighted_sum = sum(r.get("weighted_score", 0.0) for r in all_records)
        scores = [r["sentiment_score"] for r in all_records]

        avg_score = round(float(np.mean(scores)), 4)
        weighted_avg = round(float(weighted_sum / total_weight) if total_weight > 0 else avg_score, 4)

        labels = [r["sentiment_label"] for r in all_records]
        total_count = len(labels)
        pos_count = labels.count("Positive")
        neg_count = labels.count("Negative")
        neu_count = labels.count("Neutral")

        pos_pct = round((pos_count / total_count) * 100.0, 1)
        neg_pct = round((neg_count / total_count) * 100.0, 1)
        neu_pct = round((neu_count / total_count) * 100.0, 1)

        # Direction based on combined weighted score and class dominance
        if weighted_avg >= 0.12 or pos_pct >= 45.0:
            direction = "Bullish"
        elif weighted_avg <= -0.12 or neg_pct >= 45.0:
            direction = "Bearish"
        else:
            direction = "Neutral"

        return {
            "total_mentions": len(social_records),
            "news_count": len(news_records),
            "sentiment_score": avg_score,
            "weighted_sentiment": weighted_avg,
            "sentiment_direction": direction,
            "pos_pct": pos_pct,
            "neg_pct": neg_pct,
            "neu_pct": neu_pct,
            "counts": {"Positive": pos_count, "Negative": neg_count, "Neutral": neu_count}
        }

