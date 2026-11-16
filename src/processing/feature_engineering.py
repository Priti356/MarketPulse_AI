"""
Financial Feature Engineering for MarketPulse AI.
Merges price action, technical indicators, and multi-source sentiment.
Ensures strict temporal alignment to avoid lookahead bias / data leakage.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Builds tabular feature matrices for ML price prediction."""

    @staticmethod
    def aggregate_daily_sentiment(
        sentiment_records: List[Dict[str, Any]],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Aggregate post and news sentiment records by date (YYYY-MM-DD).
        Calculates:
          - mention_count
          - weighted_sentiment
          - avg_sentiment
          - pos_ratio
          - neg_ratio
          - news_sentiment
          - news_count
          - engagement_score
        """
        if not sentiment_records:
            return pd.DataFrame()

        df = pd.DataFrame(sentiment_records)
        if "timestamp" not in df.columns:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()

        daily_rows = []
        for dt, group in df.groupby("date"):
            social_grp = group[group["source_type"] == "social"]
            news_grp = group[group["source_type"] == "news"]

            social_mentions = len(social_grp)
            news_mentions = len(news_grp)

            # Sentiment scores
            soc_scores = social_grp["sentiment_score"].tolist()
            news_scores = news_grp["sentiment_score"].tolist()
            all_scores = soc_scores + news_scores

            weights = social_grp.get("weight", pd.Series(1.0, index=social_grp.index)).tolist() + \
                      news_grp.get("weight", pd.Series(1.5, index=news_grp.index)).tolist()

            avg_sentiment = float(np.mean(all_scores)) if all_scores else 0.0
            total_weight = sum(weights)
            weighted_sentiment = float(sum(s * w for s, w in zip(all_scores, weights)) / total_weight) if total_weight > 0 else avg_sentiment

            # Pos/Neg ratios
            pos_count = sum(1 for s in all_scores if s >= 0.15)
            neg_count = sum(1 for s in all_scores if s <= -0.15)
            total_count = len(all_scores)

            pos_ratio = pos_count / total_count if total_count > 0 else 0.0
            neg_ratio = neg_count / total_count if total_count > 0 else 0.0
            news_sentiment = float(np.mean(news_scores)) if news_scores else 0.0
            engagement_score = float(np.mean(social_grp["weight"])) if not social_grp.empty else 1.0

            daily_rows.append({
                "Date": dt,
                "mention_count": social_mentions,
                "weighted_sentiment": round(weighted_sentiment, 4),
                "avg_sentiment": round(avg_sentiment, 4),
                "pos_ratio": round(pos_ratio, 4),
                "neg_ratio": round(neg_ratio, 4),
                "news_sentiment": round(news_sentiment, 4),
                "news_count": news_mentions,
                "engagement_score": round(engagement_score, 3)
            })

        daily_df = pd.DataFrame(daily_rows)
        daily_df.set_index("Date", inplace=True)
        daily_df = daily_df.sort_index()
        return daily_df

    def build_feature_dataset(
        self,
        price_df: pd.DataFrame,
        sentiment_records: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Merge historical prices with daily sentiment features.
        Calculates:
          - return_1d, return_5d, volatility_10d, volume_change_1d
          - rsi_14, macd, macd_signal
          - lagged sentiment: sentiment_lag_1d, sentiment_lag_2d
          - lagged price returns: return_lag_1d, return_lag_2d
          - target: target_next_return (Close[t+1] / Close[t] - 1)
        """
        if price_df.empty:
            return pd.DataFrame()

        df = price_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)
            else:
                df.index = pd.to_datetime(df.index)

        df.index = df.index.tz_localize(None).normalize()
        df = df.sort_index()

        # 1. Price-derived core features
        df["return_1d"] = df["Close"].pct_change().fillna(0.0)
        df["return_5d"] = df["Close"].pct_change(5).fillna(0.0)
        df["volatility_10d"] = df["return_1d"].rolling(10).std().fillna(0.0)
        df["volume_change_1d"] = df["Volume"].pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0)

        # Technical indicators
        delta = df["Close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(window=14, min_periods=5).mean()
        avg_loss = loss.rolling(window=14, min_periods=5).mean().replace(0.0, 1e-9)
        rs = avg_gain / avg_loss
        df["rsi_14"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

        # MACD
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        # 2. Daily sentiment features
        sent_df = self.aggregate_daily_sentiment(sentiment_records)
        if not sent_df.empty:
            df = df.join(sent_df, how="left")
        else:
            df["mention_count"] = 0
            df["weighted_sentiment"] = 0.0
            df["avg_sentiment"] = 0.0
            df["pos_ratio"] = 0.0
            df["neg_ratio"] = 0.0
            df["news_sentiment"] = 0.0
            df["news_count"] = 0
            df["engagement_score"] = 1.0

        # Fill missing sentiment on non-discussion days with neutral baseline
        df["mention_count"] = df["mention_count"].fillna(0)
        df["weighted_sentiment"] = df["weighted_sentiment"].fillna(0.0)
        df["avg_sentiment"] = df["avg_sentiment"].fillna(0.0)
        df["pos_ratio"] = df["pos_ratio"].fillna(0.0)
        df["neg_ratio"] = df["neg_ratio"].fillna(0.0)
        df["news_sentiment"] = df["news_sentiment"].fillna(0.0)
        df["news_count"] = df["news_count"].fillna(0)
        df["engagement_score"] = df["engagement_score"].fillna(1.0)

        # 3. Lagged features (strictly past values to prevent lookahead leakage)
        df["sentiment_lag_1d"] = df["weighted_sentiment"].shift(1).fillna(0.0)
        df["sentiment_lag_2d"] = df["weighted_sentiment"].shift(2).fillna(0.0)
        df["return_lag_1d"] = df["return_1d"].shift(1).fillna(0.0)
        df["return_lag_2d"] = df["return_1d"].shift(2).fillna(0.0)

        # 4. Target variable: Next session return R_{t+1}
        df["target_next_return"] = df["return_1d"].shift(-1)

        # Clean infs/nans in features
        feature_cols = self.get_feature_columns()
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        return df

    @staticmethod
    def get_feature_columns() -> List[str]:
        """Return standardized feature column names."""
        return [
            "return_1d",
            "return_5d",
            "volatility_10d",
            "volume_change_1d",
            "rsi_14",
            "macd",
            "macd_signal",
            "mention_count",
            "weighted_sentiment",
            "avg_sentiment",
            "pos_ratio",
            "neg_ratio",
            "news_sentiment",
            "news_count",
            "engagement_score",
            "sentiment_lag_1d",
            "sentiment_lag_2d",
            "return_lag_1d",
            "return_lag_2d"
        ]

