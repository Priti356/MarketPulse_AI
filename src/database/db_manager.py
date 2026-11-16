"""
Database Manager for MarketPulse AI.
Supports SQLite (default) and PostgreSQL via SQLAlchemy.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Text, DateTime,
    Index, UniqueConstraint, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

logger = logging.getLogger(__name__)

Base = declarative_base()


class RawSocialPost(Base):
    __tablename__ = "raw_social_posts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    asset_name = Column(String(100), nullable=False)
    exchange = Column(String(20), nullable=False, index=True)
    post_text = Column(Text, nullable=False)
    post_timestamp = Column(DateTime, nullable=False, index=True)
    likes = Column(Integer, default=0)
    reposts = Column(Integer, default=0)
    views = Column(Integer, default=0)
    author_followers = Column(Integer, default=0)
    source_name = Column(String(50), nullable=False, index=True)
    post_url = Column(String(500), nullable=False)
    post_id = Column(String(100), unique=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_social_ticker_time", "ticker", "post_timestamp"),
    )


class RawNewsArticle(Base):
    __tablename__ = "raw_news_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    asset_name = Column(String(100), nullable=False)
    exchange = Column(String(20), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    url = Column(String(500), unique=True, nullable=False, index=True)
    published_date = Column(DateTime, nullable=False, index=True)
    source = Column(String(100), nullable=False)
    snippet = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_news_ticker_pubdate", "ticker", "published_date"),
    )


class CleanedSentimentRecord(Base):
    __tablename__ = "cleaned_sentiment_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_type = Column(String(20), nullable=False)  # 'social' or 'news'
    source_id = Column(String(100), nullable=True)
    ticker = Column(String(20), nullable=False, index=True)
    cleaned_text = Column(Text, nullable=False)
    sentiment_label = Column(String(20), nullable=False)  # Positive, Negative, Neutral
    sentiment_score = Column(Float, nullable=False)  # -1.0 to 1.0
    prob_positive = Column(Float, default=0.0)
    prob_negative = Column(Float, default=0.0)
    prob_neutral = Column(Float, default=0.0)
    weighted_score = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_sentiment_ticker_time", "ticker", "timestamp"),
    )


class PriceHistory(Base):
    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    daily_return = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_ticker_date"),
        Index("idx_price_ticker_date", "ticker", "date"),
    )


class EngineeredFeature(Base):
    __tablename__ = "engineered_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    features_json = Column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_feat_ticker_date"),
    )


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False, index=True)
    prediction_date = Column(DateTime, nullable=False, index=True)
    model_name = Column(String(50), nullable=False)
    current_price = Column(Float, nullable=False)
    predicted_return = Column(Float, nullable=False)
    predicted_price = Column(Float, nullable=False)
    actual_price = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    directional_accuracy = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class TickerIntelligenceRecord(Base):
    __tablename__ = "ticker_intelligence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), unique=True, nullable=False, index=True)
    asset_name = Column(String(100), nullable=False)
    exchange = Column(String(20), nullable=False, index=True)
    latest_price = Column(Float, nullable=False)
    total_mentions = Column(Integer, default=0)
    sentiment_score = Column(Float, default=0.0)
    sentiment_direction = Column(String(20), default="Neutral")
    pos_pct = Column(Float, default=0.0)
    neg_pct = Column(Float, default=0.0)
    neu_pct = Column(Float, default=0.0)
    investor_position = Column(String(50), default="Not stated")
    position_quote = Column(Text, nullable=True)
    support_level = Column(Float, nullable=False)
    resistance_level = Column(Float, nullable=False)
    recent_news_json = Column(Text, nullable=True)
    predicted_price = Column(Float, nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class DatabaseManager:
    """Encapsulates database operations for MarketPulse AI."""

    def __init__(self, db_url: Optional[str] = None):
        if not db_url:
            db_url = os.getenv("DATABASE_URL", "sqlite:///data/marketpulse.db")
        
        # Ensure directory exists for SQLite
        if db_url.startswith("sqlite:///"):
            file_path = db_url.replace("sqlite:///", "")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        self.engine = create_engine(db_url, echo=False)
        self.SessionFactory = scoped_session(sessionmaker(bind=self.engine))
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized at: {db_url}")

    def get_session(self):
        return self.SessionFactory()

    def save_social_posts(self, posts: List[Dict[str, Any]]) -> int:
        """Batch save social posts, skipping duplicates by post_id."""
        if not posts:
            return 0
        session = self.get_session()
        added_count = 0
        try:
            for p in posts:
                post_id = str(p.get("post_id") or p.get("post_url"))
                existing = session.query(RawSocialPost).filter_by(post_id=post_id).first()
                if not existing:
                    timestamp = p.get("post_timestamp")
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        except Exception:
                            timestamp = datetime.now(timezone.utc)
                    elif timestamp is None:
                        timestamp = datetime.now(timezone.utc)

                    record = RawSocialPost(
                        ticker=p["ticker"],
                        asset_name=p.get("asset_name", ""),
                        exchange=p.get("exchange", ""),
                        post_text=p["post_text"],
                        post_timestamp=timestamp,
                        likes=int(p.get("likes", 0) or 0),
                        reposts=int(p.get("reposts", 0) or 0),
                        views=int(p.get("views", 0) or 0),
                        author_followers=int(p.get("author_followers", 0) or 0),
                        source_name=p.get("source_name", "Unknown"),
                        post_url=p.get("post_url", ""),
                        post_id=post_id,
                    )
                    session.add(record)
                    added_count += 1
            session.commit()
            return added_count
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving social posts: {e}")
            raise
        finally:
            session.close()

    def save_news_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Batch save news articles, skipping duplicates by URL."""
        if not articles:
            return 0
        session = self.get_session()
        added_count = 0
        try:
            for a in articles:
                url = a.get("url")
                if not url:
                    continue
                existing = session.query(RawNewsArticle).filter_by(url=url).first()
                if not existing:
                    pub_date = a.get("published_date")
                    if isinstance(pub_date, str):
                        try:
                            pub_date = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                        except Exception:
                            pub_date = datetime.now(timezone.utc)
                    elif pub_date is None:
                        pub_date = datetime.now(timezone.utc)

                    record = RawNewsArticle(
                        ticker=a["ticker"],
                        asset_name=a.get("asset_name", ""),
                        exchange=a.get("exchange", ""),
                        title=a.get("title", ""),
                        url=url,
                        published_date=pub_date,
                        source=a.get("source", "Exa News"),
                        snippet=a.get("snippet", ""),
                    )
                    session.add(record)
                    added_count += 1
            session.commit()
            return added_count
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving news articles: {e}")
            raise
        finally:
            session.close()

    def save_sentiment_records(self, records: List[Dict[str, Any]]) -> int:
        """Batch save sentiment records."""
        if not records:
            return 0
        session = self.get_session()
        try:
            objs = []
            for r in records:
                ts = r.get("timestamp")
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except Exception:
                        ts = datetime.now(timezone.utc)
                elif ts is None:
                    ts = datetime.now(timezone.utc)

                objs.append(CleanedSentimentRecord(
                    source_type=r.get("source_type", "social"),
                    source_id=str(r.get("source_id", "")),
                    ticker=r["ticker"],
                    cleaned_text=r["cleaned_text"],
                    sentiment_label=r["sentiment_label"],
                    sentiment_score=float(r["sentiment_score"]),
                    prob_positive=float(r.get("prob_positive", 0.0)),
                    prob_negative=float(r.get("prob_negative", 0.0)),
                    prob_neutral=float(r.get("prob_neutral", 0.0)),
                    weighted_score=float(r.get("weighted_score", r["sentiment_score"])),
                    timestamp=ts,
                ))
            session.bulk_save_objects(objs)
            session.commit()
            return len(objs)
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving sentiment records: {e}")
            raise
        finally:
            session.close()

    def save_price_history(self, ticker: str, df: pd.DataFrame) -> int:
        """Upsert price history DataFrame (indexed by Date or containing 'Date' column)."""
        if df.empty:
            return 0
        session = self.get_session()
        count = 0
        try:
            df_reset = df.reset_index() if "Date" not in df.columns else df.copy()
            for _, row in df_reset.iterrows():
                dt = row.get("Date") or row.get("Datetime") or row.get("date")
                if isinstance(dt, str):
                    dt = pd.to_datetime(dt)
                if hasattr(dt, "to_pydatetime"):
                    dt = dt.to_pydatetime()
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

                existing = session.query(PriceHistory).filter_by(ticker=ticker, date=dt).first()
                if not existing:
                    rec = PriceHistory(
                        ticker=ticker,
                        date=dt,
                        open=float(row.get("Open", 0.0)),
                        high=float(row.get("High", 0.0)),
                        low=float(row.get("Low", 0.0)),
                        close=float(row.get("Close", 0.0)),
                        volume=float(row.get("Volume", 0.0)),
                        daily_return=float(row.get("daily_return", 0.0)) if "daily_return" in row else None
                    )
                    session.add(rec)
                    count += 1
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving price history for {ticker}: {e}")
            raise
        finally:
            session.close()

    def upsert_ticker_intelligence(self, data: Dict[str, Any]) -> None:
        """Upsert single ticker intelligence structured report."""
        session = self.get_session()
        try:
            ticker = data["ticker"]
            rec = session.query(TickerIntelligenceRecord).filter_by(ticker=ticker).first()
            if not rec:
                rec = TickerIntelligenceRecord(ticker=ticker)
                session.add(rec)

            rec.asset_name = data.get("asset_name", "")
            rec.exchange = data.get("exchange", "")
            rec.latest_price = float(data.get("latest_price", 0.0))
            rec.total_mentions = int(data.get("total_mentions", 0))
            rec.sentiment_score = float(data.get("sentiment_score", 0.0))
            rec.sentiment_direction = data.get("sentiment_direction", "Neutral")
            rec.pos_pct = float(data.get("pos_pct", 0.0))
            rec.neg_pct = float(data.get("neg_pct", 0.0))
            rec.neu_pct = float(data.get("neu_pct", 0.0))
            rec.investor_position = data.get("investor_position", "Not stated")
            rec.position_quote = data.get("position_quote", "")
            rec.support_level = float(data.get("support_level", 0.0))
            rec.resistance_level = float(data.get("resistance_level", 0.0))
            rec.recent_news_json = json.dumps(data.get("recent_news", []), default=str)
            rec.predicted_price = float(data.get("predicted_price", 0.0))
            rec.updated_at = datetime.now(timezone.utc)

            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error upserting ticker intelligence: {e}")
            raise
        finally:
            session.close()

    def get_social_posts(self, ticker: Optional[str] = None, days: int = 10) -> pd.DataFrame:
        """Fetch social posts as DataFrame."""
        session = self.get_session()
        try:
            query = session.query(RawSocialPost)
            if ticker:
                query = query.filter(RawSocialPost.ticker == ticker)
            posts = query.order_by(RawSocialPost.post_timestamp.desc()).all()
            if not posts:
                return pd.DataFrame()
            data = [{
                "id": p.id,
                "ticker": p.ticker,
                "asset_name": p.asset_name,
                "exchange": p.exchange,
                "post_text": p.post_text,
                "post_timestamp": p.post_timestamp,
                "likes": p.likes,
                "reposts": p.reposts,
                "views": p.views,
                "author_followers": p.author_followers,
                "source_name": p.source_name,
                "post_url": p.post_url,
                "post_id": p.post_id
            } for p in posts]
            return pd.DataFrame(data)
        finally:
            session.close()

    def get_news_articles(self, ticker: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        """Fetch news articles as DataFrame."""
        session = self.get_session()
        try:
            query = session.query(RawNewsArticle)
            if ticker:
                query = query.filter(RawNewsArticle.ticker == ticker)
            articles = query.order_by(RawNewsArticle.published_date.desc()).limit(limit).all()
            if not articles:
                return pd.DataFrame()
            data = [{
                "id": a.id,
                "ticker": a.ticker,
                "asset_name": a.asset_name,
                "exchange": a.exchange,
                "title": a.title,
                "url": a.url,
                "published_date": a.published_date,
                "source": a.source,
                "snippet": a.snippet
            } for a in articles]
            return pd.DataFrame(data)
        finally:
            session.close()

    def get_price_history(self, ticker: str) -> pd.DataFrame:
        """Fetch price history for a given ticker."""
        session = self.get_session()
        try:
            records = session.query(PriceHistory).filter_by(ticker=ticker).order_by(PriceHistory.date.asc()).all()
            if not records:
                return pd.DataFrame()
            data = [{
                "Date": r.date,
                "Open": r.open,
                "High": r.high,
                "Low": r.low,
                "Close": r.close,
                "Volume": r.volume,
                "daily_return": r.daily_return
            } for r in records]
            df = pd.DataFrame(data)
            df.set_index("Date", inplace=True)
            return df
        finally:
            session.close()

    def get_all_ticker_intelligence(self) -> List[Dict[str, Any]]:
        """Fetch all current ticker intelligence records."""
        session = self.get_session()
        try:
            records = session.query(TickerIntelligenceRecord).order_by(
                TickerIntelligenceRecord.total_mentions.desc()
            ).all()
            res = []
            for r in records:
                recent_news = []
                if r.recent_news_json:
                    try:
                        recent_news = json.loads(r.recent_news_json)
                    except Exception:
                        recent_news = []
                res.append({
                    "ticker": r.ticker,
                    "asset_name": r.asset_name,
                    "exchange": r.exchange,
                    "latest_price": r.latest_price,
                    "total_mentions": r.total_mentions,
                    "sentiment_score": r.sentiment_score,
                    "sentiment_direction": r.sentiment_direction,
                    "pos_pct": r.pos_pct,
                    "neg_pct": r.neg_pct,
                    "neu_pct": r.neu_pct,
                    "investor_position": r.investor_position,
                    "position_quote": r.position_quote,
                    "support_level": r.support_level,
                    "resistance_level": r.resistance_level,
                    "recent_news": recent_news,
                    "predicted_price": r.predicted_price,
                    "updated_at": r.updated_at
                })
            return res
        finally:
            session.close()
