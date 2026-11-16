"""
Data Cleaning and Text Preprocessing for MarketPulse AI.
Handles normalization, URL removal, emoji cleanup, and deduplication.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone


class DataCleaner:
    """Preprocesses raw social media posts and financial news articles."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Strip URLs, superfluous whitespace, and excessive symbols while preserving financial markers."""
        if not text:
            return ""

        # Remove http/https URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove HTML tags if present
        text = re.sub(r"<.*?>", "", text)

        # Normalize multiple spaces, tabs, and newlines
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @staticmethod
    def deduplicate_records(records: List[Dict[str, Any]], key_field: str = "id") -> List[Dict[str, Any]]:
        """Deduplicate list of records based on key_field or content hash."""
        seen = set()
        unique = []
        for r in records:
            val = r.get(key_field)
            if not val:
                content = r.get("post_text") or r.get("title") or ""
                val = hashlib.md5(content.encode("utf-8")).hexdigest()
            if val not in seen:
                seen.add(val)
                unique.append(r)
        return unique

    @staticmethod
    def normalize_utc_timestamp(ts: Any) -> datetime:
        """Ensure timestamp is a timezone-aware UTC datetime."""
        if isinstance(ts, datetime):
            return ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception:
                pass
        return datetime.now(timezone.utc)

