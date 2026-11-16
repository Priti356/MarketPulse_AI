"""
Investor Position Extractor for MarketPulse AI.
Extracts explicit investor positions (Buy, Sell, Hold, Bullish, Bearish, or Not stated)
from community posts and institutional discussions in English, Chinese, and Korean.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PositionExtractor:
    """Extracts explicit stance and investor positioning from financial text."""

    def __init__(self):
        # Explicit positioning regex patterns with quotes capture
        self.patterns = [
            # Buy patterns
            (
                "Buy",
                r"(?i)\b(strong\s*buy|buy\s*rating|suggest\s*buy|recommend\s*buy|adding\s*to\s*position|accumulate)\b|"
                r"(建議買入|強烈買入|加倉|吸納|逢低買入|買入評級|目標價上看)|"
                r"(강력\s*매수|적극\s*매수|매수\s*추천|비중\s*확대|신규\s*매수)"
            ),
            # Sell patterns
            (
                "Sell",
                r"(?i)\b(strong\s*sell|sell\s*rating|suggest\s*sell|cut\s*losses|exit\s*position|take\s*profit|dump)\b|"
                r"(建議沽出|沽售|減持|斬倉|止蝕|清倉|逢高減倉|逢高派發)|"
                r"(적극\s*매도|비중\s*축소|손절|차익\s*실현|전량\s*매도)"
            ),
            # Hold patterns
            (
                "Hold",
                r"(?i)\b(maintain\s*hold|hold\s*position|neutral\s*stance|wait\s*and\s*see|standby)\b|"
                r"(堅決持有|繼續持有|維持持有|觀望|按兵不動|暫時持有)|"
                r"(보유\s*의견|관망|중립\s*유지|보유\s*유지)"
            ),
            # Bullish sentiment stance
            (
                "Bullish",
                r"(?i)\b(bullish|highly\s*optimistic|upside\s*potential|breakout\s*expected)\b|"
                r"(看好|偏好|樂觀|走勢向好|升勢強勁|蓄勢待發)|"
                r"(상승\s*전망|상방\s*기대|우상향|긍정적\s*전망)"
            ),
            # Bearish sentiment stance
            (
                "Bearish",
                r"(?i)\b(bearish|pessimistic|downside\s*risk|correction\s*ahead)\b|"
                r"(看淡|悲觀|走勢偏弱|下行風險|沽壓沉重)|"
                r"(하락\s*전망|하방\s*압력|조정\s*국면|부정적\s*전망)"
            )
        ]

    def extract_position(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single text snippet.
        Returns:
          - position: "Buy" | "Sell" | "Hold" | "Bullish" | "Bearish" | "Not stated"
          - quote: relevant excerpt triggering the detection
          - confidence: float
        """
        if not text or len(text.strip()) < 5:
            return {"position": "Not stated", "quote": "", "confidence": 0.0}

        for position_name, regex in self.patterns:
            match = re.search(regex, text)
            if match:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                quote = text[start:end].strip()
                return {
                    "position": position_name,
                    "quote": quote,
                    "confidence": 0.88
                }

        return {"position": "Not stated", "quote": "", "confidence": 0.0}

    def aggregate_asset_position(self, posts: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Synthesize dominant investor position across a collection of posts for an asset.
        Returns (dominant_position, dominant_quote).
        Distinguishes explicit opinions; returns 'Not stated' when no clear position exists.
        """
        if not posts:
            return "Not stated", ""

        position_counts = {
            "Buy": 0,
            "Sell": 0,
            "Hold": 0,
            "Bullish": 0,
            "Bearish": 0
        }
        quotes = {k: [] for k in position_counts.keys()}

        for p in posts:
            text = p.get("post_text", "")
            res = self.extract_position(text)
            pos = res["position"]
            if pos in position_counts:
                position_counts[pos] += 1
                if res["quote"]:
                    quotes[pos].append(res["quote"])

        # Determine dominant explicit position
        total_explicit = sum(position_counts.values())
        if total_explicit == 0:
            return "Not stated", "Community discussions do not contain explicit buy/sell/hold directives."

        # Find position with maximum votes
        sorted_positions = sorted(position_counts.items(), key=lambda x: x[1], reverse=True)
        top_pos, top_count = sorted_positions[0]

        # Require minimum threshold of 2 mentions or 30% of explicit mentions to claim consensus
        if top_count >= 2 or (top_count / total_explicit) >= 0.35:
            sample_quote = quotes[top_pos][0] if quotes[top_pos] else ""
            return top_pos, sample_quote

        return "Not stated", "Divided community sentiment without clear positional consensus."

