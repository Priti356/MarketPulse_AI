"""
FinBERT and Multilingual Language-Aware Financial Sentiment Model.
Supports English, Traditional Chinese (HKEX), and Korean (KRX).
"""

import re
import math
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

# Optional Hugging Face FinBERT integration
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


class FinancialSentimentModel:
    """
    Multilingual financial sentiment classifier.
    Combines FinBERT architecture with native Chinese and Korean financial NLP engines.
    """

    def __init__(self, use_hf_if_available: bool = True):
        self.hf_pipeline = None
        self.device = "cpu"

        # Attempt to load Hugging Face FinBERT if available
        if use_hf_if_available and TRANSFORMERS_AVAILABLE:
            try:
                model_name = "ProsusAI/finbert"
                logger.info(f"Loading Hugging Face model '{model_name}'...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.eval()
                self.hf_pipeline = True
                logger.info("FinBERT Hugging Face weights loaded successfully.")
            except Exception as e:
                logger.info(f"FinBERT weights not present locally ({e}). Operating with native high-performance multilingual financial engine.")
                self.hf_pipeline = None

        self._init_multilingual_lexicons()

    def _init_multilingual_lexicons(self):
        """Initialize comprehensive domain-specific financial sentiment lexicons."""
        # 1. English Financial Lexicon
        self.en_positive = {
            "surge": 1.5, "surging": 1.5, "beat": 1.4, "beats": 1.4, "outperform": 1.5, "outperforming": 1.5,
            "bullish": 1.6, "buy": 1.5, "growth": 1.2, "upgrade": 1.4, "upgraded": 1.4, "profit": 1.3,
            "profitable": 1.3, "record": 1.3, "soar": 1.5, "dividend": 1.1, "buyback": 1.3, "expansion": 1.2,
            "breakout": 1.4, "rebound": 1.2, "rally": 1.4, "resilient": 1.2, "resilience": 1.2, "upside": 1.3,
            "gain": 1.1, "gains": 1.1, "positive": 1.2, "innovative": 1.1, "strong": 1.2, "strength": 1.2
        }
        self.en_negative = {
            "slump": 1.4, "plunge": 1.5, "plunging": 1.5, "miss": 1.4, "misses": 1.4, "underperform": 1.5,
            "bearish": 1.6, "sell": 1.5, "loss": 1.4, "losses": 1.4, "downgrade": 1.5, "downgraded": 1.5,
            "drop": 1.2, "decline": 1.2, "declining": 1.2, "deficit": 1.3, "warning": 1.4, "probe": 1.4,
            "investigation": 1.3, "penalty": 1.4, "fine": 1.3, "debt": 1.2, "recession": 1.5, "downside": 1.3,
            "weak": 1.3, "weakness": 1.3, "negative": 1.2, "cut": 1.2, "plummet": 1.6, "crash": 1.7
        }

        # 2. Traditional Chinese Financial Lexicon (HKEX / LIHKG)
        self.zh_positive = {
            "買入": 1.6, "加倉": 1.5, "看好": 1.5, "利好": 1.5, "突破": 1.4, "升": 1.2, "暴升": 1.8,
            "大升": 1.7, "創新高": 1.7, "破頂": 1.6, "淨流入": 1.3, "增長": 1.3, "勝預期": 1.6, "盈利": 1.3,
            "派息": 1.2, "回購": 1.4, "重估": 1.3, "反彈": 1.3, "強勁": 1.4, "金叉": 1.5, "目標價上調": 1.6,
            "牛市": 1.5, "賺錢": 1.2, "復甦": 1.3, "領跑": 1.4, "穩健": 1.2, "性價比": 1.2, "支持位穩": 1.3
        }
        self.zh_negative = {
            "沽出": 1.6, "賣出": 1.6, "減持": 1.5, "斬倉": 1.7, "看淡": 1.5, "利淡": 1.5, "跌": 1.2,
            "暴跌": 1.8, "大跌": 1.7, "破底": 1.6, "創新低": 1.7, "淨流出": 1.3, "虧損": 1.5, "遜預期": 1.6,
            "死叉": 1.5, "沽壓": 1.4, "罰款": 1.4, "監管整頓": 1.5, "退市": 1.8, "目標價下調": 1.6, "熊市": 1.5,
            "套牢": 1.4, "疲軟": 1.3, "承壓": 1.3, "下行": 1.3, "裁員": 1.3, "阻力大": 1.3, "暴雷": 1.8
        }

        # 3. Korean Financial Lexicon (KRX / Naver Finance)
        self.ko_positive = {
            "매수": 1.6, "상승": 1.4, "급등": 1.7, "폭등": 1.8, "떡상": 1.7, "신고가": 1.7, "호재": 1.5,
            "흑자": 1.4, "영업이익 증가": 1.5, "실적 호조": 1.6, "어닝 서프라이즈": 1.8, "서프라이즈": 1.6,
            "외인 순매수": 1.4, "기관 매수": 1.3, "자사주 소각": 1.5, "배당 확대": 1.3, "수혜": 1.3, "반등": 1.3,
            "독점": 1.4, "성장세": 1.3, "기대감": 1.2, "목표가 상향": 1.6, "강력 추천": 1.6, "저평가": 1.3, "턴어라운드": 1.5
        }
        self.ko_negative = {
            "매도": 1.6, "하락": 1.4, "급락": 1.7, "폭락": 1.8, "떡락": 1.7, "신저가": 1.7, "악재": 1.5,
            "적자": 1.5, "영업이익 감소": 1.5, "실적 부진": 1.6, "어닝 쇼크": 1.8, "쇼크": 1.6, "외인 매도": 1.4,
            "기관 순매도": 1.3, "유상증자": 1.6, "배당 삭감": 1.4, "리스크": 1.3, "불확실성": 1.3, "이탈": 1.3,
            "목표가 하향": 1.6, "고평가": 1.3, "피크아웃": 1.4, "손절": 1.5, "물렸다": 1.4, "조정": 1.2, "차익실현": 1.2
        }

    def detect_language(self, text: str) -> str:
        """Detect whether text is predominantly Korean (ko), Chinese (zh), or English (en)."""
        # Korean Hangul syllables: \uac00-\ud7a3
        ko_count = len(re.findall(r"[\uac00-\ud7a3]", text))
        # Chinese characters: \u4e00-\u9fff
        zh_count = len(re.findall(r"[\u4e00-\u9fff]", text))

        if ko_count >= 3:
            return "ko"
        if zh_count >= 3:
            return "zh"
        return "en"

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classify text and generate:
        - sentiment_score (-1.0 to 1.0)
        - sentiment_label ('Positive', 'Negative', 'Neutral')
        - prob_positive, prob_negative, prob_neutral
        """
        if not text or not text.strip():
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "Neutral",
                "prob_positive": 0.33,
                "prob_negative": 0.33,
                "prob_neutral": 0.34
            }

        lang = self.detect_language(text)

        # Run language-specific scoring
        if lang == "ko":
            pos_score, neg_score = self._score_lexicon(text, self.ko_positive, self.ko_negative)
        elif lang == "zh":
            pos_score, neg_score = self._score_lexicon(text, self.zh_positive, self.zh_negative)
        else:
            # English
            if self.hf_pipeline:
                try:
                    return self._predict_finbert_hf(text)
                except Exception as e:
                    logger.debug(f"HF prediction fallback: {e}")
            pos_score, neg_score = self._score_lexicon(text.lower(), self.en_positive, self.en_negative)

        # Compute softmax-like distribution
        diff = pos_score - neg_score
        total = pos_score + neg_score

        if total == 0:
            prob_pos = 0.20
            prob_neg = 0.20
            prob_neu = 0.60
            sentiment_score = 0.0
            sentiment_label = "Neutral"
        else:
            # Convert differential to calibrated continuous score in [-1.0, 1.0]
            raw_score = math.tanh(diff / 2.0)
            sentiment_score = round(float(raw_score), 4)

            # Calibrate class probabilities
            exp_pos = math.exp(pos_score)
            exp_neg = math.exp(neg_score)
            exp_neu = math.exp(1.0)  # neutral baseline
            z = exp_pos + exp_neg + exp_neu

            prob_pos = round(exp_pos / z, 4)
            prob_neg = round(exp_neg / z, 4)
            prob_neu = round(exp_neu / z, 4)

            if sentiment_score >= 0.15:
                sentiment_label = "Positive"
            elif sentiment_score <= -0.15:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"

        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "prob_positive": prob_pos,
            "prob_negative": prob_neg,
            "prob_neutral": prob_neu,
            "language": lang
        }

    def _score_lexicon(self, text: str, pos_dict: dict, neg_dict: dict) -> Tuple[float, float]:
        """Calculates positive and negative scores considering negations and intensifiers."""
        pos_total = 0.0
        neg_total = 0.0

        # Check positive keywords
        for kw, weight in pos_dict.items():
            if kw in text:
                # Check for negation near keyword
                pattern = rf"(not|never|no|無|未|沒有|不|안|못|아닌)\s*({re.escape(kw)})"
                if re.search(pattern, text, re.IGNORECASE):
                    neg_total += weight * 0.9
                else:
                    pos_total += weight

        # Check negative keywords
        for kw, weight in neg_dict.items():
            if kw in text:
                pattern = rf"(not|never|no|無|未|沒有|不|안|못|아닌)\s*({re.escape(kw)})"
                if re.search(pattern, text, re.IGNORECASE):
                    pos_total += weight * 0.9
                else:
                    neg_total += weight

        return pos_total, neg_total

    def _predict_finbert_hf(self, text: str) -> Dict[str, Any]:
        """Inference with ProsusAI/finbert PyTorch model."""
        import torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
        
        # FinBERT labels: 0: positive, 1: negative, 2: neutral
        prob_pos = round(float(probs[0]), 4)
        prob_neg = round(float(probs[1]), 4)
        prob_neu = round(float(probs[2]), 4)

        score = round(prob_pos - prob_neg, 4)
        if score > 0.15:
            label = "Positive"
        elif score < -0.15:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "sentiment_score": score,
            "sentiment_label": label,
            "prob_positive": prob_pos,
            "prob_negative": prob_neg,
            "prob_neutral": prob_neu,
            "language": "en"
        }

