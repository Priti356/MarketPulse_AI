"""
Price Predictor for MarketPulse AI.
Loads trained models, forecasts next-session return, and computes predicted prices.
"""

import os
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

from src.processing.feature_engineering import FeatureEngineer
from src.prediction.train_model import ModelTrainer
from src.database.db_manager import DatabaseManager, ModelPrediction

logger = logging.getLogger(__name__)


class PricePredictor:
    """Predicts next-session price for HKEX and KRX assets."""

    def __init__(self, models_dir: str = "models", db_manager: Optional[DatabaseManager] = None):
        self.models_dir = models_dir
        self.db = db_manager or DatabaseManager()
        self.trainer = ModelTrainer(models_dir=models_dir)

    def predict_next_price(
        self,
        ticker: str,
        feature_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Forecast next trading session's price for ticker.
        If a saved model exists, load it; otherwise train.
        """
        if feature_df.empty:
            raise ValueError(f"Feature DataFrame for {ticker} is empty.")

        safe_ticker = ticker.replace(".", "_")
        model_path = os.path.join(self.models_dir, f"model_{safe_ticker}.joblib")

        model_payload = None
        if os.path.exists(model_path):
            try:
                model_payload = joblib.load(model_path)
            except Exception as e:
                logger.warning(f"Could not load saved model from {model_path}: {e}")

        if not model_payload:
            logger.info(f"Training fresh models for {ticker}...")
            train_res = self.trainer.train_models_for_asset(ticker, feature_df)
            if os.path.exists(model_path):
                model_payload = joblib.load(model_path)
            else:
                raise RuntimeError(f"Failed to train and save model for {ticker}")

        model = model_payload["model"]
        feature_cols = model_payload.get("feature_cols", FeatureEngineer.get_feature_columns())
        best_model_name = model_payload.get("best_model_name", "RandomForest")
        test_metrics = model_payload.get("test_metrics", {}).get(best_model_name, {})

        # Latest session feature row
        latest_row = feature_df.iloc[-1]
        current_price = float(latest_row["Close"])
        latest_features = latest_row[feature_cols].values.reshape(1, -1)

        # Predict next-session percentage return
        predicted_return = float(model.predict(latest_features)[0])

        # Clip excessive prediction return outliers to [-15%, +15%]
        predicted_return = float(np.clip(predicted_return, -0.15, 0.15))

        # Convert to predicted price: P_{t+1} = P_t * (1 + R_{t+1})
        predicted_price = current_price * (1.0 + predicted_return)

        # Format depending on price magnitude (KRW vs HKD)
        is_krw = current_price > 1000.0
        decimals = 0 if is_krw else 2
        predicted_price = round(predicted_price, decimals)

        expected_change_pct = round(predicted_return * 100.0, 2)

        # Store prediction in database
        try:
            session = self.db.get_session()
            pred_record = ModelPrediction(
                ticker=ticker,
                prediction_date=pd.to_datetime(latest_row.name).to_pydatetime() if hasattr(latest_row.name, "to_pydatetime") else latest_row.name,
                model_name=best_model_name,
                current_price=current_price,
                predicted_return=predicted_return,
                predicted_price=predicted_price,
                mae=test_metrics.get("price_mae"),
                rmse=test_metrics.get("price_rmse"),
                directional_accuracy=test_metrics.get("directional_accuracy")
            )
            session.add(pred_record)
            session.commit()
            session.close()
        except Exception as e:
            logger.warning(f"Could not record prediction to DB: {e}")

        return {
            "ticker": ticker,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_return": round(predicted_return, 4),
            "expected_change_pct": expected_change_pct,
            "model_name": best_model_name,
            "test_metrics": test_metrics,
            "test_metrics_all": model_payload.get("test_metrics", {})
        }

