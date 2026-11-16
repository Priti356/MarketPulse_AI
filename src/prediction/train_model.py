"""
Model Training Module for MarketPulse AI.
Trains Random Forest, Gradient Boosting, and Baseline models using chronological splits.
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from src.processing.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class BaselineModel:
    """Naive Baseline model forecasting rolling average return or zero drift."""

    def __init__(self, window: int = 5):
        self.window = window
        self.rolling_mean = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(y) > 0:
            self.rolling_mean = float(np.mean(y[-self.window:]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(shape=(len(X),), fill_value=self.rolling_mean)


class ModelTrainer:
    """Manages walk-forward / chronological splits and model persistence."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.feature_cols = FeatureEngineer.get_feature_columns()

    def chronological_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset chronologically without shuffling.
        Default: 70% Train, 15% Validation, 15% Test.
        """
        valid_df = df.dropna(subset=["target_next_return"]).copy()
        n = len(valid_df)
        if n < 20:
            # For minimal datasets, use 60/20/20 or fallback
            train_end = int(n * 0.6)
            val_end = int(n * 0.8)
        else:
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

        train_data = valid_df.iloc[:train_end]
        val_data = valid_df.iloc[train_end:val_end]
        test_data = valid_df.iloc[val_end:]

        logger.info(f"Chronological split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        return train_data, val_data, test_data

    def train_models_for_asset(
        self,
        ticker: str,
        feature_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Train Random Forest, Gradient Boosting, and Baseline models for next-session return prediction.
        Selects best model on validation set and saves to disk.
        """
        train_df, val_df, test_df = self.chronological_split(feature_df)

        if train_df.empty or test_df.empty:
            logger.warning(f"Insufficient history to train ML models for {ticker}.")
            return {}

        X_train = train_df[self.feature_cols].values
        y_train = train_df["target_next_return"].values

        X_val = val_df[self.feature_cols].values
        y_val = val_df["target_next_return"].values

        X_test = test_df[self.feature_cols].values
        y_test = test_df["target_next_return"].values

        # 1. Random Forest Regressor
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=4,
            random_state=42
        )
        rf.fit(X_train, y_train)

        # 2. Gradient Boosting Regressor
        gb = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
        gb.fit(X_train, y_train)

        # 3. Naive Baseline
        baseline = BaselineModel(window=5)
        baseline.fit(X_train, y_train)

        # Evaluate on validation set
        val_eval = {
            "RandomForest": mean_squared_error(y_val, rf.predict(X_val)),
            "GradientBoosting": mean_squared_error(y_val, gb.predict(X_val)),
            "Baseline": mean_squared_error(y_val, baseline.predict(X_val))
        }

        # Select best model (lowest validation MSE)
        best_name = min(val_eval, key=val_eval.get)
        models_map = {
            "RandomForest": rf,
            "GradientBoosting": gb,
            "Baseline": baseline
        }
        best_model = models_map[best_name]

        # Evaluate on test set
        from src.prediction.evaluate_model import ModelEvaluator
        evaluator = ModelEvaluator()
        test_metrics = {}
        for m_name, mdl in models_map.items():
            preds = mdl.predict(X_test)
            test_metrics[m_name] = evaluator.calculate_metrics(
                y_true_returns=y_test,
                y_pred_returns=preds,
                base_prices=test_df["Close"].values
            )

        # Save best model to disk
        safe_ticker = ticker.replace(".", "_")
        model_path = os.path.join(self.models_dir, f"model_{safe_ticker}.joblib")
        joblib.dump({
            "ticker": ticker,
            "best_model_name": best_name,
            "model": best_model,
            "feature_cols": self.feature_cols,
            "val_eval": val_eval,
            "test_metrics": test_metrics
        }, model_path)
        logger.info(f"Saved best model ({best_name}) for {ticker} to {model_path}")

        return {
            "ticker": ticker,
            "best_model_name": best_name,
            "model_path": model_path,
            "test_metrics": test_metrics,
            "test_df": test_df,
            "val_df": val_df
        }

