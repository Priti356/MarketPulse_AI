"""
Tests for Machine Learning Training, Evaluation, and Prediction.
"""

import pytest
import os
import numpy as np
import pandas as pd

from src.processing.feature_engineering import FeatureEngineer
from src.prediction.train_model import ModelTrainer, BaselineModel
from src.prediction.evaluate_model import ModelEvaluator
from src.prediction.predict_price import PricePredictor
from src.database.db_manager import DatabaseManager


@pytest.fixture
def dummy_feature_df():
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=60, freq="B")
    feat_cols = FeatureEngineer.get_feature_columns()
    data = {col: np.random.normal(0, 0.02, size=60) for col in feat_cols}
    data["Close"] = [100.0 + i * 0.5 for i in range(60)]
    data["target_next_return"] = np.random.normal(0.001, 0.015, size=60)
    df = pd.DataFrame(data, index=dates)
    return df


def test_chronological_split(tmp_path, dummy_feature_df):
    trainer = ModelTrainer(models_dir=str(tmp_path))
    train, val, test = trainer.chronological_split(dummy_feature_df, train_ratio=0.70, val_ratio=0.15)

    assert len(train) > len(val)
    assert len(val) == len(test)
    # Check strict chronological order
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_model_evaluation_metrics():
    evaluator = ModelEvaluator()
    y_true = np.array([0.02, -0.01, 0.03, -0.02])
    y_pred = np.array([0.01, -0.015, 0.02, -0.01])
    base_prices = np.array([100.0, 100.0, 100.0, 100.0])

    metrics = evaluator.calculate_metrics(y_true, y_pred, base_prices)
    assert metrics["price_mae"] > 0
    assert metrics["directional_accuracy"] == 100.0  # all signs match!


def test_end_to_end_prediction(tmp_path, dummy_feature_df):
    db = DatabaseManager(db_url=f"sqlite:///{tmp_path}/pred_test.db")
    predictor = PricePredictor(models_dir=str(tmp_path), db_manager=db)
    res = predictor.predict_next_price("0700.HK", dummy_feature_df)

    assert "predicted_price" in res
    assert "current_price" in res
    assert "model_name" in res
    assert res["predicted_price"] > 0

