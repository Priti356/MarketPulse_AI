"""
Model Evaluation Module for MarketPulse AI.
Computes MAE, RMSE, and Directional Accuracy on returns and prices.
"""

import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Any


class ModelEvaluator:
    """Evaluates financial prediction models on test partitions."""

    @staticmethod
    def calculate_metrics(
        y_true_returns: np.ndarray,
        y_pred_returns: np.ndarray,
        base_prices: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculates:
          - return_mae: MAE on percentage returns
          - return_rmse: RMSE on percentage returns
          - price_mae: MAE on converted predicted prices
          - price_rmse: RMSE on converted predicted prices
          - directional_accuracy: % where sign(pred) == sign(true)
        """
        if len(y_true_returns) == 0:
            return {
                "return_mae": 0.0,
                "return_rmse": 0.0,
                "price_mae": 0.0,
                "price_rmse": 0.0,
                "directional_accuracy": 50.0
            }

        # Return errors
        ret_mae = float(mean_absolute_error(y_true_returns, y_pred_returns))
        ret_rmse = float(math.sqrt(mean_squared_error(y_true_returns, y_pred_returns)))

        # Price conversion: P_{t+1} = P_t * (1 + R_{t+1})
        true_prices = base_prices * (1.0 + y_true_returns)
        pred_prices = base_prices * (1.0 + y_pred_returns)

        price_mae = float(mean_absolute_error(true_prices, pred_prices))
        price_rmse = float(math.sqrt(mean_squared_error(true_prices, pred_prices)))

        # Directional Accuracy: hit rate of return sign
        # Positive return -> 1, Negative return -> -1, Zero -> 0
        true_dir = np.sign(y_true_returns)
        pred_dir = np.sign(y_pred_returns)
        correct_directions = np.sum(true_dir == pred_dir)
        dir_acc = float((correct_directions / len(y_true_returns)) * 100.0)

        return {
            "return_mae": round(ret_mae, 5),
            "return_rmse": round(ret_rmse, 5),
            "price_mae": round(price_mae, 2),
            "price_rmse": round(price_rmse, 2),
            "directional_accuracy": round(dir_acc, 2)
        }

