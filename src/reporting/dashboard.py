"""
Dashboard visual helpers and Plotly chart generators for MarketPulse AI.
"""

from typing import Dict, Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class DashboardVisualizer:
    """Provides reusable Plotly chart components for the MarketPulse AI Streamlit application."""

    DARK_PAPER = "#111827"
    DARK_PLOT = "#111827"
    TEXT_COLOR = "#cbd5e1"

    @classmethod
    def plot_price_history_with_pivots(
        cls,
        price_df: pd.DataFrame,
        support: float,
        resistance: float,
        predicted_price: float,
        ticker: str
    ) -> go.Figure:
        """Interactive Candlestick chart with Floor Support (S1) and Resistance (R1)."""
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=price_df.index,
            open=price_df["Open"],
            high=price_df["High"],
            low=price_df["Low"],
            close=price_df["Close"],
            name=f"{ticker} OHLC"
        ))

        # Add Support Line
        if support > 0:
            fig.add_hline(
                y=support,
                line_dash="dash",
                line_color="#38bdf8",
                annotation_text=f"Support: {support:,.2f}",
                annotation_position="bottom right"
            )

        # Add Resistance Line
        if resistance > 0:
            fig.add_hline(
                y=resistance,
                line_dash="dash",
                line_color="#f472b6",
                annotation_text=f"Resistance: {resistance:,.2f}",
                annotation_position="top right"
            )

        fig.update_layout(
            paper_bgcolor=cls.DARK_PAPER,
            plot_bgcolor=cls.DARK_PLOT,
            font=dict(color=cls.TEXT_COLOR),
            height=450,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_rangeslider_visible=False
        )
        return fig

    @classmethod
    def plot_sentiment_breakdown(cls, pos_pct: float, neu_pct: float, neg_pct: float) -> go.Figure:
        """Donut chart for sentiment distribution."""
        labels = ["Positive", "Neutral", "Negative"]
        values = [pos_pct, neu_pct, neg_pct]
        colors = ["#22c55e", "#64748b", "#ef4444"]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker=dict(colors=colors)
        )])
        fig.update_layout(
            paper_bgcolor=cls.DARK_PAPER,
            font=dict(color=cls.TEXT_COLOR),
            margin=dict(l=10, r=10, t=10, b=10),
            height=260
        )
        return fig

