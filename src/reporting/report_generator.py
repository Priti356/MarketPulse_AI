"""
Report Generator for MarketPulse AI.
Produces downloadable CSV and executive PDF intelligence reports.
"""

import os
import io
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates analytical reports in CSV and PDF formats."""

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_csv_report(self, ticker_intelligence: List[Dict[str, Any]]) -> str:
        """
        Generate CSV ranking report.
        Returns file path of the saved CSV.
        """
        if not ticker_intelligence:
            return ""

        rows = []
        for item in ticker_intelligence:
            rows.append({
                "Ticker": item["ticker"],
                "Asset Name": item["asset_name"],
                "Exchange": item["exchange"],
                "Latest Price": item["latest_price"],
                "Predicted Next Price": item.get("predicted_price", 0.0),
                "Expected Return (%)": round(((item.get("predicted_price", 0.0) - item["latest_price"]) / item["latest_price"]) * 100.0, 2) if item["latest_price"] > 0 else 0.0,
                "Market Direction": item.get("sentiment_direction", "Neutral"),
                "Investor Position": item.get("investor_position", "Not stated"),
                "Support Level": item.get("support_level", 0.0),
                "Resistance Level": item.get("resistance_level", 0.0),
                "Total Social Mentions": item.get("total_mentions", 0),
                "Sentiment Score (-1 to 1)": item.get("sentiment_score", 0.0),
                "Positive (%)": item.get("pos_pct", 0.0),
                "Negative (%)": item.get("neg_pct", 0.0),
                "Neutral (%)": item.get("neu_pct", 0.0),
            })

        df = pd.DataFrame(rows)
        # Rank by total mentions and sentiment score
        df = df.sort_values(by=["Total Social Mentions", "Sentiment Score (-1 to 1)"], ascending=[False, False])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.reports_dir, f"marketpulse_intelligence_{timestamp}.csv")
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        logger.info(f"Generated CSV report: {file_path}")
        return file_path

    def generate_pdf_report(self, ticker_intelligence: List[Dict[str, Any]]) -> str:
        """
        Generate executive PDF intelligence report using ReportLab.
        Returns file path of the saved PDF.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.reports_dir, f"marketpulse_report_{timestamp}.pdf")

        doc = SimpleDocTemplate(
            file_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        # Custom dark-inspired clean styling
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1e1b4b"),
            spaceAfter=6
        )
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#4338ca"),
            spaceAfter=15
        )
        h2_style = ParagraphStyle(
            "Heading2Custom",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#1e1b4b"),
            spaceBefore=12,
            spaceAfter=8
        )
        body_style = ParagraphStyle(
            "BodyCustom",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#1f2937")
        )
        table_cell_style = ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#111827")
        )
        table_header_style = ParagraphStyle(
            "TableHeader",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            fontName="Helvetica-Bold",
            textColor=colors.white
        )

        elements = []

        # Header Title
        elements.append(Paragraph("MarketPulse AI Executive Intelligence Report", title_style))
        report_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        elements.append(Paragraph(f"Cross-Exchange Sentiment & Price Forecast Analysis | Generated: {report_time_str}", subtitle_style))
        elements.append(Spacer(1, 10))

        # Executive Summary Section
        total_assets = len(ticker_intelligence)
        total_mentions = sum(i.get("total_mentions", 0) for i in ticker_intelligence)
        avg_sent = round(sum(i.get("sentiment_score", 0.0) for i in ticker_intelligence) / total_assets, 3) if total_assets > 0 else 0.0

        summary_text = (
            f"This automated financial intelligence dossier encapsulates multi-source sentiment analysis and "
            f"machine learning price projections across <b>{total_assets} major equities</b> listed on the "
            f"Hong Kong Stock Exchange (HKEX) and Korea Exchange (KRX). Across the tracked window, "
            f"<b>{total_mentions:,} social media discussions</b> were processed from LIHKG, Naver Finance, and Telegram, "
            f"synthesized alongside verified institutional news via Exa AI. The cross-market average sentiment score is "
            f"<b>{avg_sent:+.3f}</b> on a scale of [-1.0, +1.0]."
        )
        elements.append(Paragraph(summary_text, body_style))
        elements.append(Spacer(1, 15))

        # Main Table: Cross-Market Overview
        elements.append(Paragraph("1. Asset Ranking and Next-Session Predictions", h2_style))

        table_data = [[
            Paragraph("Ticker", table_header_style),
            Paragraph("Asset Name", table_header_style),
            Paragraph("Exch.", table_header_style),
            Paragraph("Last Price", table_header_style),
            Paragraph("Pred. Price", table_header_style),
            Paragraph("Exp. %", table_header_style),
            Paragraph("Direction", table_header_style),
            Paragraph("Position", table_header_style),
            Paragraph("Support", table_header_style),
            Paragraph("Resistance", table_header_style),
            Paragraph("Mentions", table_header_style),
            Paragraph("Sentiment", table_header_style),
        ]]

        for item in ticker_intelligence:
            last_p = item["latest_price"]
            pred_p = item.get("predicted_price", last_p)
            exp_pct = ((pred_p - last_p) / last_p) * 100.0 if last_p > 0 else 0.0

            table_data.append([
                Paragraph(item["ticker"], table_cell_style),
                Paragraph(item["asset_name"][:14], table_cell_style),
                Paragraph(item["exchange"], table_cell_style),
                Paragraph(f"{last_p:,.1f}", table_cell_style),
                Paragraph(f"{pred_p:,.1f}", table_cell_style),
                Paragraph(f"{exp_pct:+.1f}%", table_cell_style),
                Paragraph(item.get("sentiment_direction", "Neutral"), table_cell_style),
                Paragraph(item.get("investor_position", "Not stated")[:10], table_cell_style),
                Paragraph(f"{item.get('support_level', 0):,.1f}", table_cell_style),
                Paragraph(f"{item.get('resistance_level', 0):,.1f}", table_cell_style),
                Paragraph(str(item.get("total_mentions", 0)), table_cell_style),
                Paragraph(f"{item.get('sentiment_score', 0):+.2f}", table_cell_style),
            ])

        # Style table
        t = Table(table_data, colWidths=[48, 62, 34, 46, 46, 38, 48, 48, 44, 46, 38, 42])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#312e81")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.HexColor("#f1f5f9")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 15))

        # Deep-Dive Section on Ticker Intelligence
        elements.append(Paragraph("2. Detailed Ticker Intelligence & Community Consensus", h2_style))
        for item in ticker_intelligence[:5]:
            detail_p = (
                f"<b>{item['ticker']} ({item['asset_name']}) - {item['exchange']}</b><br/>"
                f"• Current Price: {item['latest_price']:,.2f} | Predicted: {item.get('predicted_price', 0):,.2f}<br/>"
                f"• Structural Pivot Levels: Support = {item.get('support_level', 0):,.2f} | Resistance = {item.get('resistance_level', 0):,.2f}<br/>"
                f"• Community Position: <b>{item.get('investor_position', 'Not stated')}</b> "
                f"<i>({item.get('position_quote', 'No explicit directive')})</i><br/>"
                f"• Sentiment Breakdown: Positive {item.get('pos_pct', 0)}% | Neutral {item.get('neu_pct', 0)}% | Negative {item.get('neg_pct', 0)}%"
            )
            elements.append(Paragraph(detail_p, body_style))
            elements.append(Spacer(1, 6))

        # Disclaimer
        elements.append(Spacer(1, 10))
        disclaimer = (
            "<b>Disclaimer:</b> MarketPulse AI produces algorithmic research outputs using natural language processing "
            "and statistical machine learning. Financial price projections and sentiment scores do not constitute investment "
            "advice or fiduciary solicitation."
        )
        elements.append(Paragraph(disclaimer, ParagraphStyle("Disc", parent=body_style, fontSize=7, leading=9, textColor=colors.gray)))

        doc.build(elements)
        logger.info(f"Generated PDF report: {file_path}")
        return file_path

