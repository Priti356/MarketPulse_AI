# MarketPulse AI 📈
### Multi-Source Financial Sentiment Analysis & Next-Session Stock Price Prediction

MarketPulse AI is a Python-based financial intelligence project that combines stock market data, sentiment analysis, technical indicators, and machine learning to analyze 10 equities listed on the Hong Kong Stock Exchange (HKEX) and Korea Exchange (KRX).

The project provides an interactive Streamlit dashboard, stock-level analysis, next-session price forecasts, and downloadable CSV and PDF reports.

## 🚀 Key Features

- **Multi-Exchange Analysis:** Tracks 10 equities across HKEX and KRX.
- **Stock Intelligence:** Displays stock prices, predicted prices, expected returns, support and resistance levels, and investor positioning.
- **Sentiment Analysis:** Analyzes sentiment scores, positive/negative/neutral classifications, and social mention volumes.
- **Machine Learning Prediction:** Uses a Gradient Boosting model for next-session price forecasting.
- **Financial News:** Displays company-related financial news and article links.
- **Interactive Dashboard:** Provides seven sections for exploring market intelligence.
- **Report Generation:** Generates downloadable CSV datasets and PDF reports.

## 🌏 Exchange & Asset Coverage

**Hong Kong Stock Exchange (HKEX)**

| Company | Ticker |
|---|---|
| Tencent Holdings | 0700.HK |
| Alibaba Group | 9988.HK |
| HSBC Holdings | 0005.HK |
| AIA Group | 1299.HK |
| Meituan | 3690.HK |

**Korea Exchange (KRX)**

| Company | Ticker |
|---|---|
| Samsung Electronics | 005930.KS |
| SK Hynix | 000660.KS |
| NAVER | 035420.KS |
| Hyundai Motor | 005380.KS |
| Kakao | 035720.KS |

Asset details and search keywords are configured in `config/assets.yaml`.

## 🧠 Machine Learning & Analysis

The project includes:

- Gradient Boosting-based next-session price forecasting.
- Price prediction evaluation using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Directional accuracy measurement.
- Sentiment score analysis.
- Technical indicators and support/resistance calculations.
- Investor positioning classification.

### Model Results

The dashboard displayed the following model evaluation results during the November 17, 2026 run:

| Evaluation Metric | Displayed Result |
|---|---:|
| Selected Model | GradientBoosting |
| Directional Accuracy | 45.0% |
| Test Price MAE | 10.06 |
| Test Price RMSE | 12.63 |

*These are application-reported evaluation results. The exact asset and test-set scope, baseline comparison, and independent reproducibility should be verified before interpreting them as evidence of predictive performance.*

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Financial Market Data | yfinance |
| Dashboard | Streamlit |
| Data Visualization | Plotly |
| Database | SQLite, SQLAlchemy |
| Reporting | ReportLab, CSV |
| Configuration | YAML, Python dotenv |
| Testing | Pytest |

## 📊 Dashboard Sections

MarketPulse AI provides seven dashboard sections:

1. **Overview:** Executive market metrics and asset rankings.
2. **Exchange Analysis:** Comparative analysis of HKEX and KRX equities.
3. **Stock Intelligence:** Individual stock analysis, predictions, and technical levels.
4. **Sentiment Analysis:** Sentiment distribution and social mention analytics.
5. **Price Prediction:** Machine learning forecasts and evaluation metrics.
6. **Financial News:** Company-related financial news and article links.
7. **Reports & Exports:** CSV and PDF report generation.

## 🏗️ Project Structure

```text
MarketPulse/
│
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── pytest.ini
│
├── config/
│   └── assets.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── marketpulse.db
│
├── models/
├── reports/
│
├── src/
│   ├── data_collection/
│   ├── sentiment/
│   ├── processing/
│   ├── prediction/
│   ├── intelligence/
│   ├── reporting/
│   └── database/
│
└── tests/
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Priti356/MarketPulse_AI.git
cd MarketPulse_AI
```

### 2. Create a Python virtual environment

```bash
py -3.11 -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file using `.env.example` as a template.

```env
APIFY_TOKEN=
EXA_API_KEY=
DATABASE_URL=sqlite:///data/marketpulse.db
```

Apify and Exa credentials are optional for the current fixture-based fallback setup. Add valid API credentials if you want to configure those integrations.

**Important:** Never upload your `.env` file or API credentials to GitHub.

## ▶️ Running the Project

### Run the complete pipeline

```bash
python run_pipeline.py
```

The pipeline processes the configured assets and generates CSV and PDF reports.

### Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal. It is usually:

`http://localhost:8501`

## 📁 Generated Reports

Reports are saved in the `reports/` directory.

The project generates:

- **CSV Report:** Structured stock intelligence data, including prices, predictions, sentiment scores, and technical levels.
- **PDF Report:** A formatted financial intelligence report containing stock analysis and rankings.

Example output filenames:

```text
reports/
├── marketpulse_intelligence_YYYYMMDD_HHMMSS.csv
└── marketpulse_report_YYYYMMDD_HHMMSS.pdf
```

## 📌 Execution Results

During the November 17, 2026 execution:

- Successfully processed all 10 configured assets.
- Initialized the SQLite database.
- Generated stock intelligence data.
- Generated a CSV report.
- Generated a PDF report.
- Launched the Streamlit dashboard successfully.

## ⚠️ Data Sources & Limitations

- **Market prices:** The project includes Yahoo Finance-based market data collection. Availability and freshness depend on the data provider.
- **Social-media data:** Apify credentials were not configured during the documented run. The scraper used seed fixtures instead of authenticated live actor collection.
- **Financial news:** Exa API credentials were not configured during the documented run. News results should not be represented as verified live Exa retrieval.
- **Predictions:** Machine learning forecasts are experimental and should not be treated as financial advice.
- **Evaluation:** Model performance can vary across assets and test periods.

## 🔮 Future Improvements

- Configure and validate live Apify integrations.
- Configure live Exa financial news retrieval.
- Improve model evaluation and compare against baseline models.
- Add more robust multilingual financial sentiment analysis.
- Improve prediction accuracy through additional feature engineering.
- Add automated model monitoring and retraining.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**GitHub Repository:** [Priti356/MarketPulse_AI](https://github.com/Priti356/MarketPulse_AI)

