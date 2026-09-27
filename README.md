# Walmart Sales EDA and Forecasting

![CI](https://github.com/kclectic0501/Walmart-Sales-EDA/actions/workflows/ci.yml/badge.svg)

**Exploratory Data Analysis and Time Series Forecasting for Walmart Sales** — by Krishna Chaitanya N

## Project Overview

Comprehensive **EDA and ARIMA time series forecasting** on Walmart's weekly sales across 45 stores (2010-2012). Demonstrates data cleaning, statistical analysis, visualization, and sales prediction.

## Key Objectives

- Establish relationships between attributes and their impact on sales
- Conduct store-wise comparative analysis and identify performance patterns  
- Build ARIMA time series models for accurate sales forecasting
- Generate actionable insights for business improvement

## Repository Structure

```
Walmart-Sales-EDA/
├── data/
│   └── walmart.csv                 # Raw dataset (45 stores, 143 weeks)
├── src/                            # Reusable Python modules
│   ├── data.py                     # Data loading & preprocessing
│   ├── viz.py                      # Visualization utilities
│   └── forecast.py                 # ARIMA forecasting functions
├── tests/                          # Unit tests (pytest)
│   ├── test_data.py
│   └── test_forecast.py
├── notebooks/
│   ├── eda_sales_prediction.ipynb  # Main interactive analysis
│   └── eda_sales_prediction_original.ipynb
├── .github/workflows/ci.yml        # GitHub Actions CI
├── requirements.txt
├── LICENSE                         # MIT
└── README.md
```

## Dataset

**Source:** [Kaggle - Walmart Sales](https://www.kaggle.com/datasets/walmart-inc/walmart-recruiting-sales-in-retail-stores)  
**License:** [CDLA Permissive 1.0](https://cdla.io/permissive-1-0/)  
**Volume:** 6,435 records (45 stores × 143 weeks, 2010-2012)

### Features

| Column | Type | Description |
|--------|------|-------------|
| **Store** | Integer | Store ID (1-45) |
| **Date** | DateTime | Week ending date |
| **Weekly_Sales** | Float | Total weekly sales |
| **Holiday_Flag** | Binary | Holiday in week (0/1) |
| **Temperature** | Float | Avg temperature (°F) |
| **Fuel_Price** | Float | Fuel price (USD/gal) |
| **CPI** | Float | Consumer Price Index |
| **Unemployment** | Float | Unemployment rate (%) |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/kclectic0501/Walmart-Sales-EDA.git
cd Walmart-Sales-EDA

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run the Analysis

```bash
# Open interactive notebook
jupyter notebook notebooks/eda_sales_prediction.ipynb

# Run tests
pytest tests/ -v
```

## Architecture

The project uses a **modular design** for reusability and testability:

- **`src/data.py`** — Data loading with automatic datetime parsing
- **`src/viz.py`** — Parametrized plotting functions to eliminate code duplication
- **`src/forecast.py`** — ARIMA model fitting, prediction, and evaluation
- **`tests/`** — Comprehensive unit tests with pytest
- **`.github/workflows/ci.yml`** — Automated testing on every push

The notebook (`eda_sales_prediction.ipynb`) imports these modules and focuses on narrative and analysis.

## Key Findings

- **Store Performance:** Store 20 significantly outperforms others (~8x sales of Store 33)
- **Feature Impact:** Holiday periods, temperature, and unemployment each influence sales differently
- **Seasonality:** Sales exhibit clear seasonal patterns and trends
- **Forecasting:** ARIMA models effectively capture historical patterns for short-term predictions

## Development & Testing

### Run Tests

```bash
pytest tests/ -v --tb=short
```

### Add New Tests

Create test functions in `tests/` following pytest conventions. Tests verify:
- Data loading correctness (columns, types, row counts)
- Forecasting function outputs (shapes, indices, numeric validity)
- Error handling (missing files, invalid parameters)

### Update Dependencies

```bash
pip install -r requirements.txt --upgrade
```

## CI/CD

This project uses **GitHub Actions** to automatically:
- Run the test suite on Python 3.9, 3.10, 3.11
- Execute the notebook end-to-end to catch regressions
- Validate all code changes before merging

See `.github/workflows/ci.yml` for details.

## License

- **Code:** MIT License (see `LICENSE`)
- **Dataset:** CDLA Permissive 1.0

## Author

**Krishna Chaitanya N**

## Acknowledgments

- Dataset source: [Kaggle](https://www.kaggle.com)
- ARIMA methodology: Investopedia, Wikipedia
- Libraries: pandas, matplotlib, seaborn, statsmodels, pmdarima, pytest
