"""Tests for ARIMA forecasting utilities."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_walmart_data
from src.forecast import (
    train_test_split_series,
    fit_auto_arima_forecast,
    evaluate_forecast
)


class TestTrainTestSplitSeries:
    """Test suite for train_test_split_series function."""
    
    @pytest.fixture
    def sample_series(self):
        """Create a sample time series for testing."""
        dates = pd.date_range('2020-01-01', periods=100, freq='W')
        values = np.random.rand(100) * 1000
        return pd.Series(values, index=dates)
    
    def test_train_test_split_default_split(self, sample_series):
        """Test that default split is 80/20."""
        train, test = train_test_split_series(sample_series)
        assert len(train) == 80
        assert len(test) == 20
    
    def test_train_test_split_custom_fraction(self, sample_series):
        """Test custom train/test split fraction."""
        train, test = train_test_split_series(sample_series, train_fraction=0.7)
        assert len(train) == 70
        assert len(test) == 30
    
    def test_train_test_split_no_overlap(self, sample_series):
        """Test that train and test sets don't overlap."""
        train, test = train_test_split_series(sample_series)
        overlap = set(train.index) & set(test.index)
        assert len(overlap) == 0
    
    def test_train_test_split_preserves_order(self, sample_series):
        """Test that the time series order is preserved."""
        train, test = train_test_split_series(sample_series)
        assert train.index[-1] < test.index[0]


class TestFitAutoARIMAForecast:
    """Test suite for fit_auto_arima_forecast function."""
    
    @pytest.fixture
    def walmart_series(self):
        """Load Walmart data and extract a single store's time series."""
        df = load_walmart_data()
        store_sales = df[df['Store'] == 1][['Date', 'Weekly_Sales']].set_index('Date')
        return store_sales['Weekly_Sales']
    
    def test_fit_auto_arima_returns_dict(self, walmart_series):
        """Test that fit_auto_arima_forecast returns a dictionary."""
        train, test = train_test_split_series(walmart_series)
        result = fit_auto_arima_forecast(train, test)
        assert isinstance(result, dict)
    
    def test_fit_auto_arima_has_required_keys(self, walmart_series):
        """Test that result dictionary has required keys."""
        train, test = train_test_split_series(walmart_series)
        result = fit_auto_arima_forecast(train, test)
        required_keys = {'model', 'forecast', 'train', 'test'}
        assert set(result.keys()) == required_keys
    
    def test_fit_auto_arima_forecast_shape(self, walmart_series):
        """Test that forecast DataFrame has correct shape."""
        train, test = train_test_split_series(walmart_series)
        result = fit_auto_arima_forecast(train, test)
        forecast_df = result['forecast']
        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) == len(test)
        assert 'Prediction' in forecast_df.columns
    
    def test_fit_auto_arima_forecast_index_alignment(self, walmart_series):
        """Test that forecast index aligns with test index."""
        train, test = train_test_split_series(walmart_series)
        result = fit_auto_arima_forecast(train, test)
        forecast_df = result['forecast']
        pd.testing.assert_index_equal(forecast_df.index, test.index)
    
    def test_fit_auto_arima_predictions_are_numeric(self, walmart_series):
        """Test that predictions are numeric values."""
        train, test = train_test_split_series(walmart_series)
        result = fit_auto_arima_forecast(train, test)
        forecast_values = result['forecast']['Prediction'].values
        assert np.all(np.isfinite(forecast_values))
        assert forecast_values.dtype in [np.float64, np.float32, float]


class TestEvaluateForecast:
    """Test suite for evaluate_forecast function."""
    
    @pytest.fixture
    def forecast_result(self):
        """Create a sample forecast result for testing."""
        df = load_walmart_data()
        store_sales = df[df['Store'] == 1][['Date', 'Weekly_Sales']].set_index('Date')
        series = store_sales['Weekly_Sales']
        train, test = train_test_split_series(series)
        return fit_auto_arima_forecast(train, test)
    
    def test_evaluate_forecast_returns_dict(self, forecast_result):
        """Test that evaluate_forecast returns a dictionary."""
        metrics = evaluate_forecast(forecast_result)
        assert isinstance(metrics, dict)
    
    def test_evaluate_forecast_has_required_metrics(self, forecast_result):
        """Test that metrics dictionary has required keys."""
        metrics = evaluate_forecast(forecast_result)
        required_keys = {'mean_actual', 'mean_forecast', 'mean_absolute_error', 
                        'root_mean_squared_error'}
        assert set(metrics.keys()) == required_keys
    
    def test_evaluate_forecast_metrics_are_numeric(self, forecast_result):
        """Test that all metrics are numeric values."""
        metrics = evaluate_forecast(forecast_result)
        for key, value in metrics.items():
            assert isinstance(value, (int, float, np.number))
            assert np.isfinite(value)
    
    def test_evaluate_forecast_metrics_positive(self, forecast_result):
        """Test that error metrics are positive."""
        metrics = evaluate_forecast(forecast_result)
        assert metrics['mean_absolute_error'] >= 0
        assert metrics['root_mean_squared_error'] >= 0
