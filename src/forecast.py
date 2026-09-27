"""ARIMA forecasting utilities for Walmart sales prediction."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima.arima import auto_arima


def train_test_split_series(series: pd.Series, train_fraction: float = 0.8):
    """
    Split a time series into training and testing sets.
    
    Parameters
    ----------
    series : pd.Series
        Time series to split. Should have a datetime index.
    train_fraction : float
        Fraction of data to use for training (default: 0.8 for 80/20 split).
    
    Returns
    -------
    tuple of (pd.Series, pd.Series)
        Training set and testing set.
    """
    train_size = int(len(series) * train_fraction)
    train = series.iloc[:train_size]
    test = series.iloc[train_size:]
    return train, test


def fit_auto_arima_forecast(train: pd.Series, test: pd.Series, 
                            seasonal: bool = False, suppress_warnings: bool = True):
    """
    Fit an ARIMA model using pmdarima's auto_arima and generate forecasts.
    
    Uses automatic parameter selection (p, d, q) based on the training data patterns.
    Generates predictions for the test period.
    
    Parameters
    ----------
    train : pd.Series
        Training data (time series). Should have a datetime index.
    test : pd.Series
        Testing data (time series). Should have a datetime index.
    seasonal : bool
        Whether to fit a SARIMA model (seasonal ARIMA). Default: False.
    suppress_warnings : bool
        Whether to suppress convergence warnings. Default: True.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'model': Fitted ARIMA model object
        - 'forecast': pd.DataFrame with predictions indexed by test dates
        - 'train': Training set (returned for convenience)
        - 'test': Testing set (returned for convenience)
    
    Notes
    -----
    The auto_arima function automatically selects the best ARIMA parameters
    by testing various (p, d, q) combinations and selecting the one with the 
    lowest AIC (Akaike Information Criterion).
    """
    # Fit auto_arima model
    model = auto_arima(train, seasonal=seasonal, suppress_warnings=suppress_warnings)
    
    # Generate forecasts for test period
    forecast_values = model.predict(n_periods=len(test))
    forecast_df = pd.DataFrame(
        forecast_values,
        index=test.index,
        columns=['Prediction']
    )
    
    return {
        'model': model,
        'forecast': forecast_df,
        'train': train,
        'test': test
    }


def plot_arima_results(result: dict, title: str = "ARIMA Forecast Results") -> None:
    """
    Plot training data, test data, and ARIMA forecast.
    
    Parameters
    ----------
    result : dict
        Dictionary returned by fit_auto_arima_forecast() containing 
        'train', 'test', and 'forecast' keys.
    title : str
        Title for the plot.
    """
    train = result['train']
    test = result['test']
    forecast = result['forecast']
    
    # Plot 1: Train/Test split
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=train.index, y=train.values, label='Training Set')
    sns.lineplot(x=test.index, y=test.values, label='Testing Set')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.title(f'{title} - Train/Test Split')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Actual vs Predicted
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=test.index, y=test.values, label='Actual Sales', marker='o')
    sns.lineplot(x=forecast.index, y=forecast['Prediction'].values, 
                 label='ARIMA Forecast', marker='s', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.title(f'{title} - Forecast vs Actual')
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_forecast(result: dict) -> dict:
    """
    Calculate basic evaluation metrics for ARIMA forecast.
    
    Parameters
    ----------
    result : dict
        Dictionary returned by fit_auto_arima_forecast().
    
    Returns
    -------
    dict
        Dictionary with evaluation metrics:
        - 'mean_actual': Mean of actual test data
        - 'mean_forecast': Mean of forecast data
        - 'mean_absolute_error': Mean absolute error
        - 'root_mean_squared_error': RMSE
    """
    test = result['test']
    forecast = result['forecast']['Prediction'].values
    
    mae = np.mean(np.abs(test.values - forecast))
    rmse = np.sqrt(np.mean((test.values - forecast) ** 2))
    
    return {
        'mean_actual': np.mean(test.values),
        'mean_forecast': np.mean(forecast),
        'mean_absolute_error': mae,
        'root_mean_squared_error': rmse
    }
