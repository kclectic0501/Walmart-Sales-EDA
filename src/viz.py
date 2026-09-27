"""Visualization utilities for Walmart sales EDA."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_feature_over_time(df: pd.DataFrame, column: str, ylabel: str = None, 
                           title: str = None) -> None:
    """
    Plot the average value of a feature over time (across all stores).
    
    Generates a line plot showing how the mean value of a feature changes over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Date' column and the feature column.
    column : str
        Name of the column to plot.
    ylabel : str, optional
        Label for the y-axis. If None, uses the column name.
    title : str, optional
        Title for the plot. If None, uses "Avg {column} Values Over Time".
    """
    feature_over_time = df[column].groupby(df['Date']).mean()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=feature_over_time.index, y=feature_over_time)
    plt.xlabel('Date')
    plt.ylabel(ylabel or column)
    plt.title(title or f'Avg {column} Values Over Time')
    plt.tight_layout()
    plt.show()


def plot_feature_vs_sales_per_store(df: pd.DataFrame, column: str, title: str = None) -> None:
    """
    Scatter plot showing the relationship between a feature and sales for each store.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Date', 'Weekly_Sales', and the feature column.
    column : str
        Name of the column to plot against sales.
    title : str, optional
        Title for the plot. If None, uses "{column} vs Weekly_Sales".
    """
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Date', y=column, hue='Store', data=df, palette='tab20')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.title(title or f'{column} vs Weekly_Sales (Per Store)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    plt.show()


def plot_feature_correlation_with_sales(df: pd.DataFrame, column: str) -> float:
    """
    Calculate and display correlation between a feature and Weekly_Sales.
    
    Also prints the correlation coefficient.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Weekly_Sales' and the feature column.
    column : str
        Name of the column to correlate with sales.
    
    Returns
    -------
    float
        Pearson correlation coefficient between the feature and sales.
    """
    feature_values = df[column].values
    sales_values = df['Weekly_Sales'].values
    
    correlation = np.corrcoef(feature_values, sales_values)[0, 1]
    print(f"Correlation value between {column} and Sales is {round(correlation, 3)}.")
    
    return correlation
