"""Data loading and preprocessing utilities for Walmart sales dataset."""

import pandas as pd
from pathlib import Path


def load_walmart_data(path: str = "data/walmart.csv") -> pd.DataFrame:
    """
    Load and preprocess the Walmart sales dataset.
    
    Reads the CSV file and converts the 'Date' column from string (DD-MM-YYYY format)
    to pandas datetime objects.
    
    Parameters
    ----------
    path : str
        Path to the walmart.csv file (default: "data/walmart.csv")
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Store, Date, Weekly_Sales, Holiday_Flag, 
        Temperature, Fuel_Price, CPI, Unemployment.
        Date column is converted to datetime type.
    
    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the specified path.
    """
    csv_path = Path(path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path.absolute()}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert Date column to datetime (format: DD-MM-YYYY)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    
    return df.copy()
