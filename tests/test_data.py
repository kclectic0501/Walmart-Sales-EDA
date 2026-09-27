"""Tests for data loading and preprocessing."""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_walmart_data


class TestLoadWalmartData:
    """Test suite for load_walmart_data function."""
    
    def test_load_walmart_data_returns_dataframe(self):
        """Test that load_walmart_data returns a pandas DataFrame."""
        df = load_walmart_data()
        assert isinstance(df, pd.DataFrame)
    
    def test_load_walmart_data_has_expected_columns(self):
        """Test that the loaded DataFrame has all expected columns."""
        df = load_walmart_data()
        expected_columns = {'Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 
                           'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'}
        assert set(df.columns) == expected_columns
    
    def test_load_walmart_data_date_is_datetime(self):
        """Test that Date column is converted to datetime type."""
        df = load_walmart_data()
        assert pd.api.types.is_datetime64_any_dtype(df['Date'])
    
    def test_load_walmart_data_no_null_values(self):
        """Test that the dataset has no null values."""
        df = load_walmart_data()
        assert df.isnull().sum().sum() == 0
    
    def test_load_walmart_data_row_count(self):
        """Test that the dataset has the expected number of rows."""
        df = load_walmart_data()
        # The Walmart dataset should have 6,435 rows (45 stores × 143 weeks)
        assert len(df) == 6435
    
    def test_load_walmart_data_store_count(self):
        """Test that there are 45 unique stores in the dataset."""
        df = load_walmart_data()
        assert df['Store'].nunique() == 45
    
    def test_load_walmart_data_weekly_sales_positive(self):
        """Test that all Weekly_Sales values are positive."""
        df = load_walmart_data()
        assert (df['Weekly_Sales'] > 0).all()
    
    def test_load_walmart_data_numeric_types(self):
        """Test that numeric columns have appropriate data types."""
        df = load_walmart_data()
        numeric_cols = ['Store', 'Weekly_Sales', 'Holiday_Flag', 'Temperature', 
                       'Fuel_Price', 'CPI', 'Unemployment']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} is not numeric"
    
    def test_load_walmart_data_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_walmart_data(path="nonexistent/path/to/data.csv")
    
    def test_load_walmart_data_returns_copy(self):
        """Test that load_walmart_data returns a copy, not a reference."""
        df1 = load_walmart_data()
        df2 = load_walmart_data()
        # Modify one dataframe
        df1.loc[0, 'Weekly_Sales'] = 999999
        # Verify the other is unaffected
        assert df2.loc[0, 'Weekly_Sales'] != 999999
