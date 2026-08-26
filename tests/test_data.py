"""
tests/test_data.py — Unit tests for data loading & preprocessing
Run: pytest tests/ -v
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_raw_data
from src.data.preprocessor import clean_data


class TestDataLoader:
    """Tests for the ETL Extract layer."""

    def test_load_returns_dataframe(self):
        """Data loader should return a pandas DataFrame."""
        df = load_raw_data()
        assert isinstance(df, pd.DataFrame)

    def test_load_has_expected_columns(self):
        """Raw data should contain the target column 'Churn'."""
        df = load_raw_data()
        assert "Churn" in df.columns

    def test_load_non_empty(self):
        """Raw data should not be empty."""
        df = load_raw_data()
        assert len(df) > 0


class TestPreprocessor:
    """Tests for the ETL Transform layer."""

    def test_clean_returns_tuple(self):
        """clean_data should return (X, y) tuple."""
        df = load_raw_data()
        result = clean_data(df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_clean_removes_customer_id(self):
        """customerID column should be dropped after cleaning."""
        df = load_raw_data()
        X, y = clean_data(df)
        assert "customerID" not in X.columns

    def test_clean_target_is_binary(self):
        """Target should only contain 0 and 1."""
        df = load_raw_data()
        X, y = clean_data(df)
        assert set(y.unique()).issubset({0, 1})

    def test_clean_no_nulls(self):
        """No null values should remain after cleaning."""
        df = load_raw_data()
        X, y = clean_data(df)
        assert X.isnull().sum().sum() == 0
        assert y.isnull().sum() == 0
