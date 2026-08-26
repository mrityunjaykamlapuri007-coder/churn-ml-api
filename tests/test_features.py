"""
tests/test_features.py — Unit tests for feature engineering
Run: pytest tests/ -v
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_raw_data
from src.data.preprocessor import clean_data
from src.features.engineer import add_features


class TestFeatureEngineer:
    """Tests for the feature engineering pipeline."""

    @pytest.fixture
    def raw_features(self):
        """Load and clean data for testing."""
        df = load_raw_data()
        X, y = clean_data(df)
        return X

    def test_add_features_returns_dataframe(self, raw_features):
        """add_features should return a DataFrame."""
        result = add_features(raw_features)
        assert isinstance(result, pd.DataFrame)

    def test_derived_features_exist(self, raw_features):
        """Derived features should be present after engineering."""
        X = add_features(raw_features)
        expected = ["avg_monthly_spend", "is_new_customer", "high_monthly_charge"]
        for col in expected:
            assert col in X.columns, f"Missing feature: {col}"

    def test_no_object_columns_remain(self, raw_features):
        """All categorical columns should be encoded (no object dtype)."""
        X = add_features(raw_features)
        object_cols = X.select_dtypes(include="object").columns.tolist()
        assert len(object_cols) == 0, f"Object columns remain: {object_cols}"

    def test_no_nulls_after_features(self, raw_features):
        """No null values after feature engineering."""
        X = add_features(raw_features)
        assert X.isnull().sum().sum() == 0
