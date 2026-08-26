import pandas as pd
import os
from src.config import RAW_DATA_FILE

def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV file.

    Args:
        path: Optional path to override the default RAW_DATA_FILE.

    Returns:
        Raw pandas DataFrame.
    """
    load_path = path or RAW_DATA_FILE
    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"Data file not found at {load_path}. "
            "Please run 'python scripts/download_data.py' first."
        )
    
    print(f"[ETL] Loading data from {load_path} ...")
    df = pd.read_csv(load_path)
    print(f"[ETL] Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    return df
