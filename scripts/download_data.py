#!/usr/bin/env python
"""
scripts/download_data.py — Download dataset from Kaggle
Usage:
    python scripts/download_data.py
"""
import os
import shutil


def main():
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    os.makedirs(data_dir, exist_ok=True)

    target = os.path.join(data_dir, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    if os.path.exists(target):
        print(f"Data already exists: {target}")
        return

    try:
        import kagglehub

        print("Downloading Telco Customer Churn dataset from Kaggle ...")
        path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        print(f"  Downloaded to: {path}")

        # Copy CSV to data/
        for f in os.listdir(path):
            if f.endswith(".csv"):
                src = os.path.join(path, f)
                shutil.copy2(src, target)
                print(f"  Copied to: {target}")
                return

        print("Error: No CSV found in download directory.")

    except ImportError:
        print("Error: kagglehub not installed. Run: pip install kagglehub")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
