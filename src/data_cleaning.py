"""Load the raw Telco churn CSV, clean it, and save a processed copy."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "telco_churn_clean.csv"


def load_raw(path=RAW_PATH):
    return pd.read_csv(path)


def clean(df):
    df = df.copy()

    # TotalCharges is text with blank strings for customers who haven't been billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing = df["TotalCharges"].isna()
    assert (df.loc[missing, "tenure"] == 0).all(), "Missing TotalCharges for tenure > 0"
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # An ID carries no predictive signal.
    df = df.drop(columns=["customerID"])

    df["Churn"] = df["Churn"].str.strip().map({"Yes": 1, "No": 0})
    assert df["Churn"].notna().all(), "Unexpected values in Churn column"
    df["Churn"] = df["Churn"].astype(int)
    return df


def load_clean(path=CLEAN_PATH):
    """Load the processed data, building it from the raw CSV if needed."""
    if not Path(path).exists():
        main()
    return pd.read_csv(path)


def main():
    df = clean(load_raw())
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned data to {CLEAN_PATH.relative_to(ROOT)}: {df.shape}")


if __name__ == "__main__":
    main()
