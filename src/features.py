"""Feature engineering. All features are row-wise, so they can't leak test information."""
import numpy as np
import pandas as pd

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "num_services", "avg_monthly_spend"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "tenure_group",
]


def add_features(df):
    df = df.copy()
    # include_lowest so tenure == 0 lands in the first bucket instead of NaN.
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, np.inf],
        labels=["0-12", "12-24", "24-48", "48+"], include_lowest=True,
    ).astype(str)
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    # +1 avoids dividing by zero for brand-new customers (tenure == 0).
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    return df
