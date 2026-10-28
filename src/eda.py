"""Quick first look at the raw dataset (shape, dtypes, summaries, class balance)."""
from src.data_cleaning import load_raw


def main():
    df = load_raw()
    print(df.shape)
    df.info()
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nNumerical Summary Statistics:")
    print(df.describe())
    print("\nCategorical Summary Statistics:")
    print(df.describe(exclude="number"))
    print(df["Churn"].value_counts())
    print((df["Churn"].value_counts(normalize=True) * 100).round(2))


if __name__ == "__main__":
    main()
