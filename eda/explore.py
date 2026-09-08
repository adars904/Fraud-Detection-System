"""
EDA (Exploratory Data Analysis) for the IEEE-CIS fraud detection dataset.

Loads train_transaction.csv and train_identity.csv, merges them, and
provides functions to inspect shape, nulls, duplicates, and key
categorical relationships with the target (isFraud).
"""

import pandas as pd


def load_and_merge_data(
    transaction_path="data/train_transaction.csv",
    identity_path="data/train_identity.csv"
):
    """
    Load the transaction and identity datasets and merge them on
    TransactionID (left join, since not every transaction has identity
    data).

    Returns
    -------
    df1 : pd.DataFrame   raw transaction data
    df2 : pd.DataFrame   raw identity data
    df3 : pd.DataFrame   merged dataset (df1 left-joined with df2)
    """
    df1 = pd.read_csv(transaction_path)
    df2 = pd.read_csv(identity_path)
    df3 = pd.merge(df1, df2, on="TransactionID", how="left")
    return df1, df2, df3


def eda_identity(df2, verbose=True):
    """EDA on train_identity.csv (shape, info, describe, nulls, duplicates)."""
    if not verbose:
        return
    print("shape:", df2.shape)
    print(df2.info())
    print(df2.describe())
    print(df2.isnull().sum())
    print("duplicates:", df2.duplicated().sum())


def eda_transaction(df1, verbose=True):
    """EDA on train_transaction.csv (describe, info, nulls, duplicates,
    dist2 stats, numerical/categorical column split, target/category
    distributions and crosstabs vs isFraud)."""
    if not verbose:
        return

    print(df1.describe())
    print(df1.info())

    print(df1["dist2"].mean())
    print(df1["dist2"].median())
    print(df1["dist2"].max())

    print(df1.isnull().sum())
    print("duplicates:", df1.duplicated().sum())

    numerical_features = df1.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = df1.select_dtypes(include=["object"]).columns
    print("Numerical Features:", numerical_features)
    print("Categorical Features:", categorical_features)

    print(df1["isFraud"].value_counts())
    print(df1["ProductCD"].value_counts())
    print(df1["card4"].value_counts())
    print(df1["card6"].value_counts())

    print(pd.crosstab(df1["ProductCD"], df1["isFraud"], normalize="index") * 100)
    print(pd.crosstab(df1["card4"], df1["isFraud"], normalize="index") * 100)
    print(pd.crosstab(df1["card6"], df1["isFraud"], normalize="index") * 100)


def eda_merged(df3, verbose=True):
    """EDA on the merged dataset (shape, describe, info, nulls, duplicates)."""
    if not verbose:
        return
    print("merged shape:", df3.shape)
    print(df3.describe())
    print(df3.info())
    print(df3.isnull().sum())
    print("duplicates:", df3.duplicated().sum())


def run_eda(df1, df2, df3, verbose=True):
    """Run all EDA steps in sequence (transaction, identity, merged)."""
    eda_transaction(df1, verbose=verbose)
    eda_identity(df2, verbose=verbose)
    eda_merged(df3, verbose=verbose)


if __name__ == "__main__":
    df1, df2, df3 = load_and_merge_data()
    run_eda(df1, df2, df3, verbose=True)