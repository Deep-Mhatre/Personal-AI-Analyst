"""
Utilities for data cleaning.
Each function is small and testable. They operate on pandas DataFrame objects.
"""
import pandas as pd
import numpy as np
from typing import Tuple


def detect_types(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with column, dtype, num_missing, percent_missing, unique_values.

    Inputs:
        df: pandas DataFrame
    Outputs:
        pandas DataFrame with metadata about each column
    """
    cols = []
    for c in df.columns:
        vals = df[c]
        cols.append({
            "column": c,
            "dtype": str(vals.dtype),
            "num_missing": int(vals.isna().sum()),
            "percent_missing": float(vals.isna().mean() * 100),
            "unique_values": int(vals.nunique(dropna=True))
        })
    return pd.DataFrame(cols)


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate rows from df. Returns (cleaned_df, num_removed)."""
    before = len(df)
    cleaned = df.drop_duplicates()
    after = len(cleaned)
    return cleaned, before - after


def handle_nulls(df: pd.DataFrame, strategy: str = "leave") -> pd.DataFrame:
    """Handle nulls according to strategy.

    Strategies:
    - leave: do nothing
    - drop_rows: drop any row with null
    - fill_mean: fill numeric columns with mean
    - fill_median: fill numeric columns with median
    - fill_mode: fill with mode for each column
    """
    if strategy == "leave":
        return df
    df = df.copy()
    if strategy == "drop_rows":
        return df.dropna()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == "fill_mean":
        for c in numeric_cols:
            df[c] = df[c].fillna(df[c].mean())
    elif strategy == "fill_median":
        for c in numeric_cols:
            df[c] = df[c].fillna(df[c].median())
    elif strategy == "fill_mode":
        for c in df.columns:
            try:
                mode = df[c].mode()
                if not mode.empty:
                    df[c] = df[c].fillna(mode.iloc[0])
            except Exception:
                continue
    return df


def clean_dataframe(df: pd.DataFrame, null_strategy: str = "leave", drop_duplicates: bool = True) -> pd.DataFrame:
    """Run a sequence of cleaning steps and return cleaned df.

    Steps:
    - Basic type inference is left to pandas
    - Remove duplicates if requested
    - Handle nulls according to chosen strategy
    """
    if drop_duplicates:
        # use the remove_duplicates function defined above
        df, _removed = remove_duplicates(df)
    df = handle_nulls(df, strategy=null_strategy)
    # You can add more cleaning steps like type coercion here
    return df
