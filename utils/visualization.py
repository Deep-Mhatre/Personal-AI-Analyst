"""
Visualization helpers using Plotly.
Each function returns a Plotly figure for Streamlit to display.
"""
import plotly.express as px
import pandas as pd
import numpy as np


def plot_histogram(df: pd.DataFrame, column: str, nbins: int = 30, max_categories: int = 50):
    """Return a histogram for numeric or categorical columns.

    For high-cardinality categoricals, collapse tails into 'Other' to avoid overplotting.
    """
    series = df[column]
    if series.dtype.name in ["object", "category"]:
        vc = series.astype("string").fillna("<NA>").value_counts()
        if len(vc) > max_categories:
            top = vc.nlargest(max_categories)
            top_index = set(top.index)
            series = series.astype("string").fillna("<NA>").map(lambda v: v if v in top_index else "Other")
        fig = px.histogram(series.to_frame(name=column), x=column, color=column, title=f"Distribution of {column}")
    else:
        fig = px.histogram(df, x=column, nbins=nbins, title=f"Distribution of {column}")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_bar(df: pd.DataFrame, column: str, top_n: int = 20):
    """Return a bar chart of value counts for the selected column."""
    counts = df[column].value_counts().nlargest(top_n).reset_index()
    counts.columns = [column, 'count']
    fig = px.bar(counts, x=column, y='count', title=f"Top {top_n} values in {column}")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_line(df: pd.DataFrame, column: str, index_col: str | None = None):
    """Create a line chart for a numeric column.

    - If index_col provided, attempt to parse it as datetime; otherwise use row index.
    - If the y column is non-numeric, try to coerce to numeric.
    """
    y = pd.to_numeric(df[column], errors="coerce")
    data = df.copy()
    data[column] = y
    data = data.dropna(subset=[column])
    if index_col and index_col in data.columns:
        x = data[index_col]
        # Try parsing datetime if not numeric
        if not np.issubdtype(x.dtype, np.number):
            x = pd.to_datetime(x, errors="coerce")
        data = data.assign(__x=x).dropna(subset=["__x"])
        fig = px.line(data, x="__x", y=column, title=f"{column} over {index_col}")
    else:
        # default: show values in order
        fig = px.line(data.reset_index(), x="index", y=column, title=f"{column} over rows")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None):
    """Return a scatter plot for two numeric columns.

    - If x/y are not numeric, attempt to coerce to numeric and drop NA rows.
    - Optional color grouping for a categorical column.
    """
    data = df.copy()
    data[x] = pd.to_numeric(data[x], errors="coerce")
    data[y] = pd.to_numeric(data[y], errors="coerce")
    data = data.dropna(subset=[x, y])
    if color and color in data.columns:
        fig = px.scatter(data, x=x, y=y, color=color, title=f"{y} vs {x}")
    else:
        fig = px.scatter(data, x=x, y=y, title=f"{y} vs {x}")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig
