import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def describe_dataframe(df):
    nunique = df.nunique()

    def infer_feature_type(col):
        if pd.api.types.is_bool_dtype(col):
            return "Boolean"
        if pd.api.types.is_datetime64_any_dtype(col):
            return "Datetime"
        if col.dtype == "object" or nunique[col.name] < 20:
            return "Categorical"
        return "Numerical"

    summary = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing Percentage": df.isnull().mean(),
        "Distinct Values": nunique,
        "Mode": [
            col.mode().iloc[0] if not col.mode().empty else None
            for _, col in df.items()
        ],
        "Data Type": df.dtypes,
        "Feature Type": [infer_feature_type(df[col]) for col in df.columns]
    })

    return summary

def categorize_values(val):
    if pd.isnull(val):
        return 2  # NaN
    if isinstance(val, (int, float)) and val == 0:
        return 1  # Zero
    return 0      # Other

def plot_missing_values(df, show_zeros=False):
    plt.figure(figsize=(max(10, len(df.columns) * 0.5), 6))

    if show_zeros:
        heatmap_data = df.map(categorize_values)

        cmap = ListedColormap([
            "#440154",  # Other (purple)
            "#FDE725",  # Zero (yellow)
            "#FF8000"   # NaN (orange)
        ])

        sns.heatmap(heatmap_data, cbar=False, cmap=cmap)
        plt.title("Orange = NaN | Yellow = 0 | Purple = Other")

    else:
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")

    plt.xlabel("Features")
    plt.ylabel("Observations")
    plt.tight_layout()
    plt.show()

def plot_binary_target_with_stats(df, target_col, class_labels):
    counts = df.value_counts().sort_index()
    total = counts.sum()
    
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=counts.index, 
                     y=counts.values, 
                     hue=counts.index.astype(str),
                     palette="tab10"
                    )

    # Add counts and percentages on top of each bar with dynamic spacing
    max_height = counts.max()
    for i, val in enumerate(counts.values):
        percent = val / total * 100
        label = f"{val}\n({percent:.2f}%)"
        # offset = 2% of the tallest bar
        offset = max_height * 0.02
        ax.text(i, val + offset, label, ha='center', va='bottom', fontsize=10)

    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(f"Target Distribution: {target_col}")
    plt.ylim(0, max_height * 1.15)  # give extra space at top
    plt.tight_layout()
    plt.show()

def show_feature_distributions(
    df,
    columns,
    n_cols,
    cat_max_unique,
    exclude_cols=None
):
    if exclude_cols is None:
        exclude_cols = []

    plots = []

    for col in columns:
        if col in exclude_cols:
            continue

        nunique = df[col].nunique(dropna=True)

        # Skip high-cardinality categoricals
        if df[col].dtype == 'object' and nunique > cat_max_unique:
            continue

        plots.append(col)

    n_plots = len(plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, plots):

        nunique = df[col].nunique(dropna=True)

        if df[col].dtype == 'object' or nunique <= cat_max_unique:
            counts = df[col].value_counts().sort_index()

            bars = sns.barplot(
                x=counts.index.astype(str),
                y=counts.values,
                hue=counts.index.astype(str),
                palette="tab10",
                legend=False,
                ax=ax
            )

            # Add values on top of bars
            for bar, value in zip(bars.patches, counts.values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  # center x
                    height + 0.5,                       # slightly above bar
                    f"{value}",                          # integer count
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

            ax.set_ylabel("Count")
            ax.tick_params(axis='x')

        else:
            ax.hist(df[col], bins=50)
            ax.set_ylabel("Count")

        ax.set_title(col)
        ax.set_xlabel(col)

    # Remove unused axes
    for ax in axes[len(plots):]:
        ax.remove()

    plt.tight_layout()
    plt.show()

def plot_survival_rate_categorical(
    X,
    y,
    cat_max_unique=20,
    exclude_cols=None,
    n_cols=3
):
    """
    X : pd.DataFrame (features)
    y : pd.Series or pd.DataFrame (binary target)
    """

    if exclude_cols is None:
        exclude_cols = []

    # Convert target to Series if DataFrame
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # Align indices (important!)
    data = X.join(y.rename("target"), how="inner")

    cat_cols = [
        col for col in X.columns
        if col not in exclude_cols
        and (X[col].dtype == "object" or X[col].nunique() <= cat_max_unique)
    ]

    n_plots = len(cat_cols)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, cat_cols):
        rates = data.groupby(col, observed=True)["target"].mean()

        bars = sns.barplot(
            x=rates.index.astype(str),
            y=rates.values,
            hue=rates.index.astype(str),
            palette="tab10",
            ax=ax
        )

        # Add values on top of bars
        max_height = rates.max()
        for bar, value in zip(bars.patches, rates.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_height * 0.03,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        ax.set_title(f"Survival rate by {col}")
        ax.set_ylabel("Survival probability")
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="x", rotation=0)

    # Remove unused axes
    for ax in axes[len(cat_cols):]:
        ax.remove()

    plt.tight_layout()
    plt.show()