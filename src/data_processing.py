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
        "Zero Count": (df == 0).sum(numeric_only=False),
        "Zero Percentage": (df == 0).mean(numeric_only=False),
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

def plot_binary_target_with_stats(df, target_col):
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


def detect_outliers(df, col, group_cols=None, show=False):
    """
    Detect outliers in a numerical column using IQR.
    If group_cols is provided, detect outliers within each group.
    
    Returns a DataFrame of outliers.
    """
    if group_cols:
        outliers_list = []
        for group_vals, group_df in df.groupby(group_cols):
            Q1 = group_df[col].quantile(0.25)
            Q3 = group_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5*IQR
            upper_bound = Q3 + 1.5*IQR

            group_outliers = group_df[(group_df[col] < lower_bound) | (group_df[col] > upper_bound)]
            if show:
                print(f"\n{len(group_outliers)} outliers in {col} for {dict(zip(group_cols, group_vals))}:")
                print(group_outliers[[col] + list(group_cols)])
            outliers_list.append(group_outliers)
        return pd.concat(outliers_list)
    else:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if show:
            print(f"{len(outliers)} outliers in {col}:")
            print(outliers[[col]])
        return outliers

    
def plot_feature_survival_hist(
    data,
    feature,
    percent=True,
    bins=20,
    height=5,
    aspect=1,
    palette={0: "red", 1: "green"},
    title=None
):
    if feature not in data.columns:
        raise ValueError(f"'{feature}' not found in dataframe")

    multiple = "fill" if percent else "stack"
    y_label = "Proportion" if percent else "Count"

    # 🔹 Auto title
    if title is None:
        base_title = f"{feature} Distribution by Survival and Sex"
    else:
        base_title = title

    full_title = f"{base_title} ({'Percent' if percent else 'Count'})"

    g = sns.displot(
        data=data,
        x=feature,
        hue="Survived",
        col="Sex",
        multiple=multiple,
        bins=bins,
        palette=palette,
        height=height,
        aspect=aspect,
        common_norm=not percent
    )

    g.set_axis_labels(feature, y_label)
    g.set_titles("Sex: {col_name}")
    g._legend.set_title("Survived")

    # 🔹 Global title
    g.fig.suptitle(full_title, fontsize=16)
    g.fig.subplots_adjust(top=0.85)

    plt.show()

def plot_hist_count_and_percent(
    df,
    x,
    hue="Survived",
    bins=20,
    palette={0: "red", 1: "green"},
    title_prefix=""
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    # -------- Count plot --------
    sns.histplot(
        data=df,
        x=x,
        hue=hue,
        multiple="stack",
        bins=bins,
        palette=palette,
        alpha=0.7,
        ax=axes[0]
    )
    axes[0].set_title(f"{title_prefix}{x} Distribution by {hue} (Count)")
    axes[0].set_xlabel(x)
    axes[0].set_ylabel("Count")

    # -------- Percent plot --------
    sns.histplot(
        data=df,
        x=x,
        hue=hue,
        multiple="fill",     # percent per bin
        bins=bins,
        palette=palette,
        alpha=0.7,
        ax=axes[1]
    )
    axes[1].set_title(f"{title_prefix}{x} Distribution by {hue} (Percent)")
    axes[1].set_xlabel(x)
    axes[1].set_ylabel("Proportion")

    # Global title
    fig.suptitle(
        f"{x} vs {hue}: Count and Percentage Distribution",
        fontsize=16,
        y=1.05
    )

    plt.tight_layout()
    plt.show()

def plot_cat_feature_survival(df, target="Survived", bins_col="FareBinned"):
    """
    Plots two subplots for FareBinned:
    1. Count per bin
    2. Survival rate per bin
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1️⃣ Count per bin
    counts = df[bins_col].value_counts().sort_index()
    bars = sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        hue=counts.index.astype(str),
        palette="tab10",
        ax=axes[0]
    )
    axes[0].set_title(f"Passenger count per {bins_col}")
    axes[0].set_xlabel(bins_col)
    axes[0].set_ylabel("Count")
    for bar, value in zip(bars.patches, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value}",
            ha="center",
            va="bottom"
        )

    # 2️⃣ Survival rate per bin
    rates = df.groupby(bins_col, observed=True)[target].mean().sort_index()
    bars = sns.barplot(
        x=rates.index.astype(str),
        y=rates.values,
        hue=counts.index.astype(str),
        palette="tab10",
        ax=axes[1]
    )
    axes[1].set_title(f"Survival rate per {bins_col}")
    axes[1].set_xlabel(bins_col)
    axes[1].set_ylabel("Survival probability")
    axes[1].set_ylim(0, 1.0)
    for bar, value in zip(bars.patches, rates.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.show()