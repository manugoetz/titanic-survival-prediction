import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

def get_famtype(df):
    df["FamSize"] = df["SibSp"] + df["Parch"] + 1
    df["FamType"] = pd.cut(
        df["FamSize"],
        bins=[0, 1, 4, 12],
        labels=["Solo", "Small", "Large"]
    )
    
    return df

def extract_title(df):
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    # df["Title"] = df["Title"].replace(
    #     [
    #         "Lady", "the Countess", "Capt", "Col", "Don", "Dr",
    #         "Major", "Rev", "Sir", "Jonkheer", "Dona"
    #     ],
    #     "Rare"
    # )
    
    df["Title"] = df["Title"].replace(
        {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    )

    df["Title"] = df["Title"].replace(
        [
            "Capt", "Col", "Major", "Dr", "Rev"
        ],
        "Officer"
    )
    df["Title"] = df["Title"].replace(
        [
            "Jonkheer", "Don", "Sir", "the Countess", "Dona", "Lady"
        ],
        "Royalty"
    )
    
    df.drop("Name", axis=1, inplace=True)
    
    return df

def extract_deck(df, cabin_col="Cabin"):
    """
    Extracts deck letter from the Cabin column.
    Missing or NaN cabins are labeled as 'Unknown'.
    """
    
    # Extract first letter if not null, else 'Unknown'
    df["Deck"] = df[cabin_col].apply(lambda x: str(x)[0] if pd.notnull(x) else "Unknown")
    df.drop("Cabin", axis=1, inplace=True)
    
    return df

class GroupMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_cols, target_col="Age"):
        self.group_cols = group_cols
        self.target_col = target_col

    def fit(self, X, y=None):
        X = X.copy()
        self.group_medians_ = (
            X.groupby(self.group_cols)[self.target_col]
            .median()
        )
        self.global_median_ = X[self.target_col].median()
        return self

    def transform(self, X):
        X = X.copy()

        def fill_age(row):
            if pd.isna(row[self.target_col]):
                key = tuple(row[col] for col in self.group_cols)
                return self.group_medians_.get(key, self.global_median_)
            return row[self.target_col]

        X[self.target_col] = X.apply(fill_age, axis=1)
        return X[[self.target_col]].to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array([self.target_col])