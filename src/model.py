import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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