import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelectionPipeline(BaseEstimator, TransformerMixin):

    __informational = ['Unique_ID']
    __high_missing_values = ['N32', 'N27', 'N26', 'N25', 'N31', 'N30', 'N29', 'N28']
    __high_correlation = ['N6', 'N5', 'N20', 'N34']

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        features_to_drop = self.__informational + self.__high_missing_values + self.__high_correlation
        return X.drop(features_to_drop, axis=1)
