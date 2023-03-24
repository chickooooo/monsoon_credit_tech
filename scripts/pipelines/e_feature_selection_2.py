import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelectionPipeline2(BaseEstimator, TransformerMixin):

    # selected after RFE
    __features_to_drop = [
        'C3_3', 'C2_6', 'C3_2', 'C3_5', 'C2_999', 'C7_1', 'N21', 'C2_1', 'N2',
        'N14', 'C8', 'C2_7', 'C4_0', 'C7_999', 'C2_4', 'C7_6', 'C3_7', 'C2_2',
        'N16', 'C5_0', 'C4_12', 'C1_1', 'C3_1', 'C5_2', 'C7_2', 'C3_19',
        'C5_999', 'C4_999', 'C3_999', 'C7_0', 'C7_4', 'C4_31'
        ]

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.drop(self.__features_to_drop, axis=1)
