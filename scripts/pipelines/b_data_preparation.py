import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer


class DataPreparationPipeline(BaseEstimator, TransformerMixin):

    # cardinality of cat features
    __keep_values_dict = {
        'C1': 3,
        'C2': 7,
        'C3': 7,
        'C4': 5,
        'C5': 4,
        'C6': 2,
        'C7': 5,
        'C8': 2
    }

    # outlier limit of num features
    __outlier_limit = {
        'N1': 45, 'N2': 530, 'N3': 4.3, 'N4': 40, 'N7': 95,
        'N8': 33, 'N9': 45, 'N10': 4500, 'N10.1': 32,
        'N11': 80, 'N12': 100_000, 'N14': 100, 'N15': 13,
        'N16': 2, 'N17': 450_000, 'N18': 1.8, 'N19': 260_000,
        'N21': 1, 'N22': 10, 'N23': 2100, 'N24': 50_000,
        'N33': 500, 'N35': 54,
    }

    # skewed numerical features
    __skewed_features = [
        'N2',
        'N9',
        'N17',
        'N19',
    ]

    def __init__(self, imputation: str = 'knn'):
        self.__imputation = imputation
        if imputation == 'knn':
            self.__imputer = KNNImputer(n_neighbors=5)

    def fit(self, X: pd.DataFrame, y=None):
        aggr_dict = {}

        # cat and num features
        self.__cat_features = [
            feature for feature in X.columns if 'C' in feature]
        self.__num_features = [
            feature for feature in X.columns if 'N' in feature]

        # aggregate for features
        for feature in X.columns:
            if 'C' in feature:
                aggr_dict[feature] = X[feature].mode()[0]
            elif 'N' in feature:
                aggr_dict[feature] = X[feature].median()

        if self.__imputation == 'knn':
            # fit KNN imputer
            self.__imputer.fit(X[self.__num_features])

        self.__aggr_dict = aggr_dict

        return self

    def __reduce_cardinality(self, X: pd.Series) -> pd.Series:
        top_n = self.__keep_values_dict[X.name]
        values_to_keep = X.value_counts()[:top_n].index
        return X.apply(lambda x: x if x in values_to_keep else 999)

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        data = X.copy()

        # filling missing values in cat features
        for feature in self.__cat_features:
            data[feature].fillna(value=self.__aggr_dict[feature], inplace=True)

        # reducing cat features cardinality
        for feature in self.__cat_features:
            data[feature] = self.__reduce_cardinality(data[feature])

        # replacing outliers with na of num features
        for feature in self.__num_features:
            data[feature] = data[feature].apply(
                lambda x: x if x <= self.__outlier_limit[feature] else np.nan)

        # filling missing values in num features
        if self.__imputation == 'knn':

            # transforming using KNN imputer
            imputed_data = self.__imputer.transform(data[self.__num_features])
            # updating data
            data[self.__num_features] = pd.DataFrame(
                imputed_data, columns=data[self.__num_features].columns, index=data[self.__num_features].index)

        elif self.__imputation == 'aggr':
            # filling using aggregate
            for feature in self.__num_features:
                data[feature].fillna(
                    value=self.__aggr_dict[feature], inplace=True)

        # applying log(1+x) to skewed num features
        for feature in self.__skewed_features:
            data[feature] = np.log1p(data[feature])

        return data
