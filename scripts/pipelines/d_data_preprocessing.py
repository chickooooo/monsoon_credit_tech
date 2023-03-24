import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler


class DataPreprocessingPipeline(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.__scaler = RobustScaler()

    def fit(self, X: pd.DataFrame, y=None):
        # will hold features
        bool_features = []
        cat_features = []
        num_features = []

        # populating features
        for feature in X.columns:
            if 'C' in feature:
                if X[feature].dtype == 'bool':
                    bool_features.append(feature)
                else:
                    cat_features.append(feature)
            elif 'N' in feature:
                num_features.append(feature)

        # adding to class
        self.__bool_features = bool_features
        self.__cat_features = cat_features
        self.__num_features = num_features

        # fitting robust scaler
        self.__scaler.fit(X[num_features])

        return self

    def __encode_bool_data(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.applymap(lambda x: 1 if x else 0)

    def __encode_cat_data(self, X: pd.DataFrame) -> pd.DataFrame:
        cat_data = X.astype('string')

        return pd.get_dummies(cat_data)

    def __scale_num_data(self, X: pd.DataFrame) -> pd.DataFrame:
        num_data = self.__scaler.transform(X)
        # converting to dataframe
        num_data = pd.DataFrame(
            data=num_data, columns=X.columns, index=X.index)

        return num_data

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:

        # encoding and scaling data
        bool_encoded = self.__encode_bool_data(X[self.__bool_features])
        cat_encoded = self.__encode_cat_data(X[self.__cat_features])
        num_scaled = self.__scale_num_data(X[self.__num_features])

        total_data = pd.concat([bool_encoded, cat_encoded, num_scaled], axis=1)

        # if training data
        if 'Dependent_Variable' in X.columns:
            # saving column order
            self.__column_order = list(total_data.columns)
            total_data = pd.concat(
                [total_data, X[['Dependent_Variable']]], axis=1)
            
        # if test data
        else:
            # making sure all columns are present
            # and in given order
            for column in self.__column_order:
                if column not in total_data.columns:
                    total_data[column] = 0

            total_data = total_data[self.__column_order]

        return total_data
