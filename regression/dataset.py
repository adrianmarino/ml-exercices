import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tools import table_decorator
from sklearn import model_selection

pd.options.mode.chained_assignment = None


class Dataset:
    def __init__(self, data_frame, features, test_features, labels):
        self.__data_frame = data_frame
        self.__features = features
        self.__test_features = test_features
        self.__labels = labels

    def test_features(self): return self.__test_features

    def features(self): return self.__features

    def labels(self): return self.__labels

    def frame(self): return self.__data_frame

    def train_test_split(self, test_size=0.2):
        return model_selection.train_test_split(self.features(), self.labels(), test_size=test_size)

    def __len__(self): return len(self.features())

    def __str__(self): return table_decorator("Dataset head", self.frame().head())


class DatasetFactory:
    def create_from(self, data_frame, label_column="close", closed_price_offset=0.01):
        df = self.__raw_data(data_frame)
        df = self.__feature_columns(df)

        # Fill null values with -999999...
        df.fillna(-99999, inplace=True)

        # Get offset of close prices...
        closed_price_offset_rows_count = int(math.ceil(closed_price_offset * len(df)))

        # Offset up close prices to predict future close prices...
        df["label"] = df[label_column].shift(-closed_price_offset_rows_count)

        # Get normalized features...
        features = preprocessing.scale(np.array(df.drop(["label"], 1)))

        # Get feature with predict...
        first_features = features[:-closed_price_offset_rows_count]

        # Get features without label close prices...
        last_features = features[-closed_price_offset_rows_count:]
        # Drop rows with na(null) values(last rows)...
        df.dropna(inplace=True)

        # Get labels corresponding to first features...
        labels = np.array(df["label"])

        print("Dataset:")
        print("* Rows: {} / Features: {} Labels: {}.".format(len(df), len(first_features), len(labels)))
        print("* Closed price offset: {} / Features to predict: {}."
              .format(closed_price_offset_rows_count, len(last_features)))

        return Dataset(df, first_features, last_features, labels)

    def __feature_columns(self, df):
        df["hl_change"] = self.__percent_diff(df, "high", "low", total_column="close")
        df["price_change"] = self.__normal_percent_diff(df, "close", "open")
        df = df[["close", "hl_change", "price_change", "volume"]]
        return df

    def __raw_data(self, data_frame):
        df = data_frame[["adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]]
        df = df.rename(columns={
            'adj_high': 'high',
            'adj_low': 'low',
            'adj_close': 'close',
            'adj_open': 'open',
            'adj_volume': 'volume'
        })
        return df

    def __percent_diff(self, df, column1, column2, total_column):
        return (df[column1] - df[column2]) / df[total_column] * 100

    def __normal_percent_diff(self, df, new_value, old_value):
        return self.__percent_diff(df, new_value, old_value, total_column=old_value)

