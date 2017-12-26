import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tools import table_decorator
from sklearn import model_selection

pd.options.mode.chained_assignment = None


class Dataset:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def features(self): return preprocessing.scale(np.array(self.__feature_columns(self.frame())))

    def labels(self): return np.array(self.__labels_column(self.frame()))

    def head(self, size=5): return Dataset(self.frame().head(size))

    def tail(self, size=5): return Dataset(self.frame().tail(size))

    def frame(self): return self.data_frame

    def train_test_split(self, test_size=0.2):
        features = self.__feature_columns(self.frame())
        labels = self.__labels_column(self.frame())

        return model_selection.train_test_split(features, labels, test_size=test_size)

    def __labels_column(self, data_frame): return data_frame['Label']

    def __feature_columns(self, data_frame): return data_frame.drop(['Label'], 1)

    def __len__(self): return len(self.frame())

    def __str__(self): return table_decorator("Dataset head", self.frame().head())


class DatasetFactory:
    def create_from(self, data_frame, forecast_col="Close $", label_offset=0.01):
        df = data_frame[["adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]]

        df["H-L Change %"] = self.__percent_diff(df, "adj_high", "adj_low", total_column="adj_close")

        df["Price Change %"] = self.__normal_percent_diff(df, "adj_close", "adj_open")

        df = df.rename(columns={
            'adj_high': 'High $',
            'adj_low': 'Low $',
            'adj_close': 'Close $',
            'adj_open': 'Open $',
            'adj_volume': 'Trs'
        })
        df = df[["Close $", "H-L Change %", "Price Change %", "Trs"]]

        df.fillna(-99999, inplace=True)

        if label_offset > 0:
            forecast_out = int(math.ceil(label_offset * len(df)))
            df["Label"] = df[forecast_col].shift(-forecast_out)
        else:
            df["Label"] = df[forecast_col]

        df.dropna(inplace=True)

        return Dataset(df)

    def __percent_diff(self, df, column1, column2, total_column):
        return (df[column1] - df[column2]) / df[total_column] * 100

    def __normal_percent_diff(self, df, new_value, old_value):
        return self.__percent_diff(df, new_value, old_value, total_column=old_value)

