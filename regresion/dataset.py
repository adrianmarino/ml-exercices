import math


class DatasetFactory:
    def createFrom(self, raw_dataset, forecast_col="Close $", label_offset=0.01):
        print("Build dataset...")
        df = raw_dataset
        df = df[["adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]]

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

        forecast_out = int(math.ceil(label_offset * len(df)))
        df["label"] = df[forecast_col].shift(-forecast_out)

        df.dropna(inplace=True)

        return df

    def __percent_diff(self, df, column1, column2, total_column):
        return (df[column1] - df[column2]) / df[total_column] * 100

    def __normal_percent_diff(self, df, new_value, old_value):
        return self.__percent_diff(df, new_value, old_value, total_column=old_value)
