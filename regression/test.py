
from tools import ObjectStorage
from datasources import DataSources
from dataset import Dataset, DatasetFactory
from sklearn.linear_model import LinearRegression
from matplotlib import style
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime


def show_graph(data_set, predictions):
    style.use('ggplot')

    df = pd.DataFrame(index=data_set.frame().index, data={'Real': data_set.labels(), 'Predicted': np.nan})

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for prediction in predictions:
        next_date = datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [prediction, np.nan]

    df['Real'].plot()
    df['Predicted'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


classifier = ObjectStorage().load(LinearRegression)
data_frame = DataSources().google_actions()
data_set = DatasetFactory().create_from(data_frame)

# Show real future prices + predicted prices...
show_graph(data_set, predictions=classifier.predict(data_set.test_features()))
