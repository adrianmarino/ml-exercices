from classifier import Classifier
from datasources import DataSources
from tools import table_decorator, ObjectStorage, first
from dataset import Dataset, DatasetFactory
from sklearn.linear_model import LinearRegression
import pandas as pd

algorithms = [LinearRegression(n_jobs=-1)]


def classifiers(): return list(map(lambda alg: Classifier(alg), algorithms))


def confidence_table(rows):
    content = pd.DataFrame(list(map(lambda row: (row[0], row[1] * 100), rows)), columns=['Algorithm', 'Confidence (%)'])
    return table_decorator("Results", content)


data_frame = DataSources().google_actions()
data_set = DatasetFactory().create_from(data_frame)
classifiers = classifiers()

train_info = map(lambda clr: [clr.name(), clr.train(data_set), ], classifiers)
print(confidence_table(train_info))

ObjectStorage().save(first(classifiers))
print("...trained data saved!")
