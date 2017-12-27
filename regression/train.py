from classifier import Classifier
from datasources import DataSources
from tools import table_decorator, ObjectStorage, first
from dataset import Dataset, DatasetFactory
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

algorithms = [LinearRegression(n_jobs=-1)]


def classifiers(): return list(map(lambda alg: Classifier(alg), algorithms))


def confidence_table(rows):
    content = pd.DataFrame(list(map(lambda row: (row[0], row[1] * 100), rows)), columns=['Algorithm', 'Confidence (%)'])
    return table_decorator("Results", content)


def show_graph(real_labels, predicted_labels):
    df = pd.DataFrame(data={'Real': real_labels, 'Predicted': predicted_labels})
    graph = df.plot(kind='area', title="Google Action Price", stacked=False)
    graph.set_xlabel("Time")
    graph.set_ylabel("US$")
    plt.show()


data_frame = DataSources().google_actions()
data_set = DatasetFactory().create_from(data_frame)
classifiers = classifiers()

results = map(lambda clr: [clr.name(), clr.train(data_set), ], classifiers)
print(confidence_table(results))

ObjectStorage().save(first(classifiers))

