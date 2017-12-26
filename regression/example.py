from classifier import Classifier
from datasources import DataSources
from tools import table_decorator
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

# -----------------------------------------------------------------------------
# Main program...
# -----------------------------------------------------------------------------
# Params...
test_size = 0.4
label_offset = 0.01

# Prepare...
data_frame = DataSources().google_actions()
data_set = DatasetFactory().create_from(data_frame=data_frame, label_offset=label_offset)
classifiers = classifiers()

# Perform...
results = map(lambda clr: [clr.name(), clr.train(data_set, test_size), ], classifiers)
print(confidence_table(results))

_, test_features, _, real_test_labels = data_set.train_test_split(test_size=test_size)
show_graph(real_test_labels, predicted_labels=classifiers[0].predict(test_features))
