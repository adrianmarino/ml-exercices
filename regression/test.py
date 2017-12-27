
from tools import ObjectStorage
from datasources import DataSources
from dataset import Dataset, DatasetFactory
from sklearn.linear_model import LinearRegression
from matplotlib import style
import pandas as pd
import matplotlib.pyplot as plt


def show_graph(real_labels, predicted_labels):
    style.use('ggplot')
    df = pd.DataFrame(data={'Real': real_labels, 'Predicted': predicted_labels})
    graph = df.plot(kind='area', title="Google Action Price", stacked=False)
    graph.set_xlabel("Time")
    graph.set_ylabel("US$")
    plt.show()


classifier = ObjectStorage().load(LinearRegression)

data_frame = DataSources().google_actions()
data_set = DatasetFactory().create_from(data_frame)
_, test_features, _, real_test_labels = data_set.train_test_split()

show_graph(real_test_labels, predicted_labels=classifier.predict(test_features))
