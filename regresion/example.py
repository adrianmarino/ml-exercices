from classifier import Classifier
from commons import Datasets
from dataset import DatasetFactory
from sklearn import svm
from sklearn.linear_model import LinearRegression
import pandas as pd
pd.options.mode.chained_assignment = None


def showDataset(df):
    print(df.head())
    print("Dataset size: {}".format(len(df)))


def show_confidence(confidence, algorithm_name):
    print("{} Score: {:06.4f}%".format(algorithm_name, confidence * 100))


def classifiers():
    return [
        # (svm.SVR(kernel='poly'), "SVR Poly"),
        # (svm.SVR(kernel='rbf'), "SVR rbf"),
        # (svm.SVR(kernel='sigmoid'), "SVR sigmoid"),
        (svm.SVR(kernel='linear'), "SVR Linear"),
        (LinearRegression(n_jobs=-1), "LinearRegression")
    ]

# -----------------------------------------------------------------------------
# Main program...
# -----------------------------------------------------------------------------
# Params...
test_size = 0.1
label_offset = 0.01
local_dataset = False

# Prepare...
raw_dataset = Datasets().google_action_prices(local=local_dataset)
data_set = DatasetFactory().createFrom(raw_dataset, label_offset=label_offset)
showDataset(data_set)

# Perform...
for (algorithm, algorithm_name) in classifiers():
    classifier = Classifier(algorithm)
    confidence = classifier.train(data_set, test_size)
    show_confidence(confidence, algorithm_name)
