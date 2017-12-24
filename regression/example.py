from classifier import Classifier
from datasources import DataSources
from dataset import Dataset, DatasetFactory
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import pandas as pd

pd.options.mode.chained_assignment = None


def print_separator(char='-', lenght=80): print(char * lenght)


def showDataset(data_set):
    print_separator()
    print("Dataset Head (Total: {} rows)".format(data_set.rows_count()))
    print_separator()
    print(data_set.head())
    print_separator()


def show_confidence(confidence, algorithm_name):
    print_separator(char='~')
    print("{}: {:06.4f}% confidence.".format(algorithm_name, confidence * 100))
    print_separator(char='~')


def algorithms():
    return [
        # (svm.SVR(kernel='poly'), "SVR Poly"),
        # (svm.SVR(kernel='rbf'), "SVR rbf"),
        # (svm.SVR(kernel='sigmoid'), "SVR sigmoid"),
        (SVR(kernel='linear'), "SVR Linear"),
        (LinearRegression(n_jobs=-1), "Linear Regression")
    ]


# -----------------------------------------------------------------------------
# Main program...
# -----------------------------------------------------------------------------
# Params...
test_size = 0.3
label_offset = 0.01
local_dataset = True

# Prepare...
data_frame = DataSources().google_action_prices(local=local_dataset)
data_set = DatasetFactory().createFrom(data_frame, label_offset=label_offset)
showDataset(data_set)

# Perform...
for (algorithm, algorithm_name) in algorithms():
    classifier = Classifier(algorithm)
    confidence = classifier.train(data_set, test_size)
    show_confidence(confidence, algorithm_name)

