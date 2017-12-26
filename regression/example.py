from classifier import Classifier
from datasources import DataSources
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR
from tools import table_decorator
from dataset import Dataset, DatasetFactory
import pandas as pd


algorithms = [LinearSVR(), SVR(kernel='linear', cache_size=1000), LinearRegression(n_jobs=-1)]


def classifiers(): return map(lambda alg: Classifier(alg), algorithms)


def confidence_table(rows):
    content = pd.DataFrame(list(map(lambda row: (row[0], row[1]*100), rows)), columns=['Algorithm', 'Confidence (%)'])
    return table_decorator("Results", content)

# -----------------------------------------------------------------------------
# Main program...
# -----------------------------------------------------------------------------
# Params...
test_size = 0.1
label_offset = 0.01
local_dataset = False

# Prepare...
data_frame = DataSources().google_actions(local=local_dataset)
data_set = DatasetFactory().create_from(data_frame=data_frame, label_offset=label_offset)
classifiers = classifiers()

# Perform...
print(data_set)
print(confidence_table(map(lambda clr: [clr.name(), clr.train(data_set, test_size)], classifiers)))
