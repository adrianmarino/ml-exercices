from classifier import Classifier
from datasources import DataSources
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR
from tools import ConfidenceTable
from dataset import Dataset, DatasetFactory


algorithms = [LinearSVR(), SVR(kernel='linear', cache_size=1000), LinearRegression(n_jobs=-1)]


def classifiers(): return map(lambda alg: Classifier(alg), algorithms)

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
print(data_set)

# Perform...
print(ConfidenceTable(map(lambda clr: [clr.name(), clr.train(data_set, test_size)], classifiers())))
