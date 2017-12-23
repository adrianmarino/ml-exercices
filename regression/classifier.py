import numpy as np
from sklearn import preprocessing, model_selection


class Classifier:
    def __init__(self, classifier):
        self.classifier = classifier

    def train(self, data_set, test_size=0.2):
        X = self.__features_matrix(data_set)
        y = self.__labels_vector(data_set)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)

        self.classifier.fit(X_train, y_train)

        return self.classifier.score(X_test, y_test)

    def name(self): return self.classifier.__class__.__name__

    def __features_matrix(self, data_set):
        features = np.array(data_set.drop(['label'], 1))
        return preprocessing.scale(features)

    def __labels_vector(self, data_set):
        return np.array(data_set['label'])