from sklearn.model_selection import train_test_split


class Classifier:
    def __init__(self, classifier):
        self.classifier = classifier

    def train(self, data_set, test_size=0.2):
        X = data_set.features()
        y = data_set.labels()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        self.classifier.fit(X_train, y_train)

        return self.__confidence(X_test, y_test)

    def name(self): return self.classifier.__class__.__name__

    def __confidence(self, X_test, y_test):
        return self.classifier.score(X_test, y_test)
