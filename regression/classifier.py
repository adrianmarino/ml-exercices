class Classifier:
    def __init__(self, classifier):
        self.classifier = classifier

    def train(self, data_set, test_size=0.2):
        X_train, X_test, y_train, y_test = data_set.train_test_split(test_size=test_size)

        self.classifier.fit(X_train, y_train)

        return self.__confidence(X_test, y_test)

    def predict(self, features): return self.classifier.predict(features)

    def name(self): return self.classifier.__class__.__name__

    def __confidence(self, X_test, y_test):
        return self.classifier.score(X_test, y_test)
