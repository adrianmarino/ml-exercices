import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split


class DataSet:
    @staticmethod
    def load(name='train', path='./dataset'):
        features = pickle.load(open(f'{path}/{name}_data_set_features.p', 'rb'))
        labels = pickle.load(open(f'{path}/{name}_data_set_labels.p', 'rb'))
        data_set = DataSet(features, labels, name, path)
        print(f'Data set loaded! {data_set}')
        return data_set

    def __init__(self, features, labels, name='train', path='./dataset'):
        self.labels = labels
        self.features = features
        self.name = name
        self.path = path

    def __str__(self):
        return f'DataSet(name: {self.name}, features.shape: {self.features.shape}, labels.shape: {self.labels.shape})'

    def plot_example(self, example_number=0):
        plt.imshow(self.features[example_number])
        print(f'Angle: {self.labels[example_number]}')

    def save(self):
        self.__save(self.features, 'features')
        print(f'{self.name.capitalize()} data set:\n - features saved!')

        self.__save(self.labels, 'labels')
        print(f' - labels saved!')

    def __save(self, array, array_name):
        pickle.dump(array, open(self.__array_filename(array_name), 'wb'))

    def __array_filename(self, array_name):
        return f'{self.path}/{self.name}_data_set_{array_name}.p'

    def split(self, split_percent=0.1, shuffle=False, name='data_set_part'):
        options = {"test_size": split_percent}
        if shuffle:
            options['random_state'] = 42

        features_a, features_b, labels_a, labels_b = train_test_split(
            self.features,
            self.labels,
            **options
        )

        data_set_a = DataSet(features_a, features_b, name=self.name)
        data_set_b = DataSet(features_b, labels_b, name=name)
        print(f'Split {self.name} data set into (split percent: {split_percent} %):\n - {data_set_a}\n - {data_set_b}')
        return data_set_a, data_set_b
