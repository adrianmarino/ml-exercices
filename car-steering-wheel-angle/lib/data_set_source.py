import os

import cv2
import numpy as np
import pandas as pd

from lib.video import Video
from lib.data_set import DataSet


class DataSetSource:
    def __init__(self, path): self.path = path

    def __epoch_video_path(self, epoch_number):
        return os.path.join(self.path, 'epoch{:0>2}_front.mkv'.format(epoch_number))

    def __epoch_features(self, epoch_number, image_processor):
        video = Video(self.__epoch_video_path(epoch_number))
        images = video.images(image_processor)
        video.close()
        return images

    def __epoch_labels(self, epoch_number):
        path = os.path.join(self.path, 'epoch{:0>2}_steering.csv'.format(epoch_number))
        rows = pd.read_csv(path)
        return rows['wheel'].values

    @staticmethod
    def __flip_features(features, labels):
        augmented_features = []
        augmented_labels = []

        for feature, label in zip(features, labels):
            augmented_features.append(feature)
            augmented_labels.append(label)

            augmented_features.append(cv2.flip(feature, 1))
            augmented_labels.append(float(label) * -1.0)

        return augmented_features, augmented_labels

    def load(
            self,
            epochs=range(1, 10),
            flip_features=False,
            feature_processor=lambda img: img,
            data_set_name='data_set'
    ):
        features = []
        labels = []

        print(f'Loading epochs...', end='')
        for epoch_number in epochs:
            labels.extend(self.__epoch_labels(epoch_number))
            features.extend(self.__epoch_features(epoch_number, feature_processor))
            assert len(features) == len(labels)
            print(f' {epoch_number}', end='')
        print('')

        if flip_features:
            print(f'Flipping features...')
            features, labels = self.__flip_features(features, labels)

        labels = np.array(labels)
        labels = np.reshape(labels, (len(labels), 1))
        features = np.array(features)

        data_set = DataSet(features, labels, name=data_set_name)

        print(f'Data set loaded! {data_set}')
        return data_set
