import os
import time
import humanize
import lib.plot_utils as plot_utils


class ModelWrapper:
    def __init__(self, model, path='./'):
        self.model = model
        self.path = path

    def show(self):
        print(self.model.summary())
        return plot_utils.graph_model(self.model)

    def save(self):
        model_saved_path = os.path.join(self.path, "model.h5")
        json_saved_path = os.path.join(self.path, "model.json")

        json_model = self.model.to_json()
        with open(json_saved_path, "w") as json_file:
            json_file.write(json_model)

        self.model.save(model_saved_path)

    def train(self, train_set, validation_set, epochs=10, batch_size=128, shuffle=False):
        time_start = time.time()

        history = self.model.fit(
            train_set.features,
            train_set.labels,
            validation_data=(validation_set.features, validation_set.labels),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle
        )
        print(f'Training time: {humanize.naturaltime(time.time() - time_start)}')

        return history

    def evaluate(self, test_set, batch_size=128):
        test_loss = self.model.evaluate(
            test_set.features,
            test_set.labels,
            batch_size=batch_size
        )
        print(f'Loss: {test_loss}')