from statistics import mean
from matplotlib import style
import matplotlib.pyplot as plt
import random
import numpy as np


def square(data): return data ** 2


def squared_error(ys, y): return sum((y - ys) ** 2)


def regression_slope(xs, ys): return ((mean(xs) * mean(ys)) - mean(xs * ys)) / (square(mean(xs)) - mean(square(xs)))


def regression_intercept(m, xs, ys): return mean(ys) - (m * mean(xs))


def coefficient_of_determination(ys, y):
    regression_squared_error = squared_error(ys, y)

    y_mean = [mean(ys) for _ in ys]
    y_squared_error = squared_error(ys, y_mean)

    return 1 - (regression_squared_error / y_squared_error)


def crate_dataset(points_count, variance, step, correlation=0):
    last_y = 1
    ys = []
    for i in range(points_count):
        y = last_y + random.randrange(- variance, variance)
        ys.append(y)
        if correlation > 0:
            last_y += step
        elif correlation < 0:
            last_y -= step

    xs = np.array([i for i in range(len(ys))], dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)

    return xs, ys

class LinearRegressionClassifier:
    def predict(self, xs): return [self.slope * x + self.intercept for x in xs]

    def train(self, xs, ys):
        self.slope = regression_slope(xs, ys)
        self.intercept = regression_intercept(self.slope, xs, ys)
        y = classifier.predict(xs)
        return coefficient_of_determination(ys, y) * 100


def plot(xs, ys, y, x1, y1):
    style.use('ggplot')

    # Plot train points...
    plt.scatter(xs, ys, color='#003F72', label='Data')
    # Plot regression line...
    plt.plot(xs, y, label='Lineal Regression')

    # Predicted point...
    plt.scatter(x1, y1, label='Prediction')

    plt.legend(loc=4)

    plt.show()


xs, ys = crate_dataset(points_count=40, variance=40, step=2, correlation=1)
# xs, ys = crate_dataset(points_count=40, variance=10, step=2, correlation=1)
# xs, ys = crate_dataset(points_count=40, variance=10, step=2, correlation=-1)
# xs, ys = crate_dataset(points_count=40, variance=10, step=2, correlation=0)


classifier = LinearRegressionClassifier()

accuracy = classifier.train(xs, ys)
print("Coefficient of determination: {}".format(accuracy))


y = classifier.predict(xs)

x1 = [7.7]
y1 = classifier.predict(x1)

plot(xs, ys, y, x1, y1)
