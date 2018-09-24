from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt
import numpy as np

def graph_model(model):
    converter = model_to_dot(model, show_shapes=True, show_layer_names=True)
    image = converter.create(prog='dot', format='svg')
    display(SVG(image))


def plot_loss(history):
    # list all data in history
    print(history.history.keys())

    # summarize history for loss
    plt.plot(history.history['loss'], 'C0--')
    plt.plot(history.history['val_loss'], 'C0')

    # plt.title('model loss')
    plt.ylabel('Loss', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.legend(['train', 'valid'], loc='best')
    plt.xlim((0, 20))
    plt.xticks(np.arange(0, 21, 5))
    plt.grid()
    plt.show()
