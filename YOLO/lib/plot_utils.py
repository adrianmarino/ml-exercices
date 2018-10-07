from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt
import numpy as np

def graph_model(model):
    converter = model_to_dot(model, show_shapes=True, show_layer_names=True)
    image = converter.create(prog='dot', format='svg')
    display(SVG(image))