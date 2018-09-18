from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot
from keras.models import model_from_json

def graph_model(model):
    converter = model_to_dot(model, show_shapes = True, show_layer_names = True)
    image = converter.create(prog='dot', format='svg')
    display(SVG(image))

def show_model(model):
    print(model.summary())
    return graph_model(model)

DEFAULT_MODEL_FILENAME = 'model'

def save(model, filename = DEFAULT_MODEL_FILENAME): 
    model.save_weights(f'{filename}_weights.h5')

def load(model, filename = DEFAULT_MODEL_FILENAME):
    # Load weights into the new model
    model.load_weights(f'{filename}_weights.h5')

    return model