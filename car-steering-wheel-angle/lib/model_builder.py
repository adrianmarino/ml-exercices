from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


def build_model(
        config,
        conv_neurons=[24, 32, 48, 64, 64],
        conv_kernels_size=[(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
        conv_strides=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
        pool_sizes=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
        network_neurons=[1164, 100, 50, 10, 1],
        network_dropouts=[0.2, 0.2, 0.5, 0.5, -1],
        loss='mse',
        optimizer='Adam'
):
    tensor_shape = (
        config['data_set']['image']['height'],
        config['data_set']['image']['width'],
        config['data_set']['image']['channels']
    )

    model = Sequential()
    model.add(Lambda(lambda x: x / 255., input_shape=tensor_shape, name='Input_Lambda_Layer'))

    for index, neurons in enumerate(conv_neurons):
        model.add(Conv2D(
            neurons,
            kernel_size=conv_kernels_size[index],
            strides=conv_strides[index],
            padding='same',
            activation='elu',
            kernel_initializer='he_normal',
            name=f'Convolutional_2D_Layer_{index + 1}'
        ))
        model.add(BatchNormalization(
            name=f'Batch_Normalization_Layer_{index + 1}'
        ))
        model.add(MaxPooling2D(
            pool_size=pool_sizes[index],
            name=f'Max_Pooling_2D_Layer_{index + 1}'
        ))

    model.add(Flatten())

    for index, neurons in enumerate(network_neurons):
        model.add(Dense(
            neurons,
            activation='elu',
            kernel_initializer='he_normal',
            name=f'Dense_Layer_{index + 1}'
        ))
        if network_dropouts[index] > 0:
            model.add(Dropout(network_dropouts[index]))

    model.compile(loss=loss, optimizer=optimizer)

    return model
