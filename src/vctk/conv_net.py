from keras import Sequential
from keras.layers import Conv2D, InputLayer, LeakyReLU, Flatten, Dense


def get_conv_model(input_shape, output_shape) -> Sequential:
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(6, 6), strides=(2, 2), padding='valid'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='valid'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())
    model.add(Dense(output_shape, activation='relu'))

    model.compile()
    return model
