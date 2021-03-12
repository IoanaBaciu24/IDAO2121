import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Activation,Dense, Input
from tensorflow.keras.models import Model

def create_vgg16(input_shape, starting_filters = 64, classes = 2, is_regression = False):
    inputs = Input(input_shape)
    x = Conv2D(filters = starting_filters, kernel_size=(3,3),padding="same", activation="relu")(inputs)
    x = Conv2D(filters = starting_filters, kernel_size=(3,3),padding="same", activation="relu")(x)

    x = MaxPooling2D(pool_size = (2,2), strides = (2,2))(x)

    x = Conv2D(filters = starting_filters*2, kernel_size=(3,3),padding="same", activation="relu")(x)
    x = Conv2D(filters = starting_filters*2, kernel_size=(3,3),padding="same", activation="relu")(x)

    x = MaxPooling2D(pool_size = (2,2), strides = (2,2))(x)

    x = Conv2D(filters = starting_filters*4, kernel_size=(3,3),padding="same", activation="relu")(x)
    x = Conv2D(filters = starting_filters*4, kernel_size=(3,3),padding="same", activation="relu")(x)
    x = Conv2D(filters = starting_filters*4, kernel_size=(3,3),padding="same", activation="relu")(x)

    x = MaxPooling2D(pool_size = (2,2), strides = (2,2))(x)

    x = Conv2D(filters = starting_filters*8, kernel_size=(3,3),padding="same", activation="relu")(x)
    x = Conv2D(filters = starting_filters*8, kernel_size=(3,3),padding="same", activation="relu")(x)
    x = Conv2D(filters = starting_filters*8, kernel_size=(3,3),padding="same", activation="relu")(x)

    x = MaxPooling2D(pool_size = (2,2), strides = (2,2))(x)


    x = Conv2D(filters = starting_filters*8, kernel_size=(3,3),padding="same", activation="relu")(x)
    x = Conv2D(filters = starting_filters*8, kernel_size=(3,3),padding="same", activation="relu")(x)
    x = Conv2D(filters = starting_filters*8, kernel_size=(3,3),padding="same", activation="relu")(x)

    x = MaxPooling2D(pool_size = (2,2), strides = (2,2))(x)

    activation = 'sigmoid'
    if is_regression:
        activation = 'linear'
    output = Conv2D(filters=classes, kernel_size = 1, strides = 1, padding = 'same', activation= activation)(x)
    
    model = Model(inputs = inputs, outputs = output)

    model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = 'accuracy')
    model.summary()


create_vgg16((576, 576, 3), starting_filters=16)
