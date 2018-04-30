from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,\
    Dropout
from tensorflow.python.keras.models import Sequential


def VGG11(input_shape, nb_classes, dropout=False, dropout_rate=0.2):
    """
    Creates a vgg11 network.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input tensor not including the sample axis.
        Tensorflow uses the NHWC dimention ordering convention.
    nb_class : int
        The number of output class. The network will have this number of
        output nodes for one-hot encoding.
    dropout : bool
        Where or not to implement dropout in the fully-connected layers.
    dropout_rate : float
        Dropout rate.

    Returns
    -------
    keras.models.Sequential() :
        The create vgg11 network.
    """
    vgg11 = Sequential()

    # sub-net 1
    vgg11.add(Conv2D(filters=64,
                     kernel_size=3,
                     padding='same',
                     activation='relu',
                     input_shape=input_shape))
    vgg11.add(Conv2D(filters=64,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(MaxPool2D(pool_size=2))

    # sub-net 2
    vgg11.add(Conv2D(filters=128,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(Conv2D(filters=128,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(MaxPool2D(pool_size=2))

    # sub-net 3
    vgg11.add(Conv2D(filters=256,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(Conv2D(filters=256,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(MaxPool2D(pool_size=2))

    # sub-net 4
    vgg11.add(Conv2D(filters=512,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(Conv2D(filters=512,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(MaxPool2D(pool_size=2))

    # dense layers
    vgg11.add(Flatten())
    vgg11.add(Dense(units=256, activation='relu'))
    vgg11.add(Dropout(dropout_rate)) if dropout else None
    vgg11.add(Dense(units=256, activation='relu'))
    vgg11.add(Dropout(dropout_rate)) if dropout else None
    vgg11.add(Dense(units=nb_classes, activation='softmax'))

    return vgg11


# Test code
if __name__ == "__main__":
    # Creating Model
    model = VGG11(input_shape=(64, 64, 1), nb_classes=15)

    model.summary()
