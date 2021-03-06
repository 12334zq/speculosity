from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,\
    Dropout
from tensorflow.python.keras.models import Sequential


def VGG16(input_shape, nb_classes, dropout=False, dropout_rate=0.2):
    """
    Creates a VGG16 network.

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
        The create VGG16 network.
    """
    vgg16 = Sequential()

    # sub-net 1
    vgg16.add(Conv2D(filters=64,
                     kernel_size=3,
                     padding='same',
                     activation='relu',
                     input_shape=input_shape))
    vgg16.add(Conv2D(filters=64,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg16.add(MaxPool2D(pool_size=2))

    # sub-net 2
    vgg16.add(Conv2D(filters=128,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg16.add(Conv2D(filters=128,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg16.add(MaxPool2D(pool_size=2))

    # sub-net 3
    vgg16.add(Conv2D(filters=256,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg16.add(Conv2D(filters=256,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg16.add(Conv2D(filters=256,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg16.add(MaxPool2D(pool_size=2))

    # sub-net 4
    vgg16.add(Conv2D(filters=512,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg16.add(Conv2D(filters=512,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg16.add(Conv2D(filters=512,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg16.add(MaxPool2D(pool_size=2))

    # sub-net 5
    vgg16.add(Conv2D(filters=512,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg16.add(Conv2D(filters=512,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg16.add(Conv2D(filters=512,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg16.add(MaxPool2D(pool_size=2))

    # dense layers
    vgg16.add(Flatten())
    vgg16.add(Dense(units=4096, activation='relu'))
    vgg16.add(Dropout(dropout_rate)) if dropout else None
    vgg16.add(Dense(units=4096, activation='relu'))
    vgg16.add(Dropout(dropout_rate)) if dropout else None
    vgg16.add(Dense(units=nb_classes, activation='softmax'))

    return vgg16


# Test code
if __name__ == "__main__":
    # Importing data
    import os
    os.chdir('./../datasets')
    from data import load
    data = load("rings")
    image_shape = (256, 256, 1)
    nb_classes = 10

    # Creating Model
    model = VGG16(input_shape=image_shape, nb_classes=nb_classes)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=data.train.data.reshape(-1, *image_shape),
              y=data.train.labels,
              epochs=1,
              batch_size=10,
              validation_data=(data.validation.data.reshape(-1, *image_shape),
                               data.validation.labels))
