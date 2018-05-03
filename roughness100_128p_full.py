from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,\
    Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import models
from datasets import roughness100
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Importing data
root = "./../data/datasets/roughness100/"
data = roughness100.load(root=root)
input_shape = (128, 128, 1)
nb_classes = 12

# Parameters
load_model_name = "./../data/models/roughness100_128p_full"
save_model_name = "./../data/models/roughness100_128p_full"
load = False
save = True


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
    vgg11.add(Conv2D(filters=8,
                     kernel_size=3,
                     padding='same',
                     activation='relu',
                     input_shape=input_shape))
    vgg11.add(Conv2D(filters=8,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(MaxPool2D(pool_size=2))

    # sub-net 2
    vgg11.add(Conv2D(filters=16,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(Conv2D(filters=16,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(MaxPool2D(pool_size=2))

    # sub-net 3
    vgg11.add(Conv2D(filters=32,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(Conv2D(filters=32,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(MaxPool2D(pool_size=2))

    # sub-net 4
    vgg11.add(Conv2D(filters=32,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(Conv2D(filters=32,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(MaxPool2D(pool_size=2))

    # dense layers
    vgg11.add(Flatten())
    vgg11.add(Dense(units=128, activation='relu'))
    vgg11.add(Dropout(dropout_rate)) if dropout else None
    vgg11.add(Dense(units=128, activation='relu'))
    vgg11.add(Dropout(dropout_rate)) if dropout else None
    vgg11.add(Dense(units=nb_classes, activation='softmax'))

    return vgg11


# Load/Creaate Model
if load:
    model = models.load_model(load_model_name, compile=False)
    print("Model loaded from \"{0:s}\"".format(load_model_name))
    print()
else:
    model = VGG11(input_shape, nb_classes, dropout=True, dropout_rate=0.4)

# Train Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=data.train.data,
          y=data.train.labels,
          batch_size=64,
          epochs=12,
          validation_split=0.1,
          shuffle=True)

# Save model
if save:
    model.save(save_model_name)
    print("Model saved as \"{0:s}\"".format(save_model_name))
    print()

# Test network
score = model.evaluate(data.test.data, data.test.labels, batch_size=64)
print("Test loss: {0:2f}, Test Accuracy: {1:2f}".format(score[0], score[1]))


# Print confusion matrix
pred_labels = model.predict(data.test.data)
cm = confusion_matrix(y_true=np.argmax(data.test.labels, 1),
                      y_pred=np.argmax(pred_labels, 1))
plt.imshow(cm)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()
