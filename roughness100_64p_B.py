from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,\
    Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import models
from datasets import roughness100
from callbacks import PlotMetrics
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Importing data
root = "./../data/datasets/roughness100/"
data_b = roughness100.load(root=root, set64=True, classes='B')
data_a = roughness100.load(root=root, set64=True, classes='A', test_ratio=1)
input_shape = (64, 64, 1)
nb_classes = 6

# Parameters
load_model_name = "./../data/models/roughness100_64p_B"
save_model_name = "./../data/models/roughness100_64p_B"
load = True
save = False


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
    vgg11.add(Conv2D(filters=12,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(Conv2D(filters=16,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(MaxPool2D(pool_size=2))

    # sub-net 3
    vgg11.add(Conv2D(filters=16,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(Conv2D(filters=16,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    vgg11.add(MaxPool2D(pool_size=2))

    # dense layers
    vgg11.add(Flatten())
    vgg11.add(Dense(units=64, activation='relu'))
    vgg11.add(Dropout(dropout_rate)) if dropout else None
    vgg11.add(Dense(units=64, activation='relu'))
    vgg11.add(Dropout(dropout_rate)) if dropout else None
    vgg11.add(Dense(units=nb_classes, activation='softmax'))

    return vgg11


# Load/Create Model
if load:
    model = models.load_model(load_model_name, compile=False)
    print("\nModel loaded from \"{0:s}\"\n".format(load_model_name))
else:
    model = VGG11(input_shape, nb_classes, dropout=True, dropout_rate=0.4)

# Train Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=data_b.train.data,
          y=data_b.train.labels,
          batch_size=256,
          epochs=64,
          validation_split=0.1,
          callbacks=[PlotMetrics(plot_acc=True, plot_loss=True)],
          shuffle=True)

# Save model
if save:
    model.save(save_model_name)
    print("\nModel saved as \"{0:s}\"\n".format(save_model_name))

# Test network
score = model.evaluate(data_b.test.data, data_b.test.labels, batch_size=64)
print("\nTest loss: {0:2f}, Test Accuracy: {1:2f}\n".
      format(score[0], score[1]))


# Cross testing network on data from set A
score = model.evaluate(data_a.test.data, data_a.test.labels, batch_size=64)
print("\nCross testing on set A ({0:d} samples):".
      format(data_a.test.data.shape[0]))
print("Test loss: {0:2f}, Test Accuracy: {1:2f}\n".format(score[0], score[1]))

# Plot confusion matrix on set B
pred_labels = model.predict(data_b.test.data)
cm = confusion_matrix(y_true=np.argmax(data_b.test.labels, 1),
                      y_pred=np.argmax(pred_labels, 1))
plt.figure()
plt.imshow(cm)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.title("Confusion on set B")
plt.show()


# Plot confusion matrix on set A (cross-testing)
pred_labels = model.predict(data_a.test.data)
cm = confusion_matrix(y_true=np.argmax(data_a.test.labels, 1),
                      y_pred=np.argmax(pred_labels, 1))
plt.figure()
plt.imshow(cm)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.title("Confusion on set A")
plt.show()
