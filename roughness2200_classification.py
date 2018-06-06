from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,\
    Dropout
from tensorflow.python.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from datasets.utilities import imagenames, to_onehot
from imageio import imread
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
from progressbar import progressbar, ProgressBar
from callbacks import PlotMetrics

# %% Centred Patches Dataset (Run block before loading)
root = "./../data~/datasets/roughness2200/centred_patches_64p/"
img_shape = (64, 64, 1)
img_shape_flat = (64 * 64,)

# %% 64p Dataset
root = "./../data~/datasets/roughness2200/64p/"
img_shape = (64, 80, 1)
img_shape_flat = (64 * 80,)

# %% Load Dataset
classes = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7',
           'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
classes_Ra = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
# A_classes = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7']
# B_classes = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
img_per_class = 2200
img_names = list(imagenames(img_per_class))
nb_classes = 12
nb_channels = 1
test_ratio = 0.1

print("Loading data... ", flush=True)

# Loading Images
data_full = np.empty((img_per_class*nb_classes, *img_shape), dtype=np.float)
for i in progressbar(range(nb_classes)):
    class_name = classes[i]
    for j, img in enumerate(img_names):
        imgpath = root + class_name + '/' + img
        data_full[i*img_per_class + j, :, :, 0] = imread(imgpath)

data_A, data_B = np.split(data_full, 2)
data_classes = np.split(data_full, nb_classes)

# Creating labels
labels_full = np.arange(nb_classes)
labels_full = np.repeat(labels_full, img_per_class)
labels_half = np.arange(nb_classes/2)
labels_merged = np.tile(labels_half, 2)
labels_half = np.repeat(labels_half, img_per_class)
labels_merged = np.repeat(labels_merged, img_per_class)
labels_process = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
labels_process = np.repeat(labels_process, img_per_class)

# Creating one-hot labels
labels_full_onehot = to_onehot(labels_full, nb_classes)
labels_half_onehot = to_onehot(labels_half, int(nb_classes/2))
labels_merged_onehot = to_onehot(labels_merged, int(nb_classes/2))
labels_process_onehot = to_onehot(labels_process, 3)

# Creating Data-Scaler
scaler = StandardScaler()

# %% Netowrk

def new_model(nb_classes, dropout=True, dropout_rate=0.2):
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
    model = Sequential()

    # sub-net 1
    model.add(Conv2D(filters=16,
                     kernel_size=3,
                     padding='same',
                     activation='relu',
                     input_shape=img_shape))
    model.add(Conv2D(filters=16,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=2))

    # sub-net 2
    model.add(Conv2D(filters=32,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=32,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=2))

    # sub-net 3
    model.add(Conv2D(filters=64,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=64,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=2))

    # dense layers
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(dropout_rate)) if dropout else None
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(dropout_rate)) if dropout else None
    model.add(Dense(units=nb_classes, activation='softmax'))
    
    # Compile Model
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

def train(model, train_data, train_labels):
    model.fit(x=train_data,
              y=train_labels,
              batch_size=32,
              epochs=10,
              shuffle=True,
              verbose=0)
    model.fit(x=train_data,
              y=train_labels,
              batch_size=256,
              epochs=10,
              shuffle=True,
              verbose=0)
    model.fit(x=train_data,
              y=train_labels,
              batch_size=2048,
              epochs=10,
              shuffle=True,
              verbose=0)
    
    return model


# %% Testing model
    
print("Training Convolutional Classifier... ", flush=True)

acc_full = []
acc_merged = []
acc_process = []
nb_runs = 10
epochs = 5

with ProgressBar(max_value=nb_runs*3 + 2) as bar:
    bar.update(0)
    for i in range(nb_runs):
        # Randomizing dataset order
        np.random.seed(i)
        indexes = np.arange(data_full.shape[0])
        np.random.shuffle(indexes)
        data_full_r = data_full[indexes]
        labels_full_r = labels_full_onehot[indexes]
        labels_merged_r = labels_merged_onehot[indexes]
        labels_process_r = labels_process_onehot[indexes]
    
        # Extracting sets
        test_index = int(test_ratio * data_full.shape[0])
        test_data, train_data = np.split(data_full_r, [test_index])
        test_labels_f, train_labels_f = np.split(labels_full_r, [test_index])
        test_labels_m, train_labels_m = np.split(labels_merged_r, [test_index])
        test_labels_p, train_labels_p = np.split(labels_process_r, [test_index])
    
        # Normalizing sets
        if False:
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.transform(test_data)
            
        # Traning & Testing Model with Full labels
        model = new_model(nb_classes)
        model = train(model, train_data, train_labels_f)
        acc_full.append(model.evaluate(x=test_data,
                                       y=test_labels_f,
                                       verbose=0)[1])
        bar.update(i*3 + 1)
    
        # Traning & Testing Model with Merged labels
        model = new_model(int(nb_classes/2))
        model = train(model, train_data, train_labels_m)
        acc_merged.append(model.evaluate(x=test_data,
                                         y=test_labels_m,
                                         verbose=0)[1])
        bar.update(i*3 + 2)
    
        # Training & Testing Model with Process labels
        model = new_model(3)
        model = train(model, train_data, train_labels_p)
        acc_process.append(model.evaluate(x=test_data,
                                          y=test_labels_p,
                                          verbose=0)[1])
        bar.update(i*3 + 3)
        
    # Traing with set A and Testing with test B
    model = new_model(int(nb_classes/2))
    model = train(model, data_A, labels_half_onehot)
    acc_cross_ab = model.evaluate(x=data_B, y=labels_half_onehot, verbose=0)[1]
    bar.update(3*nb_runs + 1)
    
    # Traing with set A and Testing with test B
    model = new_model(int(nb_classes/2))
    model = train(model, data_B, labels_half_onehot)
    acc_cross_ba = model.evaluate(x=data_A, y=labels_half_onehot, verbose=0)[1]
    bar.update(3*nb_runs + 2)
    
    print("Accuracy (full set) : mean = {0:3f}, std = {1:3f} ({2:d} runs)".
          format(np.mean(acc_full), np.std(acc_full), nb_runs))
    print("Accuracy (merged set) : mean = {0:3f}, std = {1:3f} ({2:d} runs)".
          format(np.mean(acc_merged), np.std(acc_merged), nb_runs))
    print("Accuracy (process labels) : mean = {0:3f}, std = {1:3f} ({2:d} runs)".
          format(np.mean(acc_process), np.std(acc_process), nb_runs))
    print("Accuracy (trained on A, tested on B) : mean = {0:3f}, std = {1:3f} "
          "({2:d} run)".format(acc_cross_ab, 0, 1))
    print("Accuracy (trained on B, tested on A) : mean = {0:3f}, std = {1:3f} "
          "({2:d} run)".format(acc_cross_ba, 0, 1))

