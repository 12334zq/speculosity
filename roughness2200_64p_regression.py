from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,\
    Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import models
from datasets.utilities import imagenames, to_onehot
from callbacks import PlotMetrics
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from sklearn.preprocessing import StandardScaler


# %% Creating Dataset
root = "./../data~/datasets/roughness2200/64p/"
classes = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7',
           'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
classes_Ra = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
img_per_class = 2200
img_names = list(imagenames(img_per_class))
img_shape = (64, 80)
img_shape_flat = (64 * 80,)
nb_classes = 12
nb_channels = 1
test_ratio = 0.1

print("Loading data... ", end='\r')

# Loading Images
data = []
for class_name in classes:
    for img in img_names:
        imgpath = root + class_name + '/' + img
        data.append(imread(imgpath))

data = np.asarray(data)
data_full = data.astype(float)
data_A, data_B = np.split(data_full, 2)
data_classes = np.split(data_full, nb_classes)
del data

# Creating labels
labels_half = np.arange(nb_classes/2)
labels_merged = np.tile(labels_half, 2)
labels_half = np.repeat(labels_half, img_per_class)
labels_merged = np.repeat(labels_merged, img_per_class)

# Creating regression labels
labels_half = classes_Ra[labels_half.astype(int)]
labels_merged = classes_Ra[labels_merged.astype(int)]

# Creating Data-Scaler
scaler = StandardScaler()

print("[DONE]")

# %% Regression model
input_shape = (64, 80, 1)

def new_model(input_shape, dropout=False, dropout_rate=0.2):
    model = Sequential()

    # sub-net 1
    model.add(Conv2D(filters=8,
                     kernel_size=3,
                     padding='same',
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(filters=8,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=2))

    # sub-net 2
    model.add(Conv2D(filters=12,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=12,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=2))

    # sub-net 3
    model.add(Conv2D(filters=16,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=16,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=2))

    # dense layers
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(dropout_rate)) if dropout else None
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(dropout_rate)) if dropout else None
    # model.add(Dense(units=nb_classes, activation='softmax'))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam',
                  loss='mean_absolute_percentage_error')
    
    return model


# %%  Train & Test Model
    
# Flattening Dataset
data_full = data_full.reshape((-1, *img_shape_flat))

print("Training Convolutional Regression Model...", end='\r')


nb_runs = 10
loss = []
for i in range(nb_runs):
    # Randomizing dataset order
    np.random.seed(i)
    indexes = np.arange(data_full.shape[0])
    np.random.shuffle(indexes)
    data_full = data_full[indexes]
    labels_merged = labels_merged[indexes]
    
    # Extracting sets
    test_index = int(test_ratio * data_full.shape[0])
    test_data, train_data = np.split(data_full, [test_index])
    test_labels, train_labels = np.split(labels_merged, [test_index])
    
    # Normalizing sets
    # train_data = scaler.fit_transform(train_data)
    # test_data = scaler.transform(test_data)
    
    # Un-Flattening Dataset
    train_data = train_data.reshape((-1, *img_shape, 1))
    test_data = test_data.reshape((-1, *img_shape, 1))
    
    # Traim model
    model = new_model(input_shape, dropout=True, dropout_rate=0.2)
    model.fit(x=train_data,
              y=train_labels,
              batch_size=64,
              epochs=10,
              validation_split=0.1,
              callbacks=[PlotMetrics(plot_acc=False, plot_loss=True)],
              shuffle=True,
              verbose=0)

    loss.append(model.evaluate(test_data, 
                               test_labels, 
                               batch_size=64,
                               verbose=0))



# %% Training on A, testing on B
train_data = data_A.reshape((-1, *img_shape, 1))
test_data = data_B.reshape((-1, *img_shape, 1))
loss_ab=[]
for i in range(nb_runs):
    model = new_model(input_shape, dropout=True, dropout_rate=0.2)
    model.fit(x=train_data,
              y=labels_half,
              batch_size=64,
              epochs=10,
              validation_split=0.1,
              callbacks=[PlotMetrics(plot_acc=False, plot_loss=True)],
              shuffle=True,
              verbose=0)
    
    loss_ab.append(model.evaluate(test_data, 
                                  labels_half, 
                                  batch_size=64,
                                  verbose=0))

# Training on B, testing on A
train_data = data_B.reshape((-1, *img_shape, 1))
test_data = data_A.reshape((-1, *img_shape, 1))
loss_ba=[]
for i in range(nb_runs):
    model = new_model(input_shape, dropout=True, dropout_rate=0.2)
    model.fit(x=train_data,
              y=labels_half,
              batch_size=64,
              epochs=10,
              validation_split=0.1,
              callbacks=[PlotMetrics(plot_acc=False, plot_loss=True)],
              shuffle=True,
              verbose=0)
    
    loss_ba.append(model.evaluate(test_data, 
                                  labels_half, 
                                  batch_size=64,
                                  verbose=0))

print("[DONE]")

print("Loss (full set) : mean = {0:3f}," 
      " std = {1:3f} ({2:d} runs)".
      format(np.mean(loss), np.std(loss), nb_runs))

print("Loss (trained on A, tested on B) : mean = {0:3f}, std = {1:3f} "
      "({2:d} run)".format(np.mean(loss_ab), np.std(loss_ab), nb_runs))

print("Loss (trained on B, tested on A) : mean = {0:3f}, std = {1:3f} "
      "({2:d} run)".format(np.mean(loss_ba), np.std(loss_ba), nb_runs))

