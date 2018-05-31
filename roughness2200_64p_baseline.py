from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,\
    Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
from datasets.utilities import imagenames, to_onehot
from imageio import imread
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt

# %% Creating Dataset
root = "./../data~/datasets/roughness2200/64p/"
classes = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7',
           'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
classes_Ra = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
# A_classes = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7']
# B_classes = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
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
data = data.reshape((-1, *img_shape_flat))
data_full = data.astype(float)
data_A, data_B = np.split(data_full, 2)
del data

# Creating labels
labels_full = np.arange(nb_classes)
labels_full = np.repeat(labels_full, img_per_class)
labels_half = np.arange(nb_classes/2)
labels_merged = np.tile(labels_half, 2)
labels_half = np.repeat(labels_half, img_per_class)
labels_merged = np.repeat(labels_merged, img_per_class)

# Creating one-hot labels
labels_full_onehot = to_onehot(labels_full, nb_classes)
labels_half_onehot = to_onehot(labels_half, int(nb_classes/2))
labels_merged_onehot = to_onehot(labels_merged, nb_classes)

# Creating Data-Scaler
scaler = StandardScaler()

print("[DONE]")

# %% Feature Space Distance Comparison
scan_steps_x = 40
scan_steps_z = 55

print("Calculating Distances... ", end='\r')

# Normalizing Data
data_A2 = np.split(data_A, 6)[0]
# data_A2 = scaler.fit_transform(data_A2)

# Average x-neighbour distance
x_neighbour_dist = np.array([])
for z in range(scan_steps_z):
    for x in range(1, scan_steps_x):
        i = x + z * scan_steps_x
        dist = np.linalg.norm(data_A2[i, :] - data_A2[i-1, :])
        x_neighbour_dist = np.append(x_neighbour_dist, [dist])

# Average y-neighbour distance
y_neighbour_dist = np.array([])
for x in range(scan_steps_x):
    for z in range(1, scan_steps_z):
        i = x + z * scan_steps_x
        i_ = x + (z-1) * scan_steps_x
        dist = np.linalg.norm(data_A2[i, :] - data_A2[i_, :])
        y_neighbour_dist = np.append(x_neighbour_dist, [dist])

# Calculating pairwise Distance accross hole data set.
dist = pairwise_distances(data_A2)

print("[DONE]")


print("X-Neighbour distance: mean = {0:3f}, std = {1:3f}".
      format(np.mean(x_neighbour_dist), np.std(x_neighbour_dist)))
print("X-Neighbour distance: mean = {0:3f}, std = {1:3f}".
      format(np.mean(y_neighbour_dist), np.std(y_neighbour_dist)))
print("Pairwise distance: mean = {0:3f}, std = {1:3f}".
      format(np.mean(dist), np.std(dist)))

# %% Nearest Centroid Classifier
print("Traing Centroid Classifier... ", end='\r')

model = NearestCentroid()
acc_full = []
acc_merged = []
acc_cross_ab = []
acc_cross_ba = []
nb_runs = 10
for i in range(nb_runs):
    # Randomizing dataset order
    np.random.seed(i)
    indexes = np.arange(data_full.shape[0])
    np.random.shuffle(indexes)
    data_full_r = data_full[indexes, :]
    labels_full_r = labels_full[indexes]
    labels_merged_r = labels_merged[indexes]

    # Extracting sets
    test_index = int(test_ratio * data_full.shape[0])
    test_data, train_data = np.split(data_full_r, [test_index])
    test_labels_f, train_labels_f = np.split(labels_full_r, [test_index])
    test_labels_m, train_labels_m = np.split(labels_merged_r, [test_index])

    # Normalizing sets
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Traning & Testing Model with Full labels
    model.fit(train_data, train_labels_f)
    acc_full.append(model.score(test_data, test_labels_f))

    # Traning & Testing Model with Merged labels
    model.fit(train_data, train_labels_m)
    acc_merged.append(model.score(test_data, test_labels_m))

# Traing with set A and Testing with test B
model.fit(data_A, labels_half)
acc_cross_ab = model.score(data_B, labels_half)

# Traing with set A and Testing with test B
model.fit(data_B, labels_half)
acc_cross_ba = model.score(data_A, labels_half)

print("[DONE]")
print("Accuracy (full set) : mean = {0:3f}, std = {1:3f} ({2:d} runs)".
      format(np.mean(acc_full), np.std(acc_full), nb_runs))
print("Accuracy (merged set) : mean = {0:3f}, std = {1:3f} ({2:d} runs)".
      format(np.mean(acc_merged), np.std(acc_merged), nb_runs))
print("Accuracy (trained on A, tested on B) : mean = {0:3f}, std = {1:3f} "
      "({2:d} run)".format(acc_cross_ab, 0, 1))
print("Accuracy (trained on B, tested on A) : mean = {0:3f}, std = {1:3f} "
      "({2:d} run)".format(acc_cross_ba, 0, 1))

# %% Linear Classifier
print("Training Linear Classifier... ", end='\r')


def new_model(nb_classes):
    model = Sequential([Dense(units=nb_classes,
                              activation='softmax',
                              input_shape=img_shape_flat)])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


acc_full = []
acc_merged = []
acc_cross_ab = []
acc_cross_ba = []
nb_runs = 10
for i in range(nb_runs):
    # Randomizing dataset order
    np.random.seed(i)
    indexes = np.arange(data_full.shape[0])
    np.random.shuffle(indexes)
    data_full_r = data_full[indexes]
    labels_full_r = labels_full_onehot[indexes]
    labels_merged_r = labels_merged_onehot[indexes]

    # Extracting sets
    test_index = int(test_ratio * data_full.shape[0])
    test_data, train_data = np.split(data_full_r, [test_index])
    test_labels_f, train_labels_f = np.split(labels_full_r, [test_index])
    test_labels_m, train_labels_m = np.split(labels_merged_r, [test_index])

    # Normalizing sets
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Traning & Testing Model with Full labels
    model = new_model(nb_classes)
    model.fit(x=train_data,
              y=train_labels_f,
              batch_size=64,
              epochs=5,
              shuffle=True,
              verbose=0)
    acc_full.append(model.evaluate(x=test_data,
                                   y=test_labels_f,
                                   verbose=0)[1])

    # Traning & Testing Model with Merged labels
    model = new_model(nb_classes)
    model.fit(x=train_data,
              y=train_labels_m,
              batch_size=64,
              epochs=5,
              shuffle=True,
              verbose=0)
    acc_merged.append(model.evaluate(x=test_data,
                                     y=test_labels_m,
                                     verbose=0)[1])

# Traing with set A and Testing with test B
model = new_model(int(nb_classes/2))
model.fit(x=data_A,
          y=labels_half_onehot,
          batch_size=64,
          epochs=5,
          shuffle=True,
          verbose=0)
acc_cross_ab = model.evaluate(x=data_B, y=labels_half_onehot, verbose=0)[1]

# Traing with set A and Testing with test B
model = new_model(int(nb_classes/2))
model.fit(x=data_B,
          y=labels_half_onehot,
          batch_size=64,
          epochs=5,
          shuffle=True,
          verbose=0)
acc_cross_ba = model.evaluate(x=data_A, y=labels_half_onehot, verbose=0)[1]

print("[DONE]")
print("Accuracy (full set) : mean = {0:3f}, std = {1:3f} ({2:d} runs)".
      format(np.mean(acc_full), np.std(acc_full), nb_runs))
print("Accuracy (merged set) : mean = {0:3f}, std = {1:3f} ({2:d} runs)".
      format(np.mean(acc_merged), np.std(acc_merged), nb_runs))
print("Accuracy (trained on A, tested on B) : mean = {0:3f}, std = {1:3f} "
      "({2:d} run)".format(acc_cross_ab, 0, 1))
print("Accuracy (trained on B, tested on A) : mean = {0:3f}, std = {1:3f} "
      "({2:d} run)".format(acc_cross_ba, 0, 1))

# %% Linear Regression
print("Training Linear Regression... ", end='\r')

# Creating regression labels
reg_labels = classes_Ra[labels_merged.astype(int)]


def test(y_true, y_pred):
    return K.abs(y_true - y_pred)


def new_model():
    model = Sequential([Dense(units=1, input_shape=img_shape_flat)])
    # model = Sequential([Dense(units=100, input_shape=img_shape_flat, activation='relu'),
    #                    Dense(units=100, activation='relu'),
    #                    Dense(units=1)])
    model.compile(optimizer='adam',
                  loss='mean_absolute_percentage_error',
                  metrics=[test])
    return model


nb_runs = 1
mean = np.zeros((nb_runs, int(nb_classes/2)))
for i in range(nb_runs):
    # Randomizing dataset order
    np.random.seed(i)
    indexes = np.arange(data_full.shape[0])
    np.random.shuffle(indexes)
    data_full_r = data_full[indexes]
    labels = reg_labels[indexes]

    # Extracting sets
    test_index = int(test_ratio * data_full.shape[0])
    test_data, train_data = np.split(data_full_r, [test_index])
    test_labels, train_labels = np.split(labels, [test_index])

    # Normalizing sets
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Traning & Testing Model
    model = new_model()
    model.fit(x=train_data,
              y=train_labels,
              batch_size=64,
              epochs=5,
              shuffle=True,
              verbose=1)
    data = np.split(scaler.transform(data_A), 6)
    for i in range(2):
        pred = model.predict(x=data[i], verbose=1)
        plt.hist(pred, 50)

print("[DONE]")
