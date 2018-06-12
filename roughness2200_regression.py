from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,\
    Dropout, BatchNormalization
from tensorflow.python.keras.models import Sequential
from datasets.utilities import imagenames, to_onehot
from imageio import imread
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar, ProgressBar
from callbacks import PlotMetrics
from tensorflow.python.keras import backend as K

# %% Centred Patches Dataset (Run block before Creating Dataset)
root = "./../data~/datasets/roughness2200/centred_patches_64p/"
img_shape = (64, 64, 1)
img_shape_flat = (64 * 64,)

# %% 64p Dataset (Run block before Creating Dataset)
root = "./../data~/datasets/roughness2200/64p/"
img_shape = (64, 80, 1)
img_shape_flat = (64 * 80,)

# %% Creating Dataset
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
labels_half = classes_Ra
labels_merged = np.tile(labels_half, 2)
labels_half = np.repeat(labels_half, img_per_class)
labels_merged = np.repeat(labels_merged, img_per_class)

# Creating Data-Scaler
scaler = StandardScaler()

print("[DONE]")


# %% Netowrk

def mean_scaled_square_error(y_true, y_pred):
    return K.square((y_true - y_pred)/y_true)

def mean_log(y_true, y_pred):
    return K.square((y_true - y_pred)/y_true)


def expontial(x):
    return K.exp(x)


def new_model(dropout=True, dropout_rate=0.2):
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
    model.add(BatchNormalization())
    model.add(Dense(units=64, activation='softplus'))
    model.add(Dropout(dropout_rate)) if dropout else None
    model.add(BatchNormalization())
    model.add(Dense(units=64, activation='softplus'))
    model.add(Dropout(dropout_rate)) if dropout else None
    model.add(BatchNormalization())
    model.add(Dense(units=6, activation='softplus'))
    model.add(Dropout(dropout_rate)) if dropout else None
    model.add(BatchNormalization())
    model.add(Dense(units=1, activation=expontial))

    # Compile Model
    model.compile(optimizer='adam', loss='mean_absolute_percentage_error')
    # model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')
    # model.compile(optimizer='adam', loss=mean_scaled_square_error)

    return model


def train(model, train_data, train_labels, verbose=0, plotloss=False, val_split=0):
    if plotloss:
        cb = [PlotMetrics()]
    else:
        cb = []

    model.fit(x=train_data,
              y=train_labels,
              batch_size=32,
              epochs=10,
              shuffle=True,
              verbose=verbose,
              callbacks=cb,
              val_split=val_split)
    model.fit(x=train_data,
              y=train_labels,
              batch_size=256,
              epochs=10,
              shuffle=True,
              verbose=verbose,
              callbacks=cb,
              val_split=val_split)
    model.fit(x=train_data,
              y=train_labels,
              batch_size=1024,
              epochs=10,
              shuffle=True,
              verbose=verbose,
              callbacks=cb,
              val_split=val_split)

    return model


# %% Testing model

print("Training Convolutional Regressor... ", flush=True)

loss = []
nb_runs = 10
epochs = 5

with ProgressBar(max_value=nb_runs + 2) as bar:
    bar.update(0)
    for i in range(nb_runs):
        # Randomizing dataset order
        np.random.seed(i)
        indexes = np.arange(data_full.shape[0])
        np.random.shuffle(indexes)
        data_r = data_full[indexes]
        labels_r = labels_merged[indexes]

        # Extracting sets
        test_index = int(test_ratio * data_r.shape[0])
        test_data, train_data = np.split(data_r, [test_index])
        test_labels, train_labels = np.split(labels_r, [test_index])

        # Normalizing sets
        if True:
            train_data = scaler.fit_transform(train_data.reshape(-1, *img_shape_flat))
            test_data = scaler.transform(test_data.reshape(-1, *img_shape_flat))
            train_data = train_data.reshape(-1, *img_shape)
            test_data = test_data.reshape(-1, *img_shape)

        # Traning & Testing Model with Full labels
        model = new_model()
        model = train(model, train_data, train_labels)
        loss.append(model.evaluate(x=test_data, y=test_labels, verbose=0))
        bar.update(i + 1)



    # Traing with set A and Testing with test B
    model = new_model()
    model = train(model, data_B, labels_half)
    loss_ba = model.evaluate(x=data_A, y=labels_half, verbose=0)
    bar.update(nb_runs + 2)

    print("Mean absolute percentage error : mean = {0:3f}, std = {1:3f} ({2:d} runs)".
          format(np.mean(loss), np.std(loss), nb_runs))
    print("Mean absolute percentage error (trained on B, tested on A) : mean = {0:3f}, std = {1:3f} "
          "({2:d} run)".format(loss_ba, 0, 1))


 # %% Traing with set A and Testing with test B

print("Training Convolutional Regressor... ", flush=True)

model = new_model()
model = train(model, data_B, labels_half, verbose=1, plotloss=True, val_split=0.1)
loss_ab = model.evaluate(x=data_A, y=labels_half, verbose=0)

print("[DONE]")
print("Mean absolute percentage error (trained on A, tested on B) : mean = {0:3f}, std = {1:3f} "
      "({2:d} run)".format(loss_ab, 0, 1))


# %% PLotting results
pred = np.empty((img_per_class, nb_classes))
for i, c in enumerate(data_classes):
    pred[:, i] = np.squeeze(model.predict(c))


def plot_reflines():
    for i, c in enumerate(classes_Ra):
        plt.hlines(c, i + 0.5, i + 1.5, colors='red')


plt.figure('Regression on row B')

ax = plt.subplot(121)
plt.boxplot(pred[:, 6:], labels=classes_Ra)
plt.title('Training Set Predictions')
# ax.set_yscale('log')
plot_reflines()

ax = plt.subplot(122)
plt.boxplot(pred[:, 0:6], labels=classes_Ra)
plt.title('Testing Set Predictions')
# ax.set_yscale('log')
plot_reflines()


# %% Traing on all classes except one

pred = np.empty((img_per_class*2, int(nb_classes/2)))

def plot_reflines():
    for i, c in enumerate(classes_Ra):
        plt.hlines(c, i + 0.5, i + 1.5, colors='red')

for i in progressbar(range(int(nb_classes/2))):
    # Generate sets
    indexes_A = np.arange(i*img_per_class, (i+1)*img_per_class)
    indexes_B = np.arange((i+6)*img_per_class, (i+7)*img_per_class)
    indexes = np.append(indexes_A, indexes_B)
    train_data = np.delete(data_full, indexes, 0)
    test_data = data_full[indexes, ...]
    train_labels = np.delete(labels_merged, indexes, 0)
    test_labels = labels_merged[indexes, ...]

    # Train netork
    model = new_model()
    model = train(model, train_data, train_labels)
    pred[:, i] = np.squeeze(model.predict(test_data))

plt.figure('Model Prediction on Un-Trained Roughnesses')
plt.boxplot(pred, labels=classes_Ra)
plot_reflines()
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.tight_layout()

