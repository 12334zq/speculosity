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
import matplotlib as mpl
from sklearn.decomposition import PCA

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
data_classes = np.split(data_full, nb_classes)
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

# %% PCA

print("Fitting PCA... ", end='\r')

# Normalizing sets
# data_full_norm = scaler.fit_transform(data_full)

pca = PCA(n_components=50)
pca_full = pca.fit(data_full_norm).transform(data_full_norm)

print("[DONE]")

# %% Plotting


plt.figure()
plt.tight_layout()
N = 4           # Number of PCA components to plot
colors = ['#8b0000', '#ff0000', '#ff5a00', '#ff9a00', '#ffce00', '#f0ff00',
          '#66d4ff', '#00b8ff', '#009bd6', '#00719c', '#00415a', '#001f2b']

print("Plotting PCA results... ", end='\r')

for row in range(N):
    for col in range(N):
        plt.subplot(N, N, col + N*row + 1)
        if row == col:
            for i, class_name, color in zip(range(nb_classes), classes, colors):
                plt.hist(pca_full[labels_full == i, col],
                         bins = 20,
                         color=color)
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
        else:
            for i, class_name, color in zip(range(nb_classes), classes, colors):
                plt.scatter(pca_full[labels_full == i, col],
                            pca_full[labels_full == i, row],
                            alpha=.5,
                            lw=0.1,
                            marker='.',
                            color=color,
                            label=class_name)
                plt.tight_layout()
                plt.xticks([])
                plt.yticks([])

plt.subplot(N,N,1)
plt.ylabel('PCA 1')
plt.subplot(N,N,5)
plt.ylabel('PCA 2')
plt.subplot(N,N,9)
plt.ylabel('PCA 3')
plt.subplot(N,N,13)
plt.xlabel('PCA 1')
plt.ylabel('PCA 4')
plt.subplot(N,N,14)
plt.xlabel('PCA 2')
plt.subplot(N,N,15)
plt.xlabel('PCA 3')
plt.subplot(N,N,16)
plt.xlabel('PCA 4')

# Explained variance
plt.figure()
plt.plot(pca.explained_variance_ratio_*100)
plt.ylabel('Explained variance [%]')
plt.xlabel('PCA component')
plt.grid('on')



print("[DONE]")

print("The {0:d} first PCA components contain {1:2.2f}%".
      format(N, sum(100*pca.explained_variance_ratio_[:N])))

# %% Plotting PCA Eigen Vectors

print("Plotting PCA Eigen Vectors... ", end='\r')

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# Eigen vectors
for n in range(N):
    eigVec = pca.components_[n,:]
    eigVec = np.reshape(eigVec, img_shape)
    plt.figure()
    imgplot = plt.imshow(eigVec,
                         norm=MidpointNormalize(midpoint=0.),
                         cmap='bwr')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.colorbar()
    # plt.title("PCA Eigenvector {0:d}".format(n))
    
print("[DONE]")
