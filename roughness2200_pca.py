from datasets.utilities import imagenames, to_onehot
from imageio import imread
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from progressbar import progressbar

# %% Centred Patches Dataset (Run block before loading)
root = "./../data~/datasets/roughness2200/centred_patches_64p/"
img_shape = (64, 64)
img_shape_flat = (64 * 64,)

# %% 64p Dataset
root = "./../data~/datasets/roughness2200/64p/"
img_shape = (64, 80)
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
data = np.empty((img_per_class*nb_classes, *img_shape), dtype=np.float)
for i in progressbar(range(nb_classes)):
    class_name = classes[i]
    for j, img in enumerate(img_names):
        imgpath = root + class_name + '/' + img
        data[i*img_per_class + j, ...] = imread(imgpath)

data_full = data.reshape((-1, *img_shape_flat))
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
labels_process = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
labels_process = np.repeat(labels_process, img_per_class)

# Creating one-hot labels
labels_full_onehot = to_onehot(labels_full, nb_classes)
labels_half_onehot = to_onehot(labels_half, int(nb_classes/2))
labels_merged_onehot = to_onehot(labels_merged, int(nb_classes/2))
labels_process_onehot = to_onehot(labels_process, 3)

# Creating Data-Scaler
scaler = StandardScaler()

# %% PCA

print("Fitting PCA... \t", end='\r')

# Normalizing sets
if True:
    input_data = scaler.fit_transform(data_full)
else:
    input_data = data_full

pca = PCA(n_components=10)
pca_full = pca.fit(input_data).transform(input_data)

print("[DONE]")

# %% Plotting Scatter Plot Matrix

print("Plotting PCA results... \t", end='\r')

N = 4               # Number of PCA components to plot

colors = ['#8b0000', '#ff0000', '#ff5a00', '#ff9a00', '#ffce00', '#f0ff00',
          '#66d4ff', '#00b8ff', '#009bd6', '#00719c', '#00415a', '#001f2b']

fig, axes = plt.subplots(N, N,
                         tight_layout=False,
                         figsize=(12, 9),
                         dpi=100)

for row in range(N):
    for col in range(N):
        ax = axes[row, col]
        ax.set_xticks([])
        ax.set_yticks([])
        if row == col:
            for i, class_name, color in zip(range(nb_classes), classes, colors):
                ax.hist(pca_full[labels_full == i, col],
                        bins=25,
                        color=color)
        else:
            handles = []
            for i, class_name, color in zip(range(nb_classes), classes, colors):
                h = ax.scatter(pca_full[labels_full == i, col],
                               pca_full[labels_full == i, row],
                               alpha=.5,
                               lw=0.1,
                               marker='.',
                               color=color,
                               label=class_name)
                handles.append(h)

# Ploting y labels
for n in range(N):
    axes[n, 0].set_ylabel('PCA {0:d}'.format(n+1), fontsize=14)

# Ploting y labels
for n in range(N):
    axes[N-1, n].set_xlabel('PCA {0:d}'.format(n+1), fontsize=14)

# PLotting legend
fig.tight_layout(pad=1.05, h_pad=1, w_pad=1, rect=(0, 0, 1, 0.96))
fig.legend(handles=handles, loc=9, fontsize=14, frameon=False,
           handletextpad=0, markerscale=3,  ncol=12, columnspacing=1.2,
           bbox_to_anchor=(0, 0, 1, 1))
# fig.legend(bbox_to_anchor=(0, 0, 1, 1), fontsize=14, loc=2)

print("[DONE]")


# %% PLotting PCA Explaind Variance

plt.figure('Explained Variance')
plt.plot(pca.explained_variance_ratio_*100)
plt.ylabel('Explained variance [%]')
plt.xlabel('PCA component')
plt.grid('on')

print("The {0:d} first PCA components contain {1:2.2f}%".
      format(N, sum(100*pca.explained_variance_ratio_[:N])))

# %% Plotting PCA Eigen Vectors

print("Plotting PCA Eigen Vectors... \t", end='\r', flush=True)

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
N = 10
for n in range(N):
    eigVec = pca.components_[n, :]
    eigVec = np.reshape(eigVec, img_shape)
    plt.figure('PCA Eigenvector {0:d}'.format(n+1),)
    imgplot = plt.imshow(eigVec,
                         norm=MidpointNormalize(midpoint=0.),
                         cmap='bwr')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("./pca_eig{0:d}.png".format(n+1),bbox_inches='tight',dpi=100)
    # plt.colorbar()
    # plt.title("PCA Eigenvector {0:d}".format(n+1))

print("[DONE]")
