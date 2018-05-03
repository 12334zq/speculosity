import numpy as np
import collections
from .utilities import to_onehot, imagenames, array_from_images

Dataset = collections.namedtuple('Dataset', ['data', 'labels'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

# Roughness dataset
_full_classes = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7',
                 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
_A_classes = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7']
_B_classes = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
_image_shape = (128, 128)
_nb_channels = 1
_roughness_test_ratio = 0.2


def load(root,
         classes="full",
         one_hot=True,
         test_ratio=0.2,
         merge=False,
         set64=False):
    """
    Returns one of the available data sets.

    :param str classes: String tag specyfing which classes to load.
    :param bool one_hot: Use one hot encoding of labels. Only relevant for
        classification datasets.
    :param bool merge: Specifies wheter to merge A and B classes of the same
        roughness value (i.e. A2 and B2 data will have the same label).
    :return: A tuple containing train_data, train_labels, val_data, val_labels,
        test_data, test_labels which are all numpy arrays. The shape of the
        array is NWHC. If no validation data is avaiable, None is return for
        both val_data and val_labels.
    :rtype: Datasets structure.
    """

    # Select classes
    if classes.lower() == "full":
        nb_classes = len(_full_classes)
        classes = _full_classes
    elif classes.lower() == 'a':
        nb_classes = len(_A_classes)
        classes = _A_classes
    elif classes.lower() == 'b':
        nb_classes = len(_B_classes)
        classes = _B_classes

    # Select root
    if set64:
        root += "64p/"
    else:
        root += "128p/"

    # Importing data
    data = np.empty((0, *_image_shape, _nb_channels))
    for class_name in classes:
        dirpath = root + class_name
        data = np.append(data,
                         array_from_images(dirpath, imagenames(100)),
                         0)

    # Creating labels
    if merge and classes.lower() == "full":
        labels = np.arange(nb_classes/2)
        labels = labels.tile(2)
    else:
        labels = np.arange(nb_classes)

    labels = np.repeat(labels, 100)

    # Mixing classes
    indexes = np.arange(100*nb_classes)
    np.random.seed(0)
    np.random.shuffle(indexes)
    data = data[indexes, :, :, :]
    labels = labels[indexes]

    # Extracting sets
    test_index = int(_roughness_test_ratio*100*nb_classes)

    test_data = data[0:test_index, :, :, :]
    test_labels = labels[0:test_index]
    train_data = data[test_index:, :, :, :]
    train_labels = labels[test_index:]

    # Converting to onehot encoding if required
    train_labels = to_onehot(train_labels, nb_classes)
    test_labels = to_onehot(test_labels, nb_classes)

    # Combining datasets
    train = Dataset(data=train_data, labels=train_labels)
    test = Dataset(data=test_data, labels=test_labels)
    validation = Dataset(data=[], labels=[])
    return Datasets(train=train, test=test, validation=validation)


# Test code
if __name__ == "__main__":
    dataset = load()
