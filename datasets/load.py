import numpy as np
import imageio
import collections

Dataset = collections.namedtuple('Dataset', ['data', 'labels'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

# Roughness dataset
_roughness_class_dir = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7',
                        'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
_roughness_image_shape = (128, 128)
_roughness_nb_channels = 1
_roughness_test_ratio = 0.2


def _imagenames(nb_images, format=".bmp", start_at_one=False):
    """
    Generator function yielding the image names for data sets name as "i.ext"
    where 'i' is the image number and '.ext' is the image file extension.

    :param int nb_images: Number of images.
    :param str format: File extension of the image format.
    :param bool start_at_one: Whether to start counting the images at 1 or 0.
    :return: The image file name.
    :rtype: str
    """

    for i in range(start_at_one, nb_images + start_at_one):
        yield "{0:d}{1:s}".format(i, format)


def _to_onehot(labels, nb_classes):
    nb_samples = len(labels)
    labels_onehot = np.zeros((nb_samples, nb_classes))
    labels_onehot[np.arange(nb_samples), labels] = 1
    return labels_onehot


def array_from_images(root, filenames):
    """
    Reads images from a folder and returns them in an array of shape
    [nb_images, height, width, channels].

    :param str root: Root folder containing the images
    :param filenames: An iterator ot generator yielding the filenames.
    :return: An array containing the images.
    :rtype: numpy.array
    """
    images = list()

    for filename in filenames:
        images.append(imageio.imread(root + '/' + filename))

    images = np.array(images)

    # If the the image is BW then create one channel
    if images.ndim != 4:
        images = np.expand_dims(images, 3)

    return images


def load(dataset, root, one_hot=True):
    """
    Returns one of the available data sets.

    :param str dataset: String tag of the dataset to use.
    :param bool one_hot: Use one hot encoding of labels. Only relevant for
        classification datasets.
    :return: A tuple containing train_data, train_labels, val_data, val_labels,
        test_data, test_labels which are all numpy arrays. The shape of the
        array is NWHC. If no validation data is avaiable, None is return for
        both val_data and val_labels.
    :rtype: tuple
    """

    classfication = False
    nb_classes = None
    val_labels = []
    val_data = []

    if dataset == "roughness":
        classfication = True
        nb_classes = len(_roughness_class_dir)

        # Importing data
        data = np.empty((0, *_roughness_image_shape, _roughness_nb_channels))
        for dir in _roughness_class_dir:
            dirpath = root + '/' + dir
            data = np.append(data,
                             array_from_images(dirpath, _imagenames(100)),
                             0)

        # Creating labels
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
    if classfication and one_hot:
        train_labels = _to_onehot(train_labels, nb_classes)
        test_labels = _to_onehot(test_labels, nb_classes)
        if val_labels:
            val_labels = _to_onehot(val_labels, nb_classes)

    # Combining datasets
    train = Dataset(data=train_data, labels=train_labels)
    test = Dataset(data=test_data, labels=test_labels)
    validation = Dataset(data=val_data, labels=val_labels)
    return Datasets(train=train, test=test, validation=validation)


# Test code
if __name__ == "__main__":
    dataset = load(dataset="roughness",
                   root="./../../data/datasets/roughness/1/bmp")
