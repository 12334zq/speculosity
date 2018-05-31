import numpy as np
import imageio

__all__ = ["array_from_images", "to_onehot", "imagenames"]


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


def to_onehot(labels, nb_classes):
    labels = labels.astype(int)
    nb_samples = len(labels)
    labels_onehot = np.zeros((nb_samples, nb_classes))
    labels_onehot[np.arange(nb_samples), labels] = 1
    return labels_onehot


def imagenames(nb_images, format=".bmp", start_at_one=False):
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
