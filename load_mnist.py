import os
import mnist
import numpy as np
import pickle

from constants import DATA_DIR

__version__ = '0.2'


def _get_saved_path(images_number, train, data_dir):
    fold = "train" if train else "test"
    if images_number is None:
        images_number = 'all'
    fpath = os.path.join(data_dir, "MNIST", "v{}".format(__version__), fold, "{}.pkl".format(images_number))
    return fpath


def _download_images(images_number, train, file_path):
    print("Downloading {} images to {}".format(images_number, file_path))
    if train:
        images = mnist.train_images()
        labels = mnist.train_labels()
    else:
        images = mnist.test_images()
        labels = mnist.test_labels()
    index_shuffled = np.random.permutation(len(labels))
    images = np.take(images, index_shuffled, axis=0)
    labels = np.take(labels, index_shuffled)
    if images_number is not None:
        images = images[:images_number]
        labels = labels[:images_number]
    images = images.astype(np.uint8)
    labels = labels.astype(int)
    data_dir = os.path.dirname(file_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(file_path, 'wb') as f:
        pickle.dump((images, labels), f)


def load_images(images_number, train=True, data_dir=DATA_DIR, digits=range(10)):
    """
    :param images_number: int, how many to load. Pass `None` to load all images.
    :param train: bool, train or test
    :param data_dir: str, directory to store cached pickled files
    :param digits: digits to take
    :return: MNIST images, labels
    """
    fpath = _get_saved_path(images_number, train, data_dir)
    if not os.path.exists(fpath):
        _download_images(images_number, train, fpath)
    with open(fpath, 'rb') as f:
        images, labels = pickle.load(f)
    digits_present_mask = np.isin(labels, list(digits))
    images = images[digits_present_mask]
    labels = labels[digits_present_mask]
    return images, labels
