import matplotlib.pyplot as plt
import numpy as np
from nupic.bindings.algorithms import SpatialPooler as SP
from tqdm import trange, tqdm

import load_mnist

uintType = "uint32"


# config
load_number = 100
inputDimensions = (28, 28)
columnDimensions = (50, 50)
num_training = 10

images, labels = load_mnist.load_images(images_number=load_number)
labels = labels.T[0]
numActiveColumnsPerInhArea = int(0.02 * np.prod(columnDimensions)),

sp = SP(inputDimensions,
        columnDimensions,
        potentialRadius = int(0.1*np.prod(inputDimensions)),
        numActiveColumnsPerInhArea = numActiveColumnsPerInhArea,
        globalInhibition = True,
        synPermActiveInc = 0.01,
        synPermInactiveDec = 0.008,
        wrapAround=False,)


def train(sp, images_train, learn=True):
    activeArray = np.zeros(columnDimensions, dtype=uintType)
    for iter_id in trange(num_training, desc="Training SP"):
        for im_train in images_train:
            sp.compute(im_train, learn, activeArray)


def train_with_history(sp, images_train, learn=True):
    activeArray = np.zeros(columnDimensions, dtype=uintType)
    sdr_history = []
    for iter_id in trange(num_training, desc="Training SP"):
        for im_train in images_train:
            sp.compute(im_train, learn, activeArray)
            sdr_history.append(np.copy(activeArray))
    return np.array(sdr_history, dtype=uintType)


def test(sp, images_test):
    sdr_history = np.zeros((len(images_test), columnDimensions[0], columnDimensions[1]), dtype=uintType)
    for im_id, im_test in enumerate(tqdm(images_test, desc="Testing SP")):
        sp.compute(im_test, False, sdr_history[im_id])
    return sdr_history


def compute_overlap(sdr_history, show=True):
    overlap = []
    for im_left in tqdm(sdr_history, desc="Computing overlap"):
        for im_right in sdr_history:
            overlap_pair = np.count_nonzero(np.logical_and(im_left, im_right))
            overlap.append(overlap_pair)
    sample_count = len(sdr_history)
    overlap = np.reshape(overlap, (sample_count, sample_count))
    np.fill_diagonal(overlap, 0)

    if show:
        print(overlap)
        print(overlap.shape)
        overlap_mean = np.mean(overlap)
        print("Overlap mean: {:.2f} / {} ({:.2f} %)".format(overlap_mean, numActiveColumnsPerInhArea,
                                                            100. * overlap_mean / numActiveColumnsPerInhArea))
        plt.imshow(overlap)
        plt.colorbar()
        plt.show()

    return overlap


def learn_identical():
    # Learning from the same image
    same_image = images[0]  # just the first image

    sdr_history_train = train_with_history(sp, [same_image])
    compute_overlap(sdr_history_train)

    plt.figure()
    plt.title("Learn identical")
    plt.subplot(211)
    plt.imshow(sdr_history_train[-1], cmap='gray_r')
    plt.subplot(212)
    plt.imshow(same_image, cmap='gray_r')
    plt.show()


def learn_mnist(label_of_interest=2, learn=True):
    images_interest = images[labels == label_of_interest]
    print(len(images_interest))
    train(sp, images, learn=learn)

    sdr_history_test = test(sp, images_interest)
    print(sdr_history_test)
    compute_overlap(sdr_history_test)

    plt.figure()
    plt.title("Learn MNIST: {}".format(learn))
    plt.subplot(221)
    plt.imshow(sdr_history_test[-1], cmap='gray_r')
    plt.subplot(222)
    plt.imshow(images_interest[-1], cmap='gray_r')
    plt.subplot(223)
    plt.imshow(sdr_history_test[-2], cmap='gray_r')
    plt.subplot(224)
    plt.imshow(images_interest[-2], cmap='gray_r')
    plt.show()


if __name__ == '__main__':
    # learn_identical()
    learn_mnist()
    # learn_mnist(learn=False)
