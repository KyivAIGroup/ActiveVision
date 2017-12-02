import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm

# Must import 'capnp' before schema import hook will work
import capnp
from nupic.proto.SpatialPoolerProto_capnp import SpatialPoolerProto

# cpp
from nupic.bindings.algorithms import SpatialPooler

# python; for debug only
# from nupic.algorithms.spatial_pooler import SpatialPooler

import load_mnist

uintType = "uint32"

# config
load_number = 1000
inputDimensions = (28, 28)
columnDimensions = (100, 100)
num_training = 300
checkpoint_step = int(num_training // 10)
numActiveColumnsPerInhArea = int(0.1 * np.prod(columnDimensions))

images, labels = load_mnist.load_images(images_number=load_number)

sp = SpatialPooler(inputDimensions,
                   columnDimensions,
                   potentialRadius=int(0.1 * np.prod(inputDimensions)),
                   numActiveColumnsPerInhArea=numActiveColumnsPerInhArea,
                   globalInhibition=True,
                   synPermActiveInc=0.01,
                   synPermInactiveDec=0.008,
                   wrapAround=False, )


def unravel_dimensions():
    """
    Flattens the input vector and mini-column dimensions
    Needed for python Spatial Pooler
    """
    global inputDimensions, columnDimensions, images
    inputDimensions = (inputDimensions[0] * inputDimensions[1],)
    columnDimensions = (columnDimensions[0] * columnDimensions[1],)
    images = images.reshape(len(images), -1)


def get_dir_config():
    root_dir = os.path.join(os.path.dirname(__file__), "result")
    folder = os.path.join(root_dir,
                          "input_{}".format('x'.join(map(str, inputDimensions))),
                          "column_{}".format('x'.join(map(str, columnDimensions))),
                          "loaded_{}".format(load_number))
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def save_model(sp, iter_id):
    """
    Serialize the Spatial Pooler and save to 'iter_id' checkpoint
    :param sp: Spatial Pooler
    :param iter_id: checkpoint
    """
    builder = SpatialPoolerProto.new_message()
    sp.write(builder)
    sp_path = os.path.join(get_dir_config(), "iter_{}.cproto".format(iter_id))
    with open(sp_path, 'wb') as f:
        builder.write_packed(f)


def load_model(model_path):
    """
    :param model_path: path to serialized Spatial Pooler
    :return: Spatial Pooler
    """
    with open(model_path, 'rb') as f:
        reader = SpatialPoolerProto.read_packed(f)
    sp = SpatialPooler.read(reader)
    return sp


def train(sp, images_train, learn=True):
    activeArray = np.zeros(columnDimensions, dtype=uintType)
    for iter_id in trange(num_training, desc="Training SP"):
        for im_train in images_train:
            sp.compute(im_train, learn, activeArray)
        if iter_id > 0 and iter_id % checkpoint_step == 0:
            save_model(sp, iter_id)


def train_with_history(sp, images_train, learn=True):
    activeArray = np.zeros(columnDimensions, dtype=uintType)
    sdr_history = []
    for iter_id in trange(num_training, desc="Training SP"):
        for im_train in images_train:
            sp.compute(im_train, learn, activeArray)
            sdr_history.append(np.copy(activeArray))
    return np.array(sdr_history, dtype=uintType)


def test(sp, images_test):
    history_shape = [len(images_test)] + list(columnDimensions)
    sdr_history = np.zeros(history_shape, dtype=uintType)
    for im_id, im_test in enumerate(tqdm(images_test, desc="Testing SP")):
        sp.compute(im_test, False, sdr_history[im_id])
    return sdr_history


def compute_overlap(sdr_history):
    overlap = []
    for im_left in tqdm(sdr_history, desc="Computing overlap"):
        for im_right in sdr_history:
            overlap_pair = np.count_nonzero(np.logical_and(im_left, im_right))
            overlap.append(overlap_pair)
    sample_count = len(sdr_history)
    overlap = np.reshape(overlap, (sample_count, sample_count))
    # np.fill_diagonal(overlap, 0)
    return overlap


def plot_overlap(overlap):
    print(overlap)
    print("Overlap: shape={}".format(overlap.shape))
    overlap_mean = np.mean(overlap)
    plt.figure()
    plt.title("Overlap mean: {:.2f} / {} ({:.2f} %)".format(overlap_mean, numActiveColumnsPerInhArea,
                                                            100. * overlap_mean / numActiveColumnsPerInhArea))
    plt.imshow(overlap)
    plt.colorbar()
    plt.savefig(os.path.join(get_dir_config(), "overlap.png"))


def learn_identical(show_example=False):
    # Learning from the same image
    same_image = images[0]  # just the first image

    sdr_history_train = train_with_history(sp, [same_image])
    overlap = compute_overlap(sdr_history_train)
    plot_overlap(overlap)

    if show_example:
        plt.figure()
        plt.title("Learn identical")
        plt.subplot(211)
        plt.imshow(sdr_history_train[-1], cmap='gray_r')
        plt.subplot(212)
        plt.imshow(same_image, cmap='gray_r')
        plt.show()


def learn_mnist(label_of_interest=2, learn=True, show_example=False):
    images_interest = images[labels == label_of_interest]
    train(sp, images, learn=learn)

    sdr_history_test = test(sp, images_interest)
    overlap = compute_overlap(sdr_history_test)
    plot_overlap(overlap)

    if show_example:
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
    # unravel_dimensions()
    learn_mnist()
    # learn_mnist(learn=False)
