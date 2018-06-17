# General experiments on encoding image to L4
# learning attractors
# It makes basic clustering
# How to make fixed number of clusters so that they fill whole input space
# now it is filling the space by hypercubes. of fixed side.


import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import load_mnist

IMAGES_NUMBER = 1000
LAYERS_PATH = os.path.join('models_bin', 'exp9_1.npy')


def kWTA(cells, sparsity):
    assert cells.ndim == 1
    n_active = max(int(sparsity * cells.size), 1)
    n_active = min(n_active, len(cells.nonzero()[0]))
    winners = np.argsort(cells)[-n_active:]
    sdr = np.zeros(cells.shape, dtype=cells.dtype)
    sdr[winners] = 1
    return sdr


def is_iterable(value):
    return hasattr(value, '__iter__')


def create_parent_dir(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def plot_threshold_impact(digits=(0, 1), hidden_size=(2000,), hidden_sparsity=0.05, include_raw_layer=False):
    if include_raw_layer:
        # raw layer only
        hidden_size = []
    images_unused, _ = load_mnist.load_images(images_number=IMAGES_NUMBER, train=True, digits=digits)
    n_images = len(images_unused)
    thr_linspace = 1.0 / np.power(2, np.arange(start=5, stop=-1, step=-1))
    n_attractors = []
    accuracies = []
    for thr in thr_linspace:
        train(digits=digits, hidden_sizes=hidden_size, hidden_sparsity=hidden_sparsity, hidden_thr_overlap=thr,
              include_raw_layer=include_raw_layer)
        layers = np.load(LAYERS_PATH)
        n_attractors.append(len(layers[-1]['attractors']))
        accuracy = _test()
        accuracies.append(accuracy)
    n_attractors = np.hstack(n_attractors) / float(n_images)
    plt.plot(thr_linspace, n_attractors, label='# attractors, normed', marker='o')
    plt.plot(thr_linspace, accuracies, label='accuracies', marker='o')
    title = 'Digits {}'.format(tuple(digits))
    if include_raw_layer:
        title += ". Raw input only"
    plt.title(title)
    plt.xlabel('overlap threshold')
    plt.legend()
    plot_path = 'plots/threshold_impact_digits={},include_raw_layer={}.png'.format(tuple(digits), include_raw_layer)
    create_parent_dir(plot_path)
    plt.savefig(plot_path)
    plt.show()


def _do_train(layers, images, labels):
    n_digits_unique = len(np.unique(labels))
    label_map = {code: None for code in range(n_digits_unique)}
    layers[-1]['label_map'] = label_map
    for im, label_true in tqdm(zip(images, labels), desc="Train"):
        y = im.flatten()
        winner = -1
        for layer in layers:
            y = kWTA(np.dot(layer['weights'], y), sparsity=layer['sparsity'])
            l_attractors = layer['attractors']
            if len(l_attractors) == 0:
                l_attractors.append(y)
            overlaps = np.dot(l_attractors, y)
            winner = overlaps.argmax()
            if overlaps[winner] < layer['threshold']:
                l_attractors.append(y)
                winner = len(l_attractors) - 1
            y = l_attractors[winner]
        label_map[winner] = label_true


def train(digits=(5, 6), hidden_sizes=(2000, 10000), hidden_sparsity=0.05, hidden_thr_overlap=(0.3, 0.1),
          include_raw_layer=False,
          display=False):
    images, labels = load_mnist.load_images(images_number=IMAGES_NUMBER, train=True, digits=digits)
    images = np.array([cv2.resize(im, (14, 14)) for im in images], dtype=np.uint8)
    x_mean = images.mean()
    images = (images > x_mean).astype(np.int32)

    layers = [dict(sparsity=np.mean(images), size=images[0].size, x_mean=x_mean, digits=digits,
                   weights=np.identity(n=images[0].size, dtype=np.int32),
                   threshold=1.0*images[0].size*hidden_thr_overlap,
                   include_raw_layer=include_raw_layer,
                   attractors=[])]

    n_hidden = len(hidden_sizes)
    if not is_iterable(hidden_sparsity):
        hidden_sparsity = [hidden_sparsity] * n_hidden
    if not is_iterable(hidden_thr_overlap):
        hidden_thr_overlap = [hidden_thr_overlap] * n_hidden
    for layer_id in range(n_hidden):
        size = hidden_sizes[layer_id]
        w = np.random.binomial(n=1, p=0.1, size=(size, layers[layer_id]['size']))
        sparsity = hidden_sparsity[layer_id]
        max_overlap = sparsity * size
        layers.append(dict(sparsity=sparsity,
                           size=size,
                           threshold=hidden_thr_overlap[layer_id] * max_overlap,
                           weights=w,
                           attractors=[]))
    first_layer_id = 1 - include_raw_layer
    _do_train(layers=layers[first_layer_id:], images=images, labels=labels)

    n_input_unique = len(images)
    for layer_id in range(first_layer_id, len(layers)):
        layer = layers[layer_id]
        n_attractors = len(layer['attractors'])
        print("Layer {}: {} inputs --> {} attractors".format(layer_id, n_input_unique, n_attractors))
        n_input_unique = n_attractors

    for layer_id in range(first_layer_id, len(layers)):
        layer = layers[layer_id]
        if display:
            layer_attractors = layer['attractors']
            for attractor in layer_attractors:
                w = layer['weights']
                sparsity_prev = layers[layer_id - 1]['sparsity']
                x_recontraction = kWTA(np.dot(w.T, attractor), sparsity=sparsity_prev)
                side_size = int(np.ceil(np.sqrt(x_recontraction.size)))
                x_recontraction.resize(side_size, side_size)
                plt.imshow(x_recontraction)
                plt.title("Layer {} reconstructed".format(layer_id-1))
                plt.show()
    np.save(LAYERS_PATH, layers)


def _test():
    layers = np.load(LAYERS_PATH)
    first_layer_id = 1 - layers[0]['include_raw_layer']
    l_input = layers[0]
    images, labels = load_mnist.load_images(images_number=IMAGES_NUMBER, train=False, digits=l_input['digits'])
    images = np.array([cv2.resize(im, (14, 14)) for im in images], dtype=np.uint8)
    images = (images > l_input['x_mean']).astype(np.int32)
    label_map = layers[-1]['label_map']
    labels_predicted = []
    for im, label_true in tqdm(zip(images, labels), desc="Test"):
        y = im.flatten()
        winner = -1
        for layer in layers[first_layer_id:]:
            y = kWTA(np.dot(layer['weights'], y), sparsity=layer['sparsity'])
            l_attractors = layer['attractors']
            overlaps = np.dot(l_attractors, y)
            winner = overlaps.argmax()
            y = l_attractors[winner]
        labels_predicted.append(label_map[winner])
    labels_predicted = np.hstack(labels_predicted)
    accuracy = np.mean(labels == labels_predicted)
    print("Accuracy: {}".format(accuracy))
    return accuracy


if __name__ == '__main__':
    np.random.seed(26)
    create_parent_dir(LAYERS_PATH)
    # train()
    # test()
    # plot_threshold_impact(digits=(5, 6))
    plot_threshold_impact(digits=range(10), hidden_size=[], include_raw_layer=True)
