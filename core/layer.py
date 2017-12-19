import numpy as np
import cv2
import matplotlib.pyplot as plt


class Area(object):
    def __init__(self):
        self.layers = {}

    def add_layer(self, layer):
        self.layers[layer.name] = layer


class Layer(object):
    def __init__(self, name, shape):
        self.name = name
        self.cells = np.zeros(shape, dtype=np.int32)
        self.size = np.prod(shape)
        self.input_layers = []
        self.weights = []
        self.sparsity = 0.1

        # association memory params
        self.cluster_size = 2
        self.clusters = 'tbd'

    def connect_input(self, new_layer):
        self.input_layers.append(new_layer)
        self.weights.append(np.random.rand(self.size, new_layer.size))

    def linear_update(self):
        signal = np.zeros(self.cells.shape, dtype=np.float32)
        for layer_id, layer in enumerate(self.input_layers):
            signal += np.dot(self.weights[layer_id], layer.cells.flatten())
        self.cells = self.k_winners_take_all(signal, self.sparsity)

    def associate(self, input_layers=[]):
        return 0
        if len(input_layers) == 1:
            input = input_layers[0].cells
        if len(input_layers) > 1:
            input = np.zeros(shape=0, dtype=int)
            for layer in input_layers:
                input = np.append(input, layer.cells)
        # clusters - 3d matrix that stored synapse membership to clusters
        probability = 0.1
        self.cluster_size = 2
        if np.count_nonzero(input):
            for i in np.nonzero(self.cells)[0]:
                # I will create clusters not whithin all active cells but just with part
                if np.random.rand() < probability:
                    # self.clusters[i] is a matrix, where return array of indices [(rows), (cols)]
                    overlap = np.intersect1d(np.where(self.clusters[i] == 0)[0], np.nonzero(input)[0])
                    # overlap show which synapses have free clusters to which I can write
                    # select cluster_size active pre neurons
                    if overlap.size >= self.cluster_size:
                        pre = np.random.choice(overlap, self.cluster_size, replace=False)
                        pre_tree = []
                        for k in range(self.cluster_size):
                            # this is to select free cluster among all synapses pairs
                            pre_tree.append(np.random.choice(np.where(self.clusters[i, pre[k]] == 0)[0]))
                        self.clusters[i, pre, pre_tree] = np.max(self.clusters[i]) + 1  # max +1 is number of cluster
                    else:
                        print 'full'

    def k_winners_take_all(self, activations, sparsity):
        n_active = max(int(sparsity * len(activations)), 1)
        winners = np.argsort(activations)[-n_active:]
        sdr = np.zeros(activations.shape, dtype=self.cells.dtype)
        sdr[winners] = 1
        return sdr


class SaliencyRegion(object):

    @staticmethod
    def get_saliency_map(image):
        image = image.astype(np.uint8)
        result = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)) + np.abs(
            cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3))
        result = result / (np.max(result) + 1e-7)
        thresh = result > 0.5
        return result * thresh

    def __iter__(self):
        return self

    def next(self):
        # todo implement
        return np.random.randint(low=-10, high=10, size=2)

    @staticmethod
    def get_vector_from_saliency(saliency_map):
        x_c, y_c = int(saliency_map.shape[0] / 2), int(saliency_map.shape[1] / 2)
        saliency_map[x_c - 5:x_c + 5, y_c - 5:y_c + 5] = 0
        x, y = np.unravel_index(saliency_map.argmax(), saliency_map.shape)
        plt.imshow(saliency_map)
        plt.show()
        print x, y
        return x - x_c, y - y_c
