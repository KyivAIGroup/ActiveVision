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


class SaliencyMap(object):

    def __init__(self):
        self.corners_xy = []
        self.curr_id = -1
        self.max_corners_init = 7
        self.max_corners = self.max_corners_init
        self.min_dist_relative_init = 0.05
        self.min_dist_relative = self.min_dist_relative_init
        self.min_dist_reduce = 0.8
        self.image_input = None

    def init_world(self, world_image):
        self.reset()
        self.image_input = np.copy(world_image)

    def compute(self):
        assert self.image_input is not None, "Init the world first"
        min_dist = np.linalg.norm(self.image_input.shape[:2]) * self.min_dist_relative
        self.corners_xy = cv2.goodFeaturesToTrack(self.image_input, maxCorners=self.max_corners,
                                                  qualityLevel=0.05, minDistance=min_dist)
        self.corners_xy = np.squeeze(self.corners_xy, axis=1)
        self.display()

    def display(self):
        image_with_corners = cv2.cvtColor(self.image_input, cv2.COLOR_GRAY2BGR)
        for x, y in self.corners_xy:
            cv2.circle(image_with_corners, (x, y), 1, (255, 0, 0), -1)
        image_with_corners = cv2.resize(image_with_corners, (700, 700))
        cv2.imshow("Corners", image_with_corners)

    @staticmethod
    def get_sobel(image):
        image = image.astype(np.uint8)
        gx = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=3)
        grad_ampl = np.sqrt(gx ** 2 + gy ** 2)
        grad_ampl *= 255 / (np.max(grad_ampl) + 1e-7)
        grad_ampl = grad_ampl.astype(np.uint8)
        return grad_ampl

    def __iter__(self):
        return self

    def reset(self):
        self.max_corners = self.max_corners_init
        self.min_dist_relative = self.min_dist_relative_init
        self.image_input = None

    def next(self):
        self.curr_id += 1
        if self.curr_id >= len(self.corners_xy):
            # We haven't looked enough. Look more carefully
            self.min_dist_relative *= self.min_dist_reduce
            self.max_corners += 1
            self.compute()
            self.curr_id = 0
        return self.corners_xy[self.curr_id]
