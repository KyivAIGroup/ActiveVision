import numpy as np
import cv2

from core.encoder import IntEncoder


class Area(object):
    def __init__(self):
        self.layers = {}

    def add_layer(self, layer):
        self.layers[layer.name] = layer


class AssociationMemory(object):
    def __init__(self, layer):
        self.patterns = {
            layer: []
        }

    def remember_activations(self, layer_paired):
        layer_paired.memory = self
        if layer_paired not in self.patterns:
            self.patterns[layer_paired] = []
        for connected_layer, activations in self.patterns.items():
            activations.append(connected_layer.cells.copy())

    def recall(self, layer_from, layer_to):
        history = np.array(self.patterns[layer_from])
        overlap = history.dot(layer_from.cells)
        largest_overlap_patterns = np.take(self.patterns[layer_to], np.where(overlap == np.max(overlap))[0])
        unique, counts = np.unique(largest_overlap_patterns, return_counts=True)
        pattern = unique[np.argmax(counts)]
        return pattern


class Layer(object):
    def __init__(self, name, shape, sparsity=0.1):
        self.name = name
        self.cells = np.zeros(shape, dtype=np.int32)
        self.size = np.prod(shape)
        self.input_layers = []
        self.weights = []
        self.associated = {}
        self.sparsity = sparsity
        self.sparsity_weights = 0.2
        self.memory = AssociationMemory(self)

        # Integration params. used only in self.integrate()
        self.sparsity_2 = 0.07  # sparsity of the integration layer
        self.tau = 0.3
        self.Y = np.zeros((10, self.size))
        self.Y_exc = np.zeros(self.size)
        self.i = 0

        # association memory params
        self.cluster_size = 2
        self.clusters = 'tbd'

    @property
    def n_active(self):
        return int(len(self.cells) * self.sparsity)

    def connect_input(self, new_layer):
        self.input_layers.append(new_layer)
        # self.weights.append(np.random.rand(self.size, new_layer.size))
        self.weights.append(np.random.binomial(1, self.sparsity_weights, size=(self.size, new_layer.size)))

    def integrate(self):
        # exeprimental feature for sequence recognition/integration
        for layer_id, layer in enumerate(self.input_layers):
            self.Y[self.i] += np.dot(self.weights[layer_id], layer.cells.flatten()) * (1 + self.Y_exc)

        self.Y[self.i] = self.kWTA(self.Y[self.i], self.sparsity)
        self.Y_exc += (self.Y[self.i] - self.Y_exc) * self.tau
        self.i += 1

    def linear_update(self):
        signal = np.zeros(self.cells.shape, dtype=np.float32)
        for layer_id, layer in enumerate(self.input_layers):
            signal += np.dot(self.weights[layer_id], layer.cells.flatten())
        self.cells = self.kWTA(signal, self.sparsity)
        # for input_weights in self.weights:
        #     input_weights[active_ids, :] = input_weights[active_ids, :] + 1

    def choose_n_active(self, n):
        active = np.where(self.cells)[0]
        return np.random.choice(active, size=n, replace=False)

    def get_sparse_bits_count(self):
        return int(self.sparsity * self.size)

    def random_activation(self):
        active = np.random.choice(self.size, size=self.get_sparse_bits_count(), replace=False)
        sdr = np.zeros(self.cells.shape, dtype=self.cells.dtype)
        sdr.ravel()[active] = 1
        self.cells = sdr

    def associate(self, other):
        self.memory.remember_activations(other)

    def recall(self, other):
        self.cells = self.memory.recall(other, self)

    def display(self, winname=None):
        if winname is None:
            winname = self.name
        map_size = int(np.ceil(np.sqrt(self.size)))
        activation_map = np.zeros((map_size, map_size), dtype=np.uint8)
        v = activation_map.ravel()
        v[:self.size] = self.cells * 255
        activation_map = cv2.resize(activation_map, (300, 300))
        cv2.imshow(winname, activation_map)

    def kWTA(self, cells, sparsity):
        n_active = max(int(sparsity * cells.size), 1)
        winners = np.argsort(cells)[-n_active:]
        sdr = np.zeros(cells.shape, dtype=self.cells.dtype)
        sdr[winners] = 1
        return sdr


class LabelLayer(Layer, IntEncoder):

    def __init__(self, name, shape):
        Layer.__init__(self, name, shape)
        IntEncoder.__init__(self, size=shape, sparsity=0.1, bins=9, similarity=0.8)

    def encode(self, scalar):
        label_sdr = super(LabelLayer, self).encode(scalar)
        self.cells = label_sdr
        return label_sdr

    def predict(self):
        label_predicted = self.decode(self.cells)
        return label_predicted

    def display(self, winname=None):
        if winname is None:
            winname = "{} {}".format(self.name, self.predict())
        super(LabelLayer, self).display(winname)


class SaliencyMap(object):

    max_corners_init = 7
    min_dist_relative_init = 0.05

    def __init__(self):
        self.corners_xy = []
        self.curr_id = -1
        self.max_corners = self.max_corners_init
        self.min_dist_relative = self.min_dist_relative_init
        self.min_dist_reduce = 0.8
        self.image_input = None

    def init_world(self, world_image):
        self.reset()
        self.image_input = np.copy(world_image)
        self.compute()

    def compute(self):
        assert self.image_input is not None, "Init the world first"
        min_dist = np.linalg.norm(self.image_input.shape[:2]) * self.min_dist_relative
        self.corners_xy = cv2.goodFeaturesToTrack(self.image_input, maxCorners=self.max_corners,
                                                  qualityLevel=0.05, minDistance=min_dist)
        self.corners_xy = np.squeeze(self.corners_xy, axis=1).astype(np.int32)

    def display(self):
        image_with_corners = cv2.cvtColor(self.image_input, cv2.COLOR_GRAY2BGR)
        for x, y in self.corners_xy:
            cv2.circle(image_with_corners, (x, y), radius=1, color=(255, 0, 0), thickness=-1)
        image_with_corners = cv2.resize(image_with_corners, (700, 700))
        cv2.imshow("Corners", image_with_corners)

    def __iter__(self):
        return self

    def reset(self):
        self.corners_xy = []
        self.curr_id = -1
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
