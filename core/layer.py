import numpy as np
import cv2

from core.encoder import IntEncoder
from constants import IMAGE_SIZE


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
            activations.append(connected_layer.cells.flatten())

    def recall(self, layer_from, layer_to):
        history = np.array(self.patterns[layer_from])
        overlap = history.dot(layer_from.cells.flatten())
        winner = np.argmax(overlap)
        return self.patterns[layer_to][winner]


class Layer(object):
    def __init__(self, name, shape, sparsity=0.1):
        self.name = name
        self.cells = np.zeros(shape, dtype=np.int32)
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
    def shape(self):
        return self.cells.shape

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def n_active(self):
        return max(1, int(self.size * self.sparsity))

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

    def linear_update(self, input_layers=None, sparsity=None, intersection=False):
        if input_layers is None:
            input_layers = self.input_layers
        if sparsity is None:
            sparsity = self.sparsity
        signal = np.zeros(self.cells.shape, dtype=np.int32)
        for layer in input_layers:
            layer_id = self.input_layers.index(layer)
            signal += np.dot(self.weights[layer_id], layer.cells.flatten())
        if intersection:
            signal &= self.cells
        self.cells = self.kWTA(signal, sparsity)
        # active_ids = np.where(self.cells)[0]
        # for input_weights in self.weights:
        #     input_weights[active_ids, :] = input_weights[active_ids, :] + 1

    def choose_n_active(self, n):
        active = np.where(self.cells)[0]
        return np.random.choice(active, size=n, replace=False)

    def get_sparse_bits_count(self):
        return int(self.sparsity * self.size)

    def random_activation(self):
        active = np.random.choice(self.size, size=self.get_sparse_bits_count(), replace=False)
        sdr = np.zeros(self.shape, dtype=self.cells.dtype)
        sdr.ravel()[active] = 1
        self.cells = sdr

    def associate(self, other):
        self.memory.remember_activations(other)

    def recall(self, other):
        self.cells = self.memory.recall(other, self)

    def display(self, winname=None):
        if winname is None:
            winname = self.name
        activations = self.cells
        if activations.ndim == 1:
            side_size = int(np.ceil(np.sqrt(self.size)))
            activations = np.r_[activations, np.zeros(side_size ** 2 - len(activations), dtype=activations.dtype)]
            activations.resize(side_size, side_size)
        else:
            assert activations.ndim == 2
        max_element = max(activations.max(), 1)
        activations = activations / float(max_element) * 255
        activations = cv2.resize(activations, (300, 300)).astype(np.uint8)
        cv2.imshow(winname, activations)

    @staticmethod
    def kWTA(cells, sparsity):
        n_active = max(int(sparsity * cells.size), 1)
        n_active = min(n_active, len(cells.nonzero()[0]))
        winners = np.argsort(cells.flatten())[-n_active:]
        sdr = np.zeros(cells.size, dtype=cells.dtype)
        sdr[winners] = 1
        sdr = sdr.reshape(cells.shape)
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


class LocationAwareLayer(Layer):

    def __init__(self, name, canvas_shape, patch_shape, sparsity=0.1):
        assert len(canvas_shape) == len(patch_shape) == 2
        super(LocationAwareLayer, self).__init__(name=name, shape=canvas_shape, sparsity=sparsity)
        self.patch_shape = np.array(patch_shape, dtype=int)
        self.canvas_shape = np.array(canvas_shape, dtype=int)

    def connect_input(self, new_layer):
        self.input_layers.append(new_layer)
        self.weights.append(np.random.binomial(1, self.sparsity_weights, size=(self.patch_shape.prod(), new_layer.size)))

    def get_patch_position(self, vector_move):
        vector_normed = np.divide(vector_move, np.array(IMAGE_SIZE, dtype=float))
        vec_xc, vec_yc = vector_normed * (self.canvas_shape - self.patch_shape) / 2
        position_patch_center = self.canvas_shape / 2. + np.array([vec_yc, vec_xc])
        position_patch = position_patch_center - self.patch_shape / 2.
        position_patch = position_patch.astype(int)
        return position_patch

    def linear_update_at_location(self, vector_move):
        y0, x0 = self.get_patch_position(vector_move)
        y1, x1 = y0 + self.patch_shape[0], x0 + self.patch_shape[1]
        for layer_id, layer in enumerate(self.input_layers):
            signal = np.dot(self.weights[layer_id], layer.cells.flatten())
            signal = self.kWTA(signal, sparsity=self.sparsity)
            signal = signal.reshape(self.patch_shape)
            self.cells[y0: y1, x0: x1] += signal

    def apply_intersection(self):
        self.cells[self.cells <= 1] = 0  # signal at intersection is greater than 1
        self.cells = self.kWTA(self.cells, sparsity=self.sparsity)
