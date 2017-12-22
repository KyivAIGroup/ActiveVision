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
        self.weights_lateral = np.zeros((self.size, self.size), dtype=np.float32)
        self.associated = {}
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

    def choose_n_active(self, n):
        active = np.where(self.cells)[0]
        return np.random.choice(active, size=n, replace=False)

    @staticmethod
    def associate_from_to(self, other):
        for cell_active in np.where(self.cells)[0]:
            for cell_root in self.associated[other.name]["self"]:
                if cell_active != cell_root:
                    self.weights_lateral[cell_root, cell_active] += 1
                    self.weights_lateral[cell_active, cell_root] += 1

    def random_activation(self):
        active = np.random.choice(self.size, size=int(self.sparsity * self.size), replace=False)
        sdr = np.zeros(self.cells.shape, dtype=self.cells.dtype)
        sdr.ravel()[active] = 1
        self.cells = sdr

    def associate(self, other):
        if other.name not in self.associated:
            assert self.name not in other.associated, "Association has to be undirected"
            cells_self = self.choose_n_active(3)
            cells_other = other.choose_n_active(3)
            self.associated[other.name] = {
                "self": cells_self,
                "other": cells_other
            }
            other.associated[self.name] = {
                "self": cells_other,
                "other": cells_self
            }
        self.associate_from_to(self, other)
        self.associate_from_to(other, self)

    def display(self, winname=None):
        if winname is None:
            winname = self.name
        map_size = int(np.ceil(np.sqrt(self.size)))
        activation_map = np.zeros((map_size, map_size), dtype=np.uint8)
        v = activation_map.ravel()
        v[:self.size] = self.cells * 255
        activation_map = cv2.resize(activation_map, (300, 300))
        cv2.imshow(winname, activation_map)

    def test_associated(self, input_layer):
        depolarized_cells = np.zeros(self.cells.shape, dtype=np.float32)
        for cell_root in self.associated[input_layer.name]["self"]:
            cell_weights = self.weights_lateral[cell_root]
            depolarize_ids = np.where(cell_weights)[0]
            depolarize_ids = np.argsort(depolarize_ids, kind="mergesort")[::-1]
            depolarize_ids = depolarize_ids[: min(4, len(depolarize_ids))]
            depolarized_cells[depolarize_ids] += 0.5
        self.cells = (depolarized_cells > 0.75).astype(np.int32)
        self.display(winname="{} associated with {}".format(self.name, input_layer.name))

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
        self.compute()

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
