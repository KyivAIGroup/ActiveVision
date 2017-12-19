# Model with eye movements. Movements go in the direction of next most salient feature, and encoded into SDR
# L23 integrates L4 and L5, visual feature in context of movement

# todo: 17.12.17 Add activation based on clusters 2. Think how to generate second output for classification


import numpy as np
import cv2
import matplotlib.pyplot as plt
from encoder import LocationEncoder

try:
    from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder
except ImportError:
    from nupic_stub import EncoderStub as RandomDistributedScalarEncoder


class World(object):
    def __init__(self):
        self.width_px = 100
        self.height_px = 100
        self.image = np.zeros((self.height_px, self.width_px), dtype=np.uint8)

    def add_image(self, new_image, position):
        # new_image is a two dimensional array
        x, y = position
        im_h, im_w = new_image.shape[:2]
        self.image[y: y+im_h, x: x+im_w] = new_image


class Agent(object):
    def __init__(self):
        self.cortex = Cortex()
        self.saliency_region = SaliencyRegion()

        self.receptive_field_angle_xy = (30, 30)  # in degrees
        # x,y,z. XY plane of an image.
        # do not change z for now, it adjusted with self.visual_field_angle to give 28x28 patch of visual input

        self.position = np.array([10, 10, 25])  # relative to the world
        self.receptive_field_pixels = (2 * self.position[2] * np.tan(np.deg2rad(self.receptive_field_angle_xy))).astype(int)

    def sense_data(self, world):
        fov_w, fov_h = self.receptive_field_pixels
        vector = next(self.saliency_region)
        self.position[:2] += vector
        x, y, z = self.position
        x = np.clip(x, a_min=fov_w // 2, a_max=world.width_px - fov_w // 2)
        y = np.clip(y, a_min=fov_h // 2, a_max=world.height_px - fov_h // 2)
        self.position[:2] = (x, y)
        retina_image = world.image[y-fov_h//2: y+fov_h//2, x-fov_w//2: x+fov_w//2]
        self.cortex.compute(retina_image, vector)


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


class Cortex(object):
    def __init__(self, sdr_size=100):
        self.V1 = Area()

        self.V1.add_layer(Layer('L4', shape=sdr_size))
        self.V1.add_layer(Layer('L23', shape=sdr_size))
        self.V1.add_layer(Layer('motor_direction', shape=sdr_size))
        self.V1.add_layer(Layer('motor_amplitude', shape=sdr_size))

        self.location_encoder = LocationEncoder(max_amplitude=28 * np.sqrt(2))
        self.retina = Layer('retina', shape=(28, 28))

        self.V1.layers['L4'].connect_input(self.retina)

        self.V1.layers['L23'].connect_input(self.V1.layers['L4'])
        self.V1.layers['L23'].connect_input(self.V1.layers['motor_direction'])
        self.V1.layers['L23'].connect_input(self.V1.layers['motor_amplitude'])

    def compute(self, retina_image, vector):
        plt.imshow(retina_image)
        plt.show()
        self.retina.cells = retina_image
        self.V1.layers['L4'].linear_update()
        self.V1.layers['L23'].linear_update()

        self.V1.layers['motor_direction'] = self.location_encoder.encode_phase(vector)
        self.V1.layers['motor_amplitude'] = self.location_encoder.encode_amplitude(vector)

        # todo: associate
        # self.V1.layers['L23'].associate([self.V1.layers['L4'], self.V1.layers['motor_amplitude'], self.V1.layers['motor_direction']])


if __name__ == '__main__':
    import load_mnist

    load_number = 100
    images, labels = load_mnist.load_images(images_number=load_number)

    flat_mnist_world = World()
    flat_mnist_world.add_image(images[0], position=(10, 10))

    poppy = Agent()
    for i in range(7):
        poppy.sense_data(flat_mnist_world)
