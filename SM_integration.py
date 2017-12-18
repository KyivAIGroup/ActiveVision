# Model with eye movements. Movements go in the direction of next most salient feature, and encoded into SDR
# L23 integrates L4 and L5, visual feature in context of movement

# todo: 17.12.17 Add activation based on clusters 2. Think how to generate second output for classification


import numpy as np
import cv2
import matplotlib.pyplot as plt
from sdr_drom_flaot import NumberEncoder


class World():
    def __init__(self):
        self.visual_field_size = (100, 100)
        self.visual_field = np.zeros(self.visual_field_size)

        self.agent = 0

    def add_visual_data(self, data, position):
        # data is a two dimensional array = image
        x, y = position
        x_m, y_m = data.shape
        self.visual_field[x:x+x_m, y:y+y_m] += data



class Agent():
    def __init__(self):
        self.input_size_angle = (30, 30)  # in degrees
        # x,y,z. XY plane of an image.
        # do not change z for now, it adjusted with self.visual_field_angle to give 28x28 patch of visual input

        self.position = np.array([10, 10, 25])  # relative to the world
        self.input_size_pixel = (2 * self.position[2] * np.tan(np.deg2rad(self.input_size_angle))).astype(int)

    def sense_data(self, world):
        x, y, z = self.position
        x_m, y_m = self.input_size_pixel
        return world.visual_field[x:x+x_m, y:y+y_m]

class Area:
    def __init__(self):
        self.layer = {}

    def add_layer(self, name, layer):
        self.layer[name] = layer


class Layer:
    def __init__(self, shape):
        self.cells = np.zeros(shape, dtype=int)
        self.shape = shape  # in case of two dimensions
        self.size = np.prod(self.shape)

        self.input_layers = []
        self.weights = []
        # association memory params
        self.cluster_size = 2
        self.clusters = 'tbd'

    def connect(self, with_layer):
        self.input_layers.append(with_layer)
        self.weights.append(np.random.rand(self.size, with_layer.size))

    def linear_update(self):
        for i, layer in enumerate(self.input_layers):
            self.cells = self.cells + np.dot(self.weights[i], layer.cells.flatten())

        self.cells = self.kWTA(self.cells, 0.1)

    def cluster_activation(self, clusters, input):
        dst = clusters[:, np.nonzero(input)[0]]
        # find counts of input*distance to see if there is a cluster
        data = [np.unique(dt[np.nonzero(dt)[0], np.nonzero(dt)[1]], return_counts=True) for dt in dst]
        # find max in each count to see if there is cluster (2 if there is a cluster)
        data2 = np.array([np.max(counts) if counts.size else 0 for counts in np.array(data)[:, 1]])
        # cells = data2 >= np.sort(data2)[-int(self.size*self.activation)]
        cells = (data2 == 2).astype(int)
        # print np.count_nonzero(cells)
        return cells

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

    def kWTA(self, vector, k=0.1):
        # k - percent of active neurons to leave
        vector[vector.argsort()[:-int(vector.size * k)]] = 0
        vector[vector.argsort()[-int(vector.size * k):]] = 1
        return vector.astype(int)

class Cortex:
    def __init__(self):
        self.V1 = Area()

        self.V1.add_layer('L4', Layer(shape=100))
        self.V1.add_layer('L23', Layer(shape=100))
        self.V1.add_layer('motor_direction', Layer(shape=100))
        self.V1.add_layer('motor_amplitude', Layer(shape=100))

        self.V1.add_layer('saliency', Layer(shape=(28, 28)))
        self.retina = Layer(shape=(28, 28))

        self.number_encoder = NumberEncoder(sdr_size=100)

        self.V1.layer['L4'].connect(self.retina)

        self.V1.layer['L23'].connect(self.V1.layer['L4'])
        self.V1.layer['L23'].connect(self.V1.layer['motor_direction'])
        self.V1.layer['L23'].connect(self.V1.layer['motor_amplitude'])


    def salience_map(self, img):
        # result = np.zeros((img.shape[0], img.shape[1]), dtype='float64')
        result = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1,0,ksize=3)) + np.abs(cv2.Sobel(img, cv2.CV_64F, 0, 1,ksize=3))
        if np.max(result):
            result = result/np.max(result)
        thresh = result > 0.5
        return result * thresh

    def get_vector_to_saliency(self, s_map):
        x_c, y_c = int(s_map.shape[0]/2), int(s_map.shape[1]/2)

        mask = np.ones(s_map.shape, dtype=int)
        mask[x_c-5:x_c+5, y_c-5:y_c+5] = 0
        s_map *= mask    # move somewhere else not to current place

        x, y = np.unravel_index(s_map.argmax(), s_map.shape)
        plt.imshow(s_map)
        plt.show()
        print x, y
        return x - x_c, y - y_c

    def get_phase_amp(self, vector):
        x, y = vector
        return np.rad2deg(np.arctan(float(x)/y)), np.sqrt(x ** 2 + y ** 2)

    def saccade(self):
        self.V1.layer['saliency'].cells = self.salience_map(self.retina.cells)
        vector = self.get_vector_to_saliency(self.V1.layer['saliency'].cells)

        # this is one of the output of the brain
        poppy.position[:2] += vector  # rewrite somehow to exclude explicit name of the agent
        self.retina.cells = poppy.sense_data(flat_mnist_world)

        angle, amplitude = self.get_phase_amp(vector)
        h, w = 30, 30
        max_amplitude = np.sqrt(h ** 2 + w ** 2)
        max_angle = 180
        self.V1.layer['motor_direction'] = self.number_encoder.encode(angle/max_angle)
        self.V1.layer['motor_amplitude'] = self.number_encoder.encode(amplitude/max_amplitude)

    def compute(self):
        self.V1.layer['L4'].linear_update()
        self.V1.layer['L23'].linear_update()

        self.V1.layer['L23'].associate([self.V1.layer['L4'], self.V1.layer['motor_amplitude'], self.V1.layer['motor_direction']])
        self.saccade()


if __name__ == '__main__':
    import load_mnist

    load_number = 100
    images, labels = load_mnist.load_images(images_number=load_number)

    images = images.astype(int)
    # images = images.reshape(load_number, -1)
    # images /= np.max(images, axis=1)[:, None]
    # images[images > 0.1] = 1
    # images[images < 0.1] = 0


    flat_mnist_world = World()
    flat_mnist_world.add_visual_data(images[0], position=(10, 10))

    poppy = Agent()
    poppy.brain = Cortex()


    poppy_init_position = np.copy(poppy.position)
    poppy.brain.retina.cells = poppy.sense_data(flat_mnist_world)

    for i in range(7):
        poppy.brain.compute()

        # plt.imshow(poppy.brain.salience_map(poppy.sense_data(flat_mnist_world)))
        # plt.show()
