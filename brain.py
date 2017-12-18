import numpy as np
import cv2

from sensor import Sensor
from nupic.encoders.scalar import ScalarEncoder
from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder


class BrainInterface(object):

    def __init__(self, sdr_shape, retina_size):
        self.retina_size = retina_size
        self.sdr_shape = sdr_shape
        self.sensor = Sensor(n_features=6)
        self.weights = np.random.random_sample(size=(np.cumprod(sdr_shape), np.cumprod(retina_size)))
        self.sparsity = 0.1
        scalar_bits_active = max(int(self.sparsity * np.cumprod(sdr_shape)), 1)
        # self.scalar_encoder = ScalarEncoder(w=scalar_bits_active, minval=0, maxval=1, n=np.cumprod(sdr_shape))
        self.scalar_encoder = RandomDistributedScalarEncoder(resolution=0.01, w=scalar_bits_active, n=np.cumprod(sdr_shape))

    def get_sdr_image(self, image):
        return np.random.randint(0, 1, size=self.sdr_shape)

    def get_sdr_move(self, vector_move):
        return np.random.randint(0, 1, size=self.sdr_shape)

    def cut_image(self, image, xy_center):
        return np.random.randint(0, 255, size=(10, 10))

    def process_image(self, image):
        h_retina, w_retina = self.retina_size
        # todo: add blur and smart resize
        image = cv2.resize(image, (w_retina, h_retina))
        xy_locations = self.sensor.extract_feature_locations(image)
        xy_locations.insert(0, xy_locations[0])
        sdr_image_history = []
        sdr_move_history = []
        for xy_prev, xy_curr in zip(xy_locations[:-1], xy_locations[1:]):
            # visual features extraction
            subimage = self.cut_image(image, xy_curr)
            sdr_image = self.get_sdr_image(subimage)
            sdr_image_history.append(sdr_image)

            # motor cortex
            vector_move = np.subtract(xy_curr, xy_prev)
            sdr_move = self.get_sdr_move(vector_move)
            sdr_move_history.append(sdr_move)
        # todo: associate sdr_image_history with sdr_move_history


class Brain(BrainInterface):

    def get_sdr_image(self, image):
        activations = self.weights.dot(image.flatten())
        n_active = max(int(self.sparsity * len(activations)), 1)
        winners = np.argsort(activations)[-n_active:]
        sdr = np.zeros(self.sdr_shape, dtype=np.bool)
        sdr[winners] = 1
        return sdr

    def get_sdr_move(self, vector_move):
        angle = 30 / 360.
        sdr_angle = self.scalar_encoder.encode(angle)
