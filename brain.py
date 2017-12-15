import numpy as np


class Brain(object):

    def __init__(self, sdr_shape):
        self.sdr_shape = sdr_shape

    def get_sdr_image(self, image):
        return np.random.randint(0, 1, size=self.sdr_shape)

    def get_sdr_move(self, vector_move):
        return np.random.randint(0, 1, size=self.sdr_shape)
