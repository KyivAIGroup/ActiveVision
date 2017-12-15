import numpy as np

from brain import Brain
from sensor import Sensor


class Agent(object):

    def __init__(self, receptive_field_fov=(30, 30), sdr_shape=(30, 30)):
        self.receptive_field_fov = receptive_field_fov
        self.brain = Brain(sdr_shape)
        self.sensor = Sensor(n_features=6)
        self.world_position = (0, 0, 0)

    def cut_image(self, image, xy_center):
        return np.random.randint(0, 255, size=(10, 10))

    def process_image(self, image):
        xy_locations = self.sensor.extract_feature_locations(image)
        xy_locations.insert(0, xy_locations[0])
        sdr_image_history = []
        sdr_move_history = []
        for xy_prev, xy_curr in zip(xy_locations[:-1], xy_locations[1:]):
            # visual features extraction
            subimage = self.cut_image(image, xy_curr)
            sdr_image = self.brain.get_sdr_image(subimage)
            sdr_image_history.append(sdr_image)

            # motor cortex
            vector_move = np.subtract(xy_curr, xy_prev)
            sdr_move = self.brain.get_sdr_move(vector_move)
            sdr_move_history.append(sdr_move)
        # todo: associate sdr_image_history with sdr_move_history
