import numpy as np

from brain import BrainInterface, Brain
from sensor import Sensor


class Agent(object):

    def __init__(self, receptive_field_fov_xy=(30, 30), sdr_shape=(30, 30)):
        self.retina_size = (10, 10)
        self.receptive_field_fov = receptive_field_fov_xy
        self.brain = Brain(sdr_shape, self.retina_size)
        self.position = (10, 10, 25)
        self.sensor = Sensor(n_features=6)

    def sense_data(self, world):
        x, y, z = self.position
        fov_width, fov_height = (2 * z * np.tan(np.deg2rad(self.receptive_field_fov))).astype(int)
        fov_image = world.image[y-fov_height//2: y+fov_height//2, x-fov_width//2: x+fov_width//2]
        return fov_image

    def process_image(self, image):
        h_retina, w_retina = self.retina_size
        xy_locations = self.sensor.extract_feature_locations(image)
        # todo: add blur and smart resize
        image = cv2.resize(image, (w_retina, h_retina))
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
