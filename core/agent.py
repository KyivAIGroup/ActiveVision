import numpy as np

from core.cortex import Cortex
from core.layer import SaliencyMap


class Agent(object):
    def __init__(self):
        self.cortex = Cortex()
        self.saliency_map = SaliencyMap()

        self.receptive_field_angle_xy = (30, 30)  # in degrees
        # x,y,z. XY plane of an image.
        # do not change z for now, it adjusted with self.visual_field_angle to give 28x28 patch of visual input

        self.position = np.array([10, 10, 25])  # relative to the world
        self.last_position = np.copy(self.position)
        self.receptive_field_pixels = (2 * self.position[2] * np.tan(np.deg2rad(self.receptive_field_angle_xy))).astype(int)

    def init_world(self, world):
        self.saliency_map.init_world(world.image)

    def sense_data(self, world):
        fov_w, fov_h = self.receptive_field_pixels
        new_position = next(self.saliency_map)
        x, y, z = new_position
        x = np.clip(x, a_min=fov_w // 2, a_max=world.width_px - fov_w // 2)
        y = np.clip(y, a_min=fov_h // 2, a_max=world.height_px - fov_h // 2)
        self.position[:2] = (x, y)
        vector_move = self.position - self.last_position
        self.last_position = np.copy(self.position)
        retina_image = world.image[y-fov_h//2: y+fov_h//2, x-fov_w//2: x+fov_w//2]
        self.cortex.compute(retina_image, vector_move)
