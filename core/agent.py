import numpy as np

from core.cortex import Cortex


class Agent(object):
    def __init__(self):
        self.cortex = Cortex()

        self.receptive_field_angle_xy = (30, 30)  # in degrees
        # x,y,z. XY plane of an image.
        # do not change z for now, it adjusted with self.visual_field_angle to give 28x28 patch of visual input

        self.position = np.array([10, 10, 25])  # relative to the world
        self.last_position = np.copy(self.position)
        self.receptive_field_pixels = (2 * self.position[2] * np.tan(np.deg2rad(self.receptive_field_angle_xy))).astype(int)

    def sense_data(self, world, position=None):
        new_position_xy, retina_image = world.clip_retina(self.receptive_field_pixels)
        self.position[:2] = new_position_xy
        if position.any():
            new_position_xy = position
        vector_move = self.position - self.last_position
        self.last_position = np.copy(self.position)
        self.cortex.compute(retina_image, vector_move, display=True)
