import numpy as np
from itertools import permutations

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
        new_position_xy, retina_image = world.clip_retina(self.receptive_field_pixels, position)
        self.position[:2] = new_position_xy
        # vector_move = self.position - self.last_position
        # self.last_position = np.copy(self.position)
        image_center = (24, 24)
        vector_move = self.position[:2] - image_center
        self.cortex.compute(retina_image, vector_move, learning=False)

    def learn_pairs(self, world, label=None):
        l23_history = []
        corners_xy = world.saliency_map.corners_xy.copy()
        for (corner_from, corner_to) in permutations(corners_xy, 2):
            self.position[:2] = corner_from
            self.sense_data(world, position=corner_to)
            l23_history.append(self.cortex.V1.layers['L23'].cells.copy())
            if label is not None:
                self.cortex.associate(label)
        l23_history = np.vstack(l23_history)
        return l23_history
