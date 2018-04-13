import numpy as np
from itertools import permutations

from core.cortex import Cortex
from constants import WORLD_CENTER


class Agent(object):
    def __init__(self):
        self.position = np.array(WORLD_CENTER, dtype=float)  # relative to the world
        self.cortex = Cortex()

    def sense_data(self, world, position=None, display=False):
        vector_move, retina_image = world.clip_retina(self.cortex.receptive_field_pixels, position)
        self.cortex.compute(retina_image, vector_move, display=display)

    def learn_pairs(self, world, label=None):
        l23_history = []
        corners_xy = world.saliency_map.corners_xy.copy()
        for (corner_from, corner_to) in permutations(corners_xy, 2):
            self.position[:] = corner_from
            self.sense_data(world, position=corner_to)
            l23_history.append(self.cortex.V1.layers['L23'].cells.copy())
            if label is not None:
                self.cortex.associate(label)
        l23_history = np.vstack(l23_history)
        return l23_history
