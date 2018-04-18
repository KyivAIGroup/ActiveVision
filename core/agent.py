import numpy as np
from itertools import permutations

from core.cortex import Cortex, CortexLocationAware, CortexIntersection
from constants import WORLD_CENTER


class Agent(object):
    def __init__(self):
        self.position = np.array(WORLD_CENTER, dtype=float)  # relative to the world
        self.cortex = CortexIntersection()

    def sense_data(self, world, position=None):
        vector_move, retina_image = world.clip_retina(self.cortex.receptive_field_pixels, position)
        self.cortex.compute(retina_image, vector_move)

    def learn_pairs(self, world, label):
        l23_history = []
        for (corner_from, corner_to) in permutations(world.saccades(), 2):
            self.position[:] = corner_from
            self.sense_data(world, position=corner_to)
            if type(self.cortex) is not CortexLocationAware:
                self.cortex.associate(label)
                l23_history.append(self.cortex.V1.layers['L23'].cells.flatten())
        if type(self.cortex) is CortexLocationAware:
            self.cortex.associate(label)
            l23_history.append(self.cortex.V1.layers['L23'].cells.flatten())
        l23_history = np.vstack(l23_history)
        return l23_history
