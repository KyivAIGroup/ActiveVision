import numpy as np


class World(object):

    def __init__(self, world_size):
        self.world_size = world_size
        self.visual_field = np.zeros(world_size, dtype=np.uint8)

    def add_image(self, image, position):
        x, y = position
        h_im, w_im = image.shape[:2]
        self.visual_field[y:y+h_im, x:x+w_im] = image
