import numpy as np


class LabeledImage(object):
    def __init__(self, image, label):
        self.image = image.copy()
        self.label = label


class World(object):
    def __init__(self):
        self.width_px = 100
        self.height_px = 100
        self.image = np.zeros((self.height_px, self.width_px), dtype=np.uint8)

    def add_image(self, new_image, position):
        # new_image is a two dimensional array
        x, y = position
        im_h, im_w = new_image.shape[:2]
        self.image[y: y+im_h, x: x+im_w] = new_image
