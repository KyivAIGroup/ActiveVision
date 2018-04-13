import numpy as np

from core.layer import SaliencyMap
from utils import apply_blur


class LabeledImage(object):
    def __init__(self, image, label):
        self.image = image.copy()
        self.label = label


class World(object):
    def __init__(self):
        self.width_px = 100
        self.height_px = 100
        self.image = np.zeros((self.height_px, self.width_px), dtype=np.uint8)
        self.saliency_map = SaliencyMap()

    def add_image(self, new_image, position):
        # new_image is a two dimensional array
        x, y = position
        im_h, im_w = new_image.shape[:2]
        self.image[y: y+im_h, x: x+im_w] = new_image
        self.reset()

    def reset(self):
        self.saliency_map.init_world(self.image)

    def clip_retina(self, receptive_field_pixels, position=None):
        if position is None:
            position = next(self.saliency_map)
        fov_w, fov_h = receptive_field_pixels
        x, y = position
        x = np.clip(x, a_min=fov_w // 2, a_max=self.width_px - fov_w // 2)
        y = np.clip(y, a_min=fov_h // 2, a_max=self.height_px - fov_h // 2)
        retina_image = self.image[y - fov_h // 2: y + fov_h // 2, x - fov_w // 2: x + fov_w // 2].copy()
        # retina_image = apply_blur(retina_image)
        return (x, y), retina_image
