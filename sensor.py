import numpy as np


class Sensor(object):

    def __init__(self, n_features):
        self.n_features = n_features

    def extract_feature_locations(self, image):
        h, w = image.shape[:2]
        xy_centers = []
        for feature_id in range(self.n_features):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            xy_centers.append((x, y))
        return xy_centers
