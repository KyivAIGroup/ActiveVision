import numpy as np
import cv2

from core.layer import Area, Layer, LabelLayer
from core.encoder import LocationEncoder, IntEncoder


class Cortex(object):
    def __init__(self, sdr_size=100, with_motor_cortex=True):
        self.V1 = Area()

        self.V1.add_layer(Layer('L4', shape=sdr_size))
        self.V1.add_layer(Layer('L23', shape=sdr_size))
        self.V1.add_layer(Layer('motor_direction', shape=sdr_size))
        self.V1.add_layer(Layer('motor_amplitude', shape=sdr_size))

        self.location_encoder = LocationEncoder(max_amplitude=28 * np.sqrt(2))
        self.label_layer = LabelLayer(name="label", shape=100)
        self.retina = Layer('retina', shape=(28, 28))

        self.V1.layers['L4'].connect_input(self.retina)

        self.V1.layers['L23'].connect_input(self.V1.layers['L4'])
        if with_motor_cortex:
            self.V1.layers['L23'].connect_input(self.V1.layers['motor_direction'])
            self.V1.layers['L23'].connect_input(self.V1.layers['motor_amplitude'])

    @staticmethod
    def display_retina(retina_image, vector):
        retina_image_show = cv2.cvtColor(retina_image, cv2.COLOR_GRAY2BGR)
        y_c, x_c = np.floor_divide(retina_image.shape, 2)
        location_from = (x_c - vector[0], y_c - vector[1])
        cv2.arrowedLine(retina_image_show, location_from, (x_c, y_c), (0, 255, 0), thickness=1, tipLength=0.2)
        retina_image_show = cv2.resize(retina_image_show, (300, 300))
        cv2.imshow("Retina", retina_image_show)

    def compute(self, retina_image, vector, display=False):
        # todo: cut image to fit retina
        if display:
            self.display_retina(retina_image, vector)
        vector = vector[:2]  # ignore z for now
        self.retina.cells = retina_image
        self.V1.layers['L4'].linear_update()
        self.V1.layers['L23'].linear_update()

        self.V1.layers['motor_direction'].cells = self.location_encoder.encode_phase(vector)
        self.V1.layers['motor_amplitude'].cells = self.location_encoder.encode_amplitude(vector)

    def associate(self, label):
        self.label_layer.encode(scalar=label)
        self.V1.layers['L23'].associate(self.label_layer)

    def predict(self):
        self.label_layer.recall(self.V1.layers['L23'])
        label_predicted = self.label_layer.predict()
        return label_predicted
