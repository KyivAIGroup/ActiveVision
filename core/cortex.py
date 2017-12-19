import numpy as np
import matplotlib.pyplot as plt
import cv2

from core.layer import Area, Layer
from core.encoder import LocationEncoder


class Cortex(object):
    def __init__(self, sdr_size=100):
        self.V1 = Area()

        self.V1.add_layer(Layer('L4', shape=sdr_size))
        self.V1.add_layer(Layer('L23', shape=sdr_size))
        self.V1.add_layer(Layer('motor_direction', shape=sdr_size))
        self.V1.add_layer(Layer('motor_amplitude', shape=sdr_size))

        self.location_encoder = LocationEncoder(max_amplitude=28 * np.sqrt(2))
        self.retina = Layer('retina', shape=(28, 28))

        self.V1.layers['L4'].connect_input(self.retina)

        self.V1.layers['L23'].connect_input(self.V1.layers['L4'])
        self.V1.layers['L23'].connect_input(self.V1.layers['motor_direction'])
        self.V1.layers['L23'].connect_input(self.V1.layers['motor_amplitude'])

    def compute(self, retina_image, vector):
        vector = vector[:2]  # ignore z for now
        cv2.imshow("Retina", cv2.resize(retina_image, (300, 300)))
        self.retina.cells = retina_image
        self.V1.layers['L4'].linear_update()
        self.V1.layers['L23'].linear_update()

        self.V1.layers['motor_direction'] = self.location_encoder.encode_phase(vector)
        self.V1.layers['motor_amplitude'] = self.location_encoder.encode_amplitude(vector)

        # todo: associate
        # self.V1.layers['L23'].associate([self.V1.layers['L4'], self.V1.layers['motor_amplitude'], self.V1.layers['motor_direction']])
