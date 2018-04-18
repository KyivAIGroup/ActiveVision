import numpy as np
import cv2

from core.layer import Area, Layer, LabelLayer, LocationAwareLayer
from core.encoder import LocationEncoder, IntEncoder
from constants import IMAGE_SIZE
from utils import cv2_step


class Cortex(object):
    def __init__(self, sdr_size=1000):
        self.receptive_field_pixels = (28, 28)
        self.V1 = Area()
        self.display = False

        self.V1.add_layer(Layer('L4', shape=sdr_size))

        self.V1.add_layer(Layer('L23', shape=sdr_size))
        self.V1.add_layer(Layer('motor_direction', shape=sdr_size))
        self.V1.add_layer(Layer('motor_amplitude', shape=sdr_size))

        self.location_encoder = LocationEncoder(max_amplitude=np.linalg.norm(IMAGE_SIZE),
                                                shape=sdr_size)
        self.label_layer = LabelLayer(name="label", shape=sdr_size)
        self.retina = Layer('retina', shape=self.receptive_field_pixels)

        self.V1.layers['L4'].connect_input(self.retina)

        self.V1.layers['L23'].connect_input(self.V1.layers['motor_direction'])
        self.V1.layers['L23'].connect_input(self.V1.layers['motor_amplitude'])
        self.V1.layers['L23'].connect_input(self.V1.layers['L4'])

    @staticmethod
    def display_retina(retina_image, vector):
        retina_image_show = cv2.cvtColor(retina_image, cv2.COLOR_GRAY2BGR)
        y_c, x_c = np.floor_divide(retina_image.shape, 2)
        location_from = (x_c - vector[0], y_c - vector[1])
        cv2.arrowedLine(retina_image_show, location_from, (x_c, y_c), (0, 255, 0), thickness=1, tipLength=0.2)
        retina_image_show = cv2.resize(retina_image_show, (300, 300))
        cv2.imshow("Retina", retina_image_show)

    def display_activations(self):
        if self.display:
            self.V1.layers['L4'].display()
            self.V1.layers['L23'].display()
            cv2_step()

    def compute(self, retina_image, vector):
        if self.display:
            self.display_retina(retina_image, vector)
        self.retina.cells[:] = retina_image
        self.V1.layers['L4'].linear_update()
        self.V1.layers['motor_direction'].cells = self.location_encoder.encode_phase(vector)
        self.V1.layers['motor_amplitude'].cells = self.location_encoder.encode_amplitude(vector)
        self.V1.layers['L23'].linear_update()
        self.display_activations()

    def reset_activations(self):
        for layer in self.V1.layers.values():
            layer.cells.fill(0)

    def associate(self, label):
        self.label_layer.encode(scalar=label)
        self.V1.layers['L23'].associate(self.label_layer)

    def predict(self):
        self.label_layer.recall(self.V1.layers['L23'])
        label_predicted = self.label_layer.predict()
        return label_predicted


class CortexIntersection(Cortex):

    def compute(self, retina_image, vector):
        if self.display:
            self.display_retina(retina_image, vector)
        self.retina.cells[:] = retina_image
        self.V1.layers['L4'].linear_update()
        self.V1.layers['motor_direction'].cells = self.location_encoder.encode_phase(vector)
        self.V1.layers['motor_amplitude'].cells = self.location_encoder.encode_amplitude(vector)
        self.V1.layers['L23'].linear_update(input_layers=[self.V1.layers['motor_direction'], self.V1.layers['motor_amplitude']], sparsity=0.3, intersection=False)
        self.V1.layers['L23'].linear_update(input_layers=[self.V1.layers['L4']], sparsity=0.05, intersection=True)
        self.display_activations()


class CortexLocationAware(Cortex):
    def __init__(self, sdr_size=1000):
        super(CortexLocationAware, self).__init__(sdr_size=sdr_size)
        self.receptive_field_pixels = (28, 28)
        self.V1 = Area()
        self.V1.add_layer(Layer('L4', shape=sdr_size))
        self.V1.add_layer(LocationAwareLayer('L23', canvas_shape=(50, 50), patch_shape=(20, 20)))
        self.location_encoder = LocationEncoder(max_amplitude=np.linalg.norm(IMAGE_SIZE),
                                                shape=sdr_size)
        self.label_layer = LabelLayer(name="label", shape=sdr_size)
        self.retina = Layer('retina', shape=self.receptive_field_pixels)
        self.V1.layers['L4'].connect_input(self.retina)
        self.V1.layers['L23'].connect_input(self.V1.layers['L4'])

    def compute(self, retina_image, vector):
        if self.display:
            self.display_retina(retina_image, vector)
        self.retina.cells[:] = retina_image
        self.V1.layers['L4'].linear_update()
        self.V1.layers['L23'].linear_update_at_location(vector_move=vector)
        self.display_activations()

    def associate(self, label):
        self.V1.layers['L23'].apply_intersection()
        if self.display:
            self.V1.layers['L23'].display()
            cv2_step()
        super(CortexLocationAware, self).associate(label=label)

    def predict(self):
        self.V1.layers['L23'].apply_intersection()
        return super(CortexLocationAware, self).predict()
