# Model with eye movements. Movements go in the direction of next most salient feature, and encoded into SDR
# L23 integrates L4 and L5, visual feature in context of movement

import cv2
import load_mnist

from core.world import World
from core.agent import Agent

# todo: 17.12.17 Add activation based on clusters 2. Think how to generate second output for classification


if __name__ == '__main__':
    load_number = 100
    images, labels = load_mnist.load_images(images_number=load_number)

    flat_mnist_world = World()
    flat_mnist_world.add_image(images[0], position=(10, 10))

    poppy = Agent()
    poppy.init_world(flat_mnist_world)
    while True:
        poppy.sense_data(flat_mnist_world)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break