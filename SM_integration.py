# Model with eye movements. Movements go in the direction of next most salient feature, and encoded into SDR
# L23 integrates L4 and L5, visual feature in context of movement

import cv2
import load_mnist
from tqdm import tqdm

from core.world import World
from core.agent import Agent
from utils import cv2_step

# todo: 17.12.17 Add activation based on clusters 2. Think how to generate second output for classification


def run(world, agent, train=True, images_number=1000):
    images, labels = load_mnist.load_images(images_number, train)
    correct = 0
    total = 0
    for digit in (5, 6):
        images_digit = images[labels == digit]
        total += len(images_digit)
        for im in tqdm(images_digit, "Processing digit={}, learn={}".format(digit, train)):
            world.add_image(im, position=(10, 10))
            for saccade in range(7):
                agent.sense_data(world)
                if train:
                    agent.cortex.associate(digit)
            if not train:
                label_predicted = agent.cortex.predict()
                correct += label_predicted == digit
    if not train:
        accuracy = float(correct) / total
        print("Accuracy: {}".format(accuracy))


def train_test():
    world = World()
    poppy = Agent()
    run(world, poppy, train=True)
    run(world, poppy, train=False)


def one_image(label_interest=5):
    images, labels = load_mnist.load_images(images_number=100)

    world = World()
    poppy = Agent()

    image_interest = images[labels == label_interest][0]
    world.add_image(image_interest, position=(10, 10))

    while True:
        poppy.sense_data(world)
        poppy.cortex.associate(label=label_interest)
        # poppy.cortex.label_layer.display()
        cv2_step()


if __name__ == '__main__':
    one_image()
