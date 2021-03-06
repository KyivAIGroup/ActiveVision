# Model with eye movements. Movements go in the direction of next most salient feature, and encoded into SDR
# L23 integrates L4 and L5, visual feature in context of movement

import cv2
import numpy as np
from tqdm import tqdm

from core.agent import Agent
from core.world import World
from utils import cv2_step, load_mnist


def run(world, agent, train=True, images_number=1000, digits=(5, 6)):
    images, labels = load_mnist.load_images(images_number, train=train, digits=digits)
    correct = 0
    total = len(labels)
    for image, label in tqdm(zip(images, labels), desc="train={}".format(train)):
        world.add_image(image)
        agent.cortex.reset_activations()
        if train:
            # agent.learn_pairs(world, label=label)
            for corner_xy in world.saccades():
                agent.sense_data(world, position=corner_xy)
            agent.cortex.associate(label)
        else:
            for corner_xy in world.saccades():
                agent.sense_data(world, position=corner_xy)
            label_predicted = agent.cortex.predict()
            correct += label_predicted == label
    if not train:
        accuracy = float(correct) / total
        print("Accuracy: {}".format(accuracy))


def train_test(display=True):
    world = World()
    poppy = Agent()
    poppy.cortex.display = display
    run(world, poppy, train=True)
    run(world, poppy, train=False)


def one_image(label_interest=5):
    images, labels = load_mnist.load_images(images_number=100)

    world = World()
    poppy = Agent()

    image_interest = images[labels == label_interest][0]
    world.add_image(image_interest)
    poppy.cortex.reset_activations()
    poppy.cortex.display = True

    while True:
        poppy.sense_data(world)
        poppy.cortex.associate(label=label_interest)


def learn_pairs(label_interest=5, n_jumps_test=50):
    """
    :param label_interest: MNIST label of interest
    :param n_jumps_test: how many test saccades to be made for one image;
                         as we increase `n_jumps_test`, we expect overlap with L23 train history to decrease in time,
                         since during the training we observe only the most significant features in an image.
                         Ideally, we'd like the overlap not to decrease much in time.
    """
    images, labels = load_mnist.load_images(images_number=100)

    world = World()
    poppy = Agent()

    images_interest = images[labels == label_interest]
    for image in images_interest:
        world.add_image(image)
        poppy.cortex.reset_activations()
        l23_train = poppy.learn_pairs(world, label_interest)
        world.reset()
        if n_jumps_test == 0:
            l23_test = poppy.learn_pairs(world, label=label_interest)
        else:
            l23_test = []
            poppy.sense_data(world)
            for saccade in range(n_jumps_test):
                poppy.sense_data(world)
                l23_test.append(poppy.cortex.V1.layers['L23'].cells.copy())
            l23_test = np.vstack(l23_test)
        overlap = np.dot(l23_train, l23_test.T)
        overlap = (overlap * 255 / poppy.cortex.V1.layers['L23'].n_active).astype(np.uint8)
        cv2.imshow('overlap', overlap)
        cv2_step()


if __name__ == '__main__':
    np.random.seed(26)
    # learn_pairs()
    one_image()
