# Model with eye movements. Movements go in the direction of next most salient feature, and encoded into SDR
# L23 integrates L4 and L5, visual feature in context of movement

import cv2
import numpy as np
import load_mnist
from tqdm import tqdm

from core.world import World
from core.agent import Agent

# todo: 17.12.17 Add activation based on clusters 2. Think how to generate second output for classification


def cv2_step():
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        quit()


def run(world, agent, train=True, images_number=1000):
    images, labels = load_mnist.load_images(images_number, train)
    correct = 0
    total = 0
    for digit in (5, 6):
        images_digit = images[labels == digit]
        total += len(images_digit)
        for im in tqdm(images_digit, "Processing digit={}, learn={}".format(digit, train)):
            world.add_image(im, position=(10, 10))
            agent.init_world(world)
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
    poppy.init_world(world)

    while True:
        poppy.sense_data(world)
        poppy.cortex.associate(label=label_interest)
        poppy.cortex.label_layer.display()
        cv2_step()


def test_translate(label_interest=5, display=False):
    images, labels = load_mnist.load_images(images_number=1000)
    poppy = Agent()
    overlaps = []
    translation_x = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    for im in tqdm(images[labels == label_interest], desc="Translation test"):
        h, w = im.shape[:2]
        im_translated = cv2.warpAffine(im, translation_x, (w, h))
        l4_sdr = []
        for img_try, title in zip((im, im_translated), ("Orig", "Translated")):
            poppy.cortex.compute(img_try, vector=(0, 0, 0))
            if display:
                poppy.cortex.V1.layers['L4'].display(winname=title)
            sdr = poppy.cortex.V1.layers['L4'].cells
            l4_sdr.append(sdr)
        if display:
            im_diff = np.abs(im.astype(np.int32) - im_translated.astype(np.int32)).astype(np.uint8)
            cv2.imshow("Translated diff", im_diff)
            cv2_step()
        assert sum(l4_sdr[0]) == sum(l4_sdr[1])
        overlap = sum(np.logical_and(l4_sdr[0], l4_sdr[1]))
        overlap /= float(sum(l4_sdr[0]))
        overlaps.append(overlap)
    print("Overlap mean={:.4f} std={:.4f}".format(np.mean(overlaps), np.std(overlaps)))


if __name__ == '__main__':
    test_translate(display=False)
