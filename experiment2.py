# Here I will try to store pairs of features for different fixations
# and for new entry look at memory for match:
# part 1: for different order of fixation of the same digit




import cv2
import load_mnist
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from core.world import World
from core.agent import Agent
from core.layer import SaliencyMap
from utils import cv2_step, apply_blur


images, labels = load_mnist.load_images(images_number=100)
world = World()
poppy = Agent()


label_interest = 0
zeros = images[labels == label_interest]
zero1 = zeros[1]

# plt.imshow(zero1)
# plt.show()

world.add_image(zero1, position=(10, 10))
# poppy.init_world(world)

num_of_fixations = np.shape(world.saliency_map.corners_xy)[0]

l = 10  # how many patterns to store
fixations = np.random.choice(num_of_fixations, size=l, replace=True)


print fixations
layer_size = poppy.cortex.V1.layers['L23'].size

initial_pairs = np.zeros((l, layer_size))   # array for storing pattern pairs encoding
for i, f in enumerate(fixations):
    poppy.sense_data(world, position=world.saliency_map.corners_xy[f])
    initial_pairs[i] = poppy.cortex.V1.layers['L23'].cells


r = 5   # how many patterns an agent view again
new_fixations = np.random.choice(num_of_fixations, size=r, replace=True)
new_pairs = np.zeros((r, layer_size))

for i, f in enumerate(new_fixations):
    poppy.sense_data(world, position=world.saliency_map.corners_xy[f])
    new_pairs[i] = poppy.cortex.V1.layers['L23'].cells

overlap = np.dot(initial_pairs, new_pairs.T)
print overlap
# overlap_self = np.dot(initial_pairs, initial_pairs.T)
plt.imshow(overlap)
plt.show()
