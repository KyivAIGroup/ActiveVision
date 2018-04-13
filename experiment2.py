# Here I will try to store pairs of features for different fixations
# and for new entry look at memory for match:
# part 1: for different order of fixation of the same digit


import matplotlib.pyplot as plt
import numpy as np

import load_mnist
from core.agent import Agent
from core.world import World


def get_l23_history(n_jumps):
    fixations = np.random.choice(num_of_fixations, size=n_jumps+1, replace=True)
    is_moved = np.r_[True, np.diff(fixations) != 0]
    fixations = fixations[is_moved]
    print(fixations)
    l23_history = []  # array for storing pattern pairs encoding
    for i, f in enumerate(fixations):
        poppy.sense_data(world, position=world.saliency_map.corners_xy[f])
        l23_history.append(poppy.cortex.V1.layers['L23'].cells)
    l23_history.pop(0)  # remove first jump from nowhere
    l23_history = np.vstack(l23_history)
    return l23_history


np.random.seed(26)
images, labels = load_mnist.load_images(images_number=100)
world = World()
poppy = Agent()

label_interest = 0
zeros = images[labels == label_interest]
zero1 = zeros[1]

world.add_image(zero1)

num_of_fixations = np.shape(world.saliency_map.corners_xy)[0]

initial_pairs = get_l23_history(n_jumps=20)   # array for storing pattern pairs encoding
new_pairs = get_l23_history(n_jumps=7)

overlap = np.dot(initial_pairs, new_pairs.T)
print(overlap)
plt.imshow(overlap)
plt.show()
