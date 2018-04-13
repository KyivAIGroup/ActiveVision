# Here I will try to store pairs of features for different fixations
# and for new entry look at memory for match:
# part 1: for different order of fixation of the same digit

# here I will try to apply learning


import matplotlib.pyplot as plt
import numpy as np

import load_mnist
from core.agent import Agent
from core.world import World


def get_l23_history(n_jumps):
    fixations = np.random.choice(num_of_fixations, size=n_jumps+1, replace=True)
    is_moved = np.r_[True, np.diff(fixations) != 0]
    # print fixations
    # print np.diff(fixations)
    # print is_moved
    fixations = fixations[is_moved]
    print(fixations)
    l23_history = []  # array for storing pattern pairs encoding
    for i, f in enumerate(fixations):
        poppy.sense_data(world, position=world.saliency_map.corners_xy[f])
        l23_history.append(poppy.cortex.V1.layers['L23'].cells)
    l23_history.pop(0)  # remove first jump from nowhere
    l23_history = np.vstack(l23_history)
    return l23_history


# np.random.seed(26)
images, labels = load_mnist.load_images(images_number=100)
world = World()
poppy = Agent()

label_interest = 0
zeros = images[labels == label_interest]
zero1 = zeros[1]

world.add_image(zero1, position=(10, 10))

num_of_fixations = np.shape(world.saliency_map.corners_xy)[0]
print(num_of_fixations)

def encode():
    n_jumps = 16
    fixations = np.random.choice(num_of_fixations, size=n_jumps+1, replace=True)
    is_moved = np.r_[True, np.diff(fixations) != 0]
    fixations = fixations[is_moved]
    print(fixations)

    l23_history = []  # array for storing pattern pairs encoding
    for i, f in enumerate(fixations):
        poppy.sense_data(world, position=world.saliency_map.corners_xy[f])
        # poppy.cortex.V1.layers['L23'].Y_exc *= 0
        # if i == n_jumps / 2:
        #     poppy.cortex.V1.layers['L23'].Y_exc *= 0
        l23_history.append(poppy.cortex.V1.layers['L23'].cells)
    l23_history.pop(0)  # remove first jump from nowhere
    l23_history = np.vstack(l23_history)
    # print(np.dot(l23_history, l23_history.T))
    return l23_history[-1]


print(encode())


# print(np.count_nonzero(l23_history, axis=1))
# print(l23_history)
# print(np.dot(l23_history, l23_history.T))
# overlap = np.dot(initial_pairs, new_pairs.T)
# print(overlap)
# print np.sum(overlap)
# plt.imshow(overlap, vmin=0, vmax=max_overlap)
# plt.show()
