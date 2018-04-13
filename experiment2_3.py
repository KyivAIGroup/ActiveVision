# Here I will try to store pairs of features for different fixations
# and for new entry look at memory for match:
# part 1: for different order of fixation of the same digit

# here I will try to apply learning, and compare representation for different instances of two classes
# as expected, weights overlearned to represent every class with the same cells. Do I need to send error signal?

# At first I need to add motor signal. It should separate features
import matplotlib.pyplot as plt
import numpy as np

import load_mnist
from core.agent import Agent
from core.world import World



# np.random.seed(26)
images, labels = load_mnist.load_images(images_number=100)
world = World()
poppy = Agent()

label_interest = 0
zeros = images[labels == label_interest]
ones = images[labels == 1]
zero1 = zeros[1]
zero2 = zeros[2]
one1 = ones[0]

world.add_image(zero1, position=(10, 10))

plt.imshow(world.saliency_map.display())
plt.show()

def encode():
    n_jumps = 16
    num_of_fixations = np.shape(world.saliency_map.corners_xy)[0]
    print(num_of_fixations)
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


zero1_y = encode()
# print(zero1_y)

world.add_image(one1, position=(10, 10))

plt.imshow(world.saliency_map.display())
plt.show()
poppy.cortex.V1.layers['L23'].Y_exc *= 0

zero2_y = encode()


print(np.count_nonzero(zero1_y * zero2_y))

# print(np.count_nonzero(l23_history, axis=1))
# print(l23_history)
# print(np.dot(l23_history, l23_history.T))
# overlap = np.dot(initial_pairs, new_pairs.T)
# print(overlap)
# print np.sum(overlap)
# plt.imshow(overlap, vmin=0, vmax=max_overlap)
# plt.show()
