# here I will test an idea of tracking vector relative to center of an object, rather than to previous position

import matplotlib.pyplot as plt
import numpy as np

import load_mnist
from core.agent import Agent
from core.world import World
from utils import cv2_step

# np.random.seed(26)
images, labels = load_mnist.load_images(images_number=100)
world = World()
poppy = Agent()


label_interest = 0
zeros = images[labels == label_interest]
ones = images[labels == 1]
zero1 = zeros[0]
zero2 = zeros[2]
one1 = ones[0]



world.add_image(zero1, position=(10, 10))




image_center = (24, 24)
# plt.imshow(world.image)
# plt.show()

# quit()

def encode_feature(image, location):
    world.add_image(image, position=(10, 10))
    poppy.sense_data(world, position=world.saliency_map.corners_xy[location])
    plt.imshow(world.saliency_map.display())
    plt.show()
    plt.imshow(poppy.cortex.retina.cells.reshape(10, 10))
    plt.show()
    # return poppy.cortex.V1.layers['motor_direction'].cells
    # return poppy.cortex.V1.layers['motor_amplitude'].cells
    # return poppy.cortex.V1.layers['L4m'].cells
    return poppy.cortex.V1.layers['L4'].cells
    # return poppy.cortex.V1.layers['L23'].cells
    # return poppy.cortex.retina.cells

# this is to check if similar features in similar location encoded with similar codes.
# Now it is, after I reduces bin size for scalar encoder. Do I need to make recognition at this stage
# for one feature?  Like opositive curvature shi=ould be encoded almoust with the same vector, or should I add
# another layer that will have even closer representation? It is question of generalization, human with autism will
# treat every curvature as unique, normal will say that it is the same. How to add this generalization?
# Do I need to think at direction of probability distribution and number of alternatives?

enc1 = encode_feature(zero1, 0)
# enc2 = encode_feature(zero2, 0)
# enc2 = encode_feature(zero2, 0)
enc2 = encode_feature(one1, 1)
print np.count_nonzero(enc1 * enc2)
quit()


def encode():
    n_jumps = 5
    num_of_fixations = np.shape(world.saliency_map.corners_xy)[0]
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
    print(np.dot(l23_history, l23_history.T))
    return l23_history[-1]


zero1_y = encode()


# world.add_image(zero2, position=(10, 10))
world.add_image(one1, position=(10, 10))

plt.imshow(world.saliency_map.display())
plt.show()

poppy.cortex.V1.layers['L23'].Y_exc *= 0


zero2_y = encode()

print(np.count_nonzero(zero1_y * zero2_y))