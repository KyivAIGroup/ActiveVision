# Here I will try to integrate features for different fixations
# and compare resulting sequence representation for:
# part 1: for different order of fixation of the same digit
# part 2": for different digits from the same class



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

world.add_image(zero1)
poppy.init_world(world)

# poppy.saliency_map.display()
# cv2_step()

num_of_fixations = np.shape(poppy.saliency_map.corners_xy)[0]
# print poppy.saliency_map.corners_xy
# print num_of_fixations
# plt.imshow(world.image)
# plt.show()

for i in range(num_of_fixations):
    poppy.sense_data(world)
    # plt.imshow(poppy.cortex.retina.cells)
    # plt.show()
    # print np.nonzero(poppy.cortex.retina.cells.flatten()),
    # print np.nonzero(poppy.cortex.V1.layers['L4'].cells),

    # cv2_step()
    # print poppy.saliency_map.corners_xy[poppy.saliency_map.curr_id]
    # print poppy.cortex.retina.cells

common_pattern = poppy.cortex.V1.layers['L23'].kWTA(np.sum(poppy.cortex.V1.layers['L23'].Y, axis=0),
                                                    poppy.cortex.V1.layers['L23'].sparsity_2)
# print np.nonzero(common_pattern)


print("Second")
poppy.position = np.array([10, 10, 25])
poppy.last_position = np.copy(poppy.position)
poppy.init_world(world)

# poppy.saliency_map.corners_xy = np.random.permutation(poppy.saliency_map.corners_xy)

poppy.cortex.reset()
# plt.imshow(world.image)
# plt.show()
poppy.cortex.V1.layers['L23'].reset_integration_params()

# print poppy.saliency_map.corners_xy


for i in range(num_of_fixations):
    poppy.sense_data(world)
    # plt.imshow(poppy.cortex.retina.cells)
    # plt.show()
    # print np.nonzero(poppy.cortex.retina.cells.flatten()),
    # print np.nonzero(poppy.cortex.V1.layers['L4'].cells),

# plt.imshow(world.image)
# plt.show()
common_pattern2 = poppy.cortex.V1.layers['L23'].kWTA(np.sum(poppy.cortex.V1.layers['L23'].Y, axis=0),
                                                            poppy.cortex.V1.layers['L23'].sparsity_2)
# print np.nonzero(common_pattern2)


print np.sum(common_pattern * common_pattern2) / float(np.count_nonzero(common_pattern))