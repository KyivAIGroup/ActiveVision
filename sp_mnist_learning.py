import load_mnist
import numpy as np
from nupic.bindings.algorithms import SpatialPooler as SP
import matplotlib.pyplot as plt
import sys

uintType = "uint32"


load_number = 100
images, labels = load_mnist.load_images(images_number=load_number)

labels = labels.T[0]
# images = images.reshape(load_number, -1)
# images /= np.max(images, axis=1)[:, None]
# images[images > 0.1] = 1
# images[images < 0.1] = 0

# Visualize images from the same class
# ones = images[labels == lbl]
# fig = plt.figure()
# num_images = ones.shape[0]
# #
# for i, img in enumerate(ones):
#     plt.subplot(int(np.sqrt(num_images))+1, int(np.sqrt(num_images))+1, i+1)
#     plt.imshow(img)
# plt.show()
# sys.exit()


inputDimensions = np.array((28,28))

columnDimensions = (50,50)

sp = SP(inputDimensions,
        columnDimensions,
        potentialRadius = int(0.1*inputDimensions.prod()),
        numActiveColumnsPerInhArea = int(0.02*50*50),
        globalInhibition = True,
        synPermActiveInc = 0.05,
        synPermInactiveDec = 0.008,
        wrapAround=False,)


lbl = 0
num_training = 100

for k in range(num_training):
    activeArray = np.zeros((50,50), dtype=uintType)
    for i, image in enumerate(images):
        activeArray *= 0
        sp.compute(image, True, activeArray)
    print k


num_images_to_test = images[labels == lbl].shape[0]
activeArray = np.zeros((50,50), dtype=uintType)
SDR = np.zeros((num_images_to_test, 50, 50))
for k, im in enumerate(images[labels == lbl]):
    sp.compute(im, False, activeArray)
    SDR[k] = activeArray

overlap = np.array([[np.count_nonzero(SDR[j] * SDR[i]) for i in range(SDR.shape[0])] for j in range(SDR.shape[0])], dtype=int)
np.fill_diagonal(overlap, 0)
print overlap
print overlap.shape
print np.mean(overlap)
plt.imshow(overlap)
plt.colorbar()
plt.show()


fig = plt.figure()
plt.subplot(221)
plt.imshow(SDR[-1], cmap='gray_r')
plt.subplot(223)
plt.imshow(images[labels == lbl][-1], cmap='gray_r')
plt.subplot(222)
plt.imshow(SDR[-2], cmap='gray_r')
plt.subplot(224)
plt.imshow(images[labels == lbl][-2], cmap='gray_r')
plt.show()
