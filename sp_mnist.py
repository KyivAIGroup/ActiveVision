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

images_and_labels = list(zip(images, labels))

# print np.count_nonzero(labels==2)
# print images[labels==2].shape
# sys.exit()

inputDimensions = np.array((28,28))

columnDimensions = (50,50)

sp = SP(inputDimensions,
        columnDimensions,
        potentialRadius = int(0.1*inputDimensions.prod()),
        numActiveColumnsPerInhArea = int(0.02*50*50),
        globalInhibition = True,
        synPermActiveInc = 0.01,
        synPermInactiveDec = 0.008,
        wrapAround=False,)


lbl = 0
num_training = 300
SDR = np.zeros((num_training, 50,50) )

# Learning from the same image
activeArray = np.zeros((50,50), dtype=uintType)
for k in range(num_training):
    activeArray *= 0
    sp.compute(images[lbl], True, activeArray)
    SDR[k] = activeArray
    print k


overlap = np.array([[np.count_nonzero(SDR[j] * SDR[i]) for i in range(num_training)] for j in range(num_training)], dtype=int)

plt.imshow(overlap)
plt.colorbar()
plt.show()


fig = plt.figure()
plt.subplot(221)
plt.imshow(SDR[-1], cmap='gray_r')
plt.subplot(223)
plt.imshow(images[lbl], cmap='gray_r')
plt.subplot(222)
plt.imshow(SDR[-2], cmap='gray_r')
plt.subplot(224)
plt.imshow(images[lbl], cmap='gray_r')
plt.show()



sys.exit()
for i, image in enumerate(images[labels==lbl]):
    activeArray *= 0
    sp.compute(image, True, activeArray)
    SDR[i] = activeArray



print np.count_nonzero(SDR[0])
print np.count_nonzero(SDR[1])
print np.count_nonzero(SDR[0]*SDR[1])

fig = plt.figure()
plt.subplot(221)
plt.imshow(SDR[0], cmap='gray_r')
plt.subplot(223)
plt.imshow(images[labels==lbl][0].reshape(28,28), cmap='gray_r')
plt.subplot(222)
plt.imshow(SDR[1], cmap='gray_r')
plt.subplot(224)
plt.imshow(images[labels==lbl][1].reshape(28,28), cmap='gray_r')
plt.show()



# plt.imshow(activeArray)
# plt.show()
# print activeArray
# print activeArray.nonzero()


# print labels, images