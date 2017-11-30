# An idea to check how are different SDRs for digit zero and same digit with missed part
# Hypothesis: overlap is high. second SDR will add new active neurons that connects to whole pattern
# and do not encode missed data

import matplotlib.pyplot as plt
import numpy as np
from nupic.bindings.algorithms import SpatialPooler as SP
from tqdm import trange, tqdm

import load_mnist

uintType = "uint32"


# config
load_number = 100
inputDimensions = (28, 28)
columnDimensions = (50, 50)
num_training = 100

images, labels = load_mnist.load_images(images_number=load_number)
labels = labels.T[0]
images = images.reshape(load_number, -1)
images /= np.max(images, axis=1)[:, None]
images[images > 0.1] = 1
images[images < 0.1] = 0

numActiveColumnsPerInhArea = int(0.1 * np.prod(columnDimensions))
potentialRadius = int(0.1*np.prod(inputDimensions))


sp = SP(inputDimensions,
        columnDimensions,
        potentialRadius = potentialRadius,
        numActiveColumnsPerInhArea = numActiveColumnsPerInhArea,
        globalInhibition = True,
        synPermActiveInc = 0.01,
        synPermInactiveDec = 0.008,
        )


def digit_to_sdr(sp, digit, learn=True):
    # compute SDR for digit
    activeArray = np.zeros(columnDimensions, dtype=uintType)
    prev_activeArray = np.zeros(columnDimensions, dtype=uintType)
    difference = np.count_nonzero(prev_activeArray*activeArray)/float(np.count_nonzero(activeArray)+1)
    connectedSynapses = np.zeros(potentialRadius, dtype="int")
    potentialSynapses = np.zeros(potentialRadius, dtype="int")
    synapses_history = np.zeros((1, potentialRadius), dtype="int")
    # synapses_history = []
    if learn:
        while difference < 0.99:
            sp.compute(digit, learn, activeArray)
            difference = np.count_nonzero(prev_activeArray*activeArray)/float(np.count_nonzero(activeArray)+1)
            prev_activeArray = np.copy(activeArray)
            # print np.ravel_multi_index(np.nonzero(activeArray), columnDimensions)[:20]
            # print difference
            # print np.ravel_multi_index(np.nonzero(prev_activeArray*activeArray), columnDimensions)

            if difference > 1.5:
                active_column_index = np.ravel_multi_index(np.nonzero(activeArray), columnDimensions)[15]
                sp.getConnectedSynapses(1495, connectedSynapses)
                sp.getPotential(1495, potentialSynapses)
                print potentialSynapses
                print connectedSynapses
                # synapses_history = np.concatenate((synapses_history, connectedSynapses[None,:]))
                # print connectedSynapses
                # synapses_history.append(connectedSynapses)
                # print '1'
    else:
        sp.compute(digit, learn, activeArray)
        return activeArray
    return activeArray





if __name__ == '__main__':

    digit_label = 0
    zero = images[labels == digit_label][0].reshape((28,28)).astype(int)


    zero_broken = np.copy(zero)
    zero_broken[14:20, 5:15] = 0

    sdr_zero = digit_to_sdr(sp, zero, learn=True)
    # active_column_index = np.ravel_multi_index(np.nonzero(sdr_zero), columnDimensions)[15]
    # connectedSynapses = np.zeros((potentialRadius,), dtype="int")
    # sp.getConnectedSynapses(active_column_index, connectedSynapses)
    # print connectedSynapses

    sdr_zero_broken = digit_to_sdr(sp, zero_broken, learn=False)
    # sp.getConnectedSynapses(active_column_index, connectedSynapses)
    # print connectedSynapses

    print np.count_nonzero(sdr_zero * sdr_zero_broken)
    print np.count_nonzero(sdr_zero)

    # quit()
    plt.figure()
    plt.subplot(221)
    plt.imshow(sdr_zero, cmap='gray_r')
    plt.subplot(222)
    plt.imshow(sdr_zero_broken, cmap='gray_r')
    plt.subplot(223)
    plt.imshow(zero, cmap='gray_r')
    plt.subplot(224)
    plt.imshow(zero_broken, cmap='gray_r')
    plt.show()
    # plt.imshow(zero_broken)
    # plt.show()
    # pass
    # learn_identical()
    # learn_mnist()
    # learn_mnist(learn=False)
