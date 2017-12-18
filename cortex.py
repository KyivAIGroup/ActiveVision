# Test of an idea with autoassociation. Image - L4(sparse) - L23
#                                                        L23 /


import load_mnist
import matplotlib.pyplot as plt
import numpy as np


class Cortex:
    def __init__(self):
        self.bottom_input = np.zeros((28, 28))
        self.bottom_input_shape = self.bottom_input.shape
        self.L4_shape = (50, 50)
        self.L4 = np.zeros(self.L4_shape)
        self.L23_shape = (20, 20)
        self.L23 = np.zeros(self.L23_shape, dtype='uint32')
        self.weights_input_L4 = np.random.rand(self.L4.size, self.bottom_input.size)
        self.weights_L4_L23 = np.random.rand(self.L23.size, self.L4.size)

        self.clusters_L4_L23 = np.zeros((self.L23.size, self.L4.size, 5), dtype=int)
        self.clusters_L23_L23 = np.zeros((self.L23.size, self.L23.size, 5), dtype=int)

        self.sparsity = 0.1
        self.output = None


    def compute(self, bottom_input):
        self.L4 = np.dot(self.weights_input_L4, bottom_input.flatten())
        self.L4 = self.kWTA(self.L4, self.sparsity)

        self.L23 = np.dot(self.weights_L4_L23, self.L4)
        self.L23 /= np.max(self.L23)
        self.L23 += self.cluster_activation(self.clusters_L4_L23, self.L4)
        self.L23 = self.kWTA(self.L23, self.sparsity)
        # add lateral excitation
        self.L23 += self.cluster_activation(self.clusters_L23_L23, self.L23)
        # print np.sort(self.L23)[-40:]
        self.L23 = self.kWTA(self.L23, self.sparsity)

        self.associate(self.L23, self.L4, self.clusters_L4_L23, 0.1)
        self.associate(self.L23, self.L23, self.clusters_L23_L23, 0.1)


        self.L4 = self.L4.reshape(self.L4_shape)
        self.L23 = self.L23.reshape(self.L23_shape)
        # print np.count_nonzero(self.clusters_L4_L23)

    def cluster_activation(self, clusters, input):
        dst = clusters[:, np.nonzero(input)[0]]
        # find counts of input*distance to see if there is a cluster
        data = [np.unique(dt[np.nonzero(dt)[0], np.nonzero(dt)[1]], return_counts=True) for dt in dst]
        # find max in each count to see if there is cluster (2 if there is a cluster)
        data2 = np.array([np.max(counts) if counts.size else 0 for counts in np.array(data)[:, 1]])
        # cells = data2 >= np.sort(data2)[-int(self.size*self.activation)]
        cells = (data2 == 2).astype(int)
        # print np.count_nonzero(cells)
        return cells



    def associate(self, output, input, clusters, probability):
        # clusters - 3d matrix that stored synapse membership to clusters
        self.cluster_size = 2
        if np.count_nonzero(input):
            for i in np.nonzero(output)[0]:
                # I will create clusters not whithin all active cells but just with part
                if np.random.rand() < probability:
                    # self.clusters[i] is a matrix, where return array of indices [(rows), (cols)]
                    overlap = np.intersect1d(np.where(clusters[i] == 0)[0], np.nonzero(input)[0])
                    # overlap show which synapses have free clusters to which I can write
                    # select cluster_size active pre neurons
                    if overlap.size >= self.cluster_size:
                        pre = np.random.choice(overlap, self.cluster_size, replace=False)
                        pre_tree = []
                        for k in range(self.cluster_size):
                            # this is to select free cluster among all synapses pairs
                            pre_tree.append(np.random.choice(np.where(clusters[i, pre[k]] == 0)[0]))
                        clusters[i, pre, pre_tree] = np.max(clusters[i]) + 1  # max +1 is number of cluster
                    else:
                        print 'full'

    def kWTA(self, vector, k):
        # k - percent of active neurons to leave
        ar = vector
        ar[ar.argsort()[:-int(ar.size * k)]] = 0
        ar[ar.argsort()[-int(ar.size * k):]] = 1
        # ar[ar < sorted(ar)[-int(ar.size * k)]] = 0
        # ar[ar > 0] = 1
        return ar.astype(int)




class World():
    def __init__(self):
        self.visual_field_size = (50, 50)
        self.visual_field = np.zeros(self.visual_field_size, dtype=int)

        self.agent = 0

    def add_visual_data(self, data, position):
        # data is a two dimensional array = image
        x, y = position
        x_m, y_m = data.shape
        self.visual_field[x:x+x_m, y:y+y_m] = data



class Agent():
    def __init__(self):
        self.world = 0
        self.input_size_angle = (30, 30)
        # x,y,z. XY plane of an image.
        # do not change z for now, it adjusted with self.visual_field_angle to give 28x28 patch of visual input

        self.position = [10, 10, 25]  # relative to the world
        self.input_size_pixel = (2 * self.position[2] * np.tan(np.deg2rad(self.input_size_angle))).astype(int)


    def sense_data(self):
        x, y, z = self.position
        x_m, y_m = self.input_size_pixel
        return self.world.visual_field[x:x+x_m, y:y+y_m]



load_number = 100
images, labels = load_mnist.load_images(images_number=load_number)

labels = labels.T[0]
images = images.astype(int)
# images = images.reshape(load_number, -1)
# images /= np.max(images, axis=1)[:, None]
# images[images > 0.1] = 1
# images[images < 0.1] = 0

images_and_labels = list(zip(images, labels))


flat_mnist_world = World()
flat_mnist_world.add_visual_data(images[0], position=(10, 10))

poppy = Agent()
poppy.world = flat_mnist_world
poppy.brain = Cortex()


poppy_init_position = np.copy(poppy.position)

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg


class MyView(pg.GraphicsWindow):
    def __init__(self):
        super(MyView, self).__init__()
        l = pg.GraphicsLayout(border=(100,100,100))
        self.setCentralWidget(l)
        self.show()
        self.setWindowTitle('pyqtgraph example: GraphicsLayout')
        self.resize(800,600)
        vb = l.addViewBox(lockAspect=True)
        self.img = pg.ImageItem()
        vb.addItem(self.img)
        vb2 = l.addViewBox(lockAspect=True)
        self.img2 = pg.ImageItem()
        vb2.addItem(self.img2)
        vb3 = l.addViewBox(lockAspect=True)
        self.img3 = pg.ImageItem()
        vb3.addItem(self.img3)
        l.nextRow()
        text = 'state of the network'
        self.label = l.addLabel(text, colspan=2)


        # Idea to plot Overlap SDR and average overlap
        l.nextRow()
        l2 = l.addLayout(colspan=3)
        l2.setContentsMargins(10, 10, 10, 10)
        l2.addLabel('Average Overlap', angle=-90)
        self.plot = l2.addPlot(colspan=2)
        self.curve = self.plot.plot()
        self.plot.setYRange(0, 1)

        # self.sp_state = np.zeros(poppy.sp.getColumnDimensions(), dtype="uint32")
        self.prev_sp_state = np.zeros(poppy.brain.L23_shape, dtype="uint32")
        self.average_difference = 0.4
        self.data = np.zeros(100)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)
        self.counter = 0

    def update(self):
        self.counter += 1

        # sp_state = poppy.form_sdr(learn=False)
        poppy.brain.compute(poppy.sense_data())
        # print self.prev_sp_state.shape
        difference = np.count_nonzero(self.prev_sp_state * poppy.brain.L23)/float(np.count_nonzero(poppy.brain.L23)+1)
        self.average_difference = self.average_difference + 0.01 * (difference - self.average_difference)
        self.prev_sp_state = np.copy(poppy.brain.L23)

        self.img.setImage(poppy.sense_data()[::-1, :].T)
        self.img2.setImage(poppy.brain.L23)
        self.img3.setImage(poppy.brain.weights_L4_L23)
        amplitude = 2
        poppy.position[0] = poppy_init_position[0] + np.random.randint(-amplitude, amplitude)
        poppy.position[1] = poppy_init_position[1] + np.random.randint(-amplitude, amplitude)

        output_text = 'Iteration: ' + str(self.counter) + '; Overlap of SDR:' + str(difference*100)[:4] + '%; ' +  'Average:' + str(self.average_difference*100)[:4]
        self.label.setText(output_text)


        # self.data[0] = self.average_difference
        # self.data = np.roll(self.data, -1)
        # self.curve.setData(self.data)




app = QtGui.QApplication([])
view = MyView()

app.exec_()