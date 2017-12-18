

from struct import unpack
import gzip
from numpy import zeros, uint8, float32
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

folder = 'data/'

train_img_file = folder + 'train-images-idx3-ubyte.gz'
train_lbl_file = folder + 'train-labels-idx1-ubyte.gz'

test_img_file = folder + 't10k-images-idx3-ubyte.gz'
test_lbl_file = folder + 't10k-labels-idx1-ubyte.gz'

def get_labeled_data(imagefile, labelfile, images_number):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')
    N = images_number
    # Get the data
    x = zeros((N, rows, cols), dtype=float32)  # Initialize numpy array
    y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i)
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (x, y)

def view_image(image, label=""):
    """View a single image."""
    plt.imshow(image, cmap=cm.gray)
    plt.show()


def load_images(images_number=100):
    filename = folder + 'digits' + str(images_number) + '.pickle'
    try:
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        images = data['images']
        labels = data['labels']

    except:
        images, labels = get_labeled_data(train_img_file, train_lbl_file, images_number)

        with open(filename, 'wb') as handle:
            pickle.dump(({'images': images,
                          'labels': labels}), handle)
        # print 'data_created'
    labels = labels.T[0]
    images = images.astype("uint32")
    labels = labels.astype("uint32")
    return images, labels


def load_test_images(images_number=100):
    filename = folder + 'digits_test' + str(images_number) + '.pickle'
    try:
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        images = data['images']
        labels = data['labels']
        # print 'data_loaded'

    except:
        images, labels = get_labeled_data(test_img_file, test_lbl_file, images_number)

        with open(filename, 'wb') as handle:
            pickle.dump(({'images': images,
                          'labels': labels}), handle)
        # print 'data_created'

    return images, labels

