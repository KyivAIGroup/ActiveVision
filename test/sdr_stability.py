import cv2
import numpy as np
import load_mnist
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from core.cortex import Cortex
from utils import cv2_step


def compute_translation_overlap(cortex, retina_img, translation_matrix, layer='L4', display=False):
    sdrs = []
    h, w = retina_img.shape[:2]
    img_translated = cv2.warpAffine(retina_img, translation_matrix, (w, h))
    for img_try, title in zip((retina_img, img_translated), ("Orig", "Translated")):
        cortex.compute(img_try, vector=(0, 0, 0))
        if display:
            cortex.V1.layers[layer].display(winname=title)
        sdr = cortex.V1.layers[layer].cells.copy()
        sdrs.append(sdr)
    n_bits_active = sum(sdrs[0])
    overlap = np.dot(sdrs[0], sdrs[1]) / float(n_bits_active)
    return overlap


def test_translate(label_interest=5, display=False):
    images, labels = load_mnist.load_images(images_number=1000)
    cortex = Cortex()
    overlaps = []
    translation_x = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    for im in tqdm(images[labels == label_interest], desc="Translation test"):
        overlap = compute_translation_overlap(cortex, im, translation_x, display=display)
        if display:
            cv2_step()
        overlaps.append(overlap)
    print("Overlap mean={:.4f} std={:.4f}".format(np.mean(overlaps), np.std(overlaps)))


def test_translate_plot(max_dist=5, layer='L4'):
    images, labels = load_mnist.load_images(images_number=1000)
    cortex = Cortex()
    overlaps = np.zeros(shape=int(np.sqrt(2 * max_dist ** 2))+1, dtype=np.float32)
    counts = np.zeros(shape=overlaps.shape, dtype=np.int32)
    for dx in trange(max_dist+1, desc="Translate image distances"):
        for dy in range(max_dist+1):
            dist = int(np.sqrt(dx ** 2 + dy ** 2))
            translation_x = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
            for im_orig in images:
                overlap = compute_translation_overlap(cortex, im_orig, translation_x, layer=layer)
                overlaps[dist] += overlap
            counts[dist] += len(images)
    overlaps /= counts
    plt.plot(np.arange(len(overlaps)), overlaps, label="retina {}".format(cortex.retina.cells.shape))
    plt.xlabel("Translation distance, px")
    plt.ylabel("Overlap with origin image")
    plt.title("{} SDR stability due to image translation".format(layer))
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    test_translate_plot()
