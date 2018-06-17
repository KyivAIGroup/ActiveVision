import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

from core.cortex import Cortex
from utils.utils import cv2_step, apply_blur
from utils import load_mnist


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
    n_bits_active = cortex.V1.layers[layer].get_sparse_bits_count()
    assert sum(sdrs[0]) == sum(sdrs[1]) == n_bits_active
    overlap = np.dot(sdrs[0], sdrs[1]) / float(n_bits_active)
    return overlap


def test_translate_display(label_interest=5, display=True):
    images, labels = load_mnist.load_images(images_number=1000)
    cortex = Cortex()
    overlaps = []
    translation_x = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    for im in tqdm(images[labels == label_interest], desc="Translation test"):
        im = apply_blur(im)
        # cortex.compute(im, (0,0,0), display=True)
        overlap = compute_translation_overlap(cortex, im, translation_x, display=display)
        overlaps.append(overlap)
        if display:
            cv2_step()
    print("Overlap mean={:.4f} std={:.4f}".format(np.mean(overlaps), np.std(overlaps)))


def test_translate_plot(max_dist=5, layer='L4'):
    images, labels = load_mnist.load_images(images_number=100)
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
    plt.plot(np.arange(len(overlaps)), overlaps, label="retina {}".format(cortex.retina.shape))
    plt.xlabel("Translation distance, px")
    plt.ylabel("Overlap with origin image")
    plt.title("{} SDR stability test: image translation".format(layer))
    plt.legend()
    plt.grid()
    plt.savefig("translation.png")
    plt.show()


def test_inner_outer_overlap(layer='L4'):
    images, labels = load_mnist.load_images(images_number=100)
    cortex = Cortex()
    prepare_lists = lambda: [[] for digit in range(10)]
    examples_sdr = prepare_lists()
    for img, label in zip(tqdm(images, desc="Inner- & outer-examples overlap test"), labels):
        cortex.compute(img, vector=(0, 0, 0))
        sdr = cortex.V1.layers[layer].cells.copy()
        examples_sdr[label].append(sdr)
    overlaps_outer = prepare_lists()
    overlaps_inner = prepare_lists()
    n_bits_active = cortex.V1.layers[layer].get_sparse_bits_count()
    for label, sdrs_same in enumerate(examples_sdr):
        pairwise_same = np.dot(sdrs_same, np.transpose(sdrs_same)) / float(n_bits_active)
        pairwise_same_idx = np.triu_indices_from(pairwise_same, k=1)
        overlaps_inner[label] = pairwise_same[pairwise_same_idx]
        for label_other in range(label+1, 10):
            sdrs_other = examples_sdr[label_other]
            other_idx = np.random.choice(len(sdrs_other), size=max(len(sdrs_other) // 10, 1), replace=False)
            sdrs_other = np.take(sdrs_other, other_idx, axis=0)
            pairwise_other = np.dot(sdrs_same, np.transpose(sdrs_other)) / float(n_bits_active)
            pairwise_other_idx = np.triu_indices_from(pairwise_other, k=0)
            overlaps = pairwise_other[pairwise_other_idx]
            overlaps_outer[label_other].append(overlaps)
            overlaps_outer[label].append(overlaps)
    width = 0.35
    plt.bar(np.arange(10), [np.mean(ovlp) for ovlp in overlaps_inner],
            yerr=[np.std(ovlp) for ovlp in overlaps_inner], label="inner", width=width)
    plt.bar(np.arange(10) + width, [np.mean(np.vstack(ovlp)) for ovlp in overlaps_outer],
            yerr=[np.std(np.vstack(ovlp)) for ovlp in overlaps_outer], label="outer", width=width)
    plt.xticks(np.arange(10))
    plt.title("{} SDR stability test: inner- & outer-examples overlap".format(layer))
    plt.xlabel("Label")
    plt.ylabel("Overlap")
    plt.legend()
    plt.savefig("inner-outer-overlap.png")
    plt.show()


if __name__ == '__main__':
    # test_translate_plot()
    test_inner_outer_overlap()
    # test_translate_display(label_interest=5, display=False)
