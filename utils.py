import cv2
import numpy as np


def cv2_step():
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        quit()


def gaussian_kernel(size, sigma):
    x = np.linspace(-size//2, size//2, num=size, endpoint=True)
    kernel_1d = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d /= np.max(kernel_2d)
    return kernel_2d


def apply_blur(image):
    im_blurred = cv2.GaussianBlur(image, ksize=(7, 7), sigmaX=5)
    weight = gaussian_kernel(image.shape[0], sigma=3)
    im_blurred = image * weight + im_blurred * (1.0 - weight)
    im_blurred = im_blurred.astype(np.uint8)
    return im_blurred
